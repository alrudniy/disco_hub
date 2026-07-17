"""
dh2.teacher_merge -- combine two cross-encoder teachers (+ optional LLM judge) into a
single graded relevance label per (query, candidate), following the spec cascade:

  1. score with BGE and Qwen  -> two probabilities in [0,1]
  2. HIGH AGREEMENT (|bge-qwen| small AND same side of mid) -> accept the mean automatically
  3. GRAY ZONE (either score mid-range, or the teachers disagree) -> claude-sonnet-4-6 judge
  4. still unresolved / LLM unavailable -> MASK (label_role="masked"), never force a binary

Also computes the positive-aware false-negative FLAG (reported, not auto-applied): a
candidate is flagged if teacher(candidate) >= teacher(designated_positive) * (1-margin).
This is the LLM-adjudicated recomputation of the 46.7% audit — documented as
"LLM-adjudicated, not human-calibrated".

Merged relevance -> grade via GRADE thresholds. label_role in:
  designated_positive | secondary_positive | ambiguous | hard_negative | easy_negative | masked
"""
from __future__ import annotations

from dataclasses import dataclass

from dh2 import config2 as C


@dataclass
class MergedLabel:
    query_id: str
    query: str
    document_id: str
    source: str
    bge_score: float
    qwen_score: float | None
    llm_grade: int | None
    llm_relevance: float | None
    relevance: float          # final merged [0,1]
    grade: int                # final graded 0-3
    label_role: str
    training_weight: float
    teacher_agreement: str    # high | grayzone | disagree
    is_designated_positive: bool
    false_negative_flag: bool
    masked: bool
    evidence: str


def _agreement(bge: float, qwen: float | None, tc: C.TeacherConfig) -> str:
    if qwen is None:
        return "grayzone"
    mid = 0.5
    same_side = (bge >= mid) == (qwen >= mid)
    if abs(bge - qwen) <= tc.agree_high_delta and same_side:
        return "high"
    return "disagree"


def _in_grayzone(bge: float, qwen: float | None, tc: C.TeacherConfig) -> bool:
    def gray(x): return tc.grayzone_lo <= x <= tc.grayzone_hi
    if qwen is None:
        return True
    return gray(bge) or gray(qwen) or abs(bge - qwen) > tc.agree_high_delta


def _grade_of(relevance: float, gc: C.GradeConfig) -> int:
    if relevance >= gc.g3:
        return 3
    if relevance >= gc.g2:
        return 2
    if relevance >= gc.g1:
        return 1
    return 0


def _role(grade: int, is_pos: bool, masked: bool) -> str:
    if masked:
        return "masked"
    if is_pos:
        return "designated_positive"
    if grade >= 2:
        return "secondary_positive"
    if grade == 1:
        return "ambiguous"
    return "easy_negative"  # grade 0; hard vs easy distinguished by caller if desired


def needs_llm(bge_score: float, qwen_score: float | None, is_designated_positive: bool,
              tc: C.TeacherConfig = C.TEACHER) -> bool:
    """True iff merge_one would route this pair to the LLM judge (gray-zone, non-positive,
    teachers not in high agreement). Used by the parallel prefetch in p0_3 so the LLM calls
    can be fired concurrently BEFORE the serial merge pass."""
    if is_designated_positive:
        return False
    if _agreement(bge_score, qwen_score, tc) == "high":
        return False
    return _in_grayzone(bge_score, qwen_score, tc)


def merge_one(*, query_id: str, query: str, document_id: str, source: str,
              doc_title: str, doc_text: str, is_designated_positive: bool,
              bge_score: float, qwen_score: float | None,
              positive_teacher_score: float | None,
              llm=None, tc: C.TeacherConfig = C.TEACHER,
              gc: C.GradeConfig = C.GRADE) -> MergedLabel:
    """Merge teachers (+ optional llm=TeacherLLM) into a graded label for one candidate."""
    agreement = _agreement(bge_score, qwen_score, tc)
    llm_grade = None
    llm_rel = None
    evidence = ""
    masked = False

    if is_designated_positive:
        # trust the label; still record teacher scores. grade 3 by construction.
        relevance = max(bge_score, qwen_score or 0.0, 0.85)
        grade = 3
    elif agreement == "high":
        relevance = (bge_score + (qwen_score or bge_score)) / 2.0
        grade = _grade_of(relevance, gc)
    else:
        # gray zone / disagreement -> LLM judge if available, else mask
        if llm is not None and _in_grayzone(bge_score, qwen_score, tc):
            try:
                j = llm.graded_relevance(query, doc_title, doc_text, source=source)
                # BUG #10: an unparseable reply is NOT a verdict. It used to arrive here
                # as grade=1/relevance=0.5 and enter the labels as a real "tangential"
                # judgment. Mask it -- the same treatment an unavailable judge gets.
                from dh2.llm_client import TeacherLLM
                if TeacherLLM.is_unparseable(j):
                    raise ValueError("unparseable_llm_reply")
                llm_grade = j["grade"]
                llm_rel = j["relevance"]
                evidence = j.get("evidence", "")
                # blend LLM with teachers: LLM is the tie-breaker, weighted toward it
                relevance = 0.6 * llm_rel + 0.4 * ((bge_score + (qwen_score or bge_score)) / 2)
                grade = llm_grade
            except Exception:
                # LLM unavailable / budget exhausted / bad reply -> mask (never crash the run)
                masked = True
                llm_grade = None
                relevance = (bge_score + (qwen_score or bge_score)) / 2.0
                grade = _grade_of(relevance, gc)
        else:
            masked = True
            relevance = (bge_score + (qwen_score or bge_score)) / 2.0
            grade = _grade_of(relevance, gc)

    # false-negative flag (reported): non-positive scoring ~as high as the positive
    fn_flag = False
    if (not is_designated_positive) and positive_teacher_score:
        bar = positive_teacher_score * (1.0 - tc.posaware_margin)
        fn_flag = max(bge_score, qwen_score or 0.0) >= bar

    role = _role(grade, is_designated_positive, masked)
    # training weight: full for confident labels, downweight ambiguous, zero masked
    if masked:
        weight = 0.0
    elif role in ("designated_positive", "secondary_positive"):
        weight = 0.85 if role == "secondary_positive" else 1.0
    elif role == "ambiguous":
        weight = 0.3
    else:
        weight = 1.0  # confident negative is useful signal

    return MergedLabel(
        query_id=query_id, query=query, document_id=document_id, source=source,
        bge_score=round(bge_score, 4),
        qwen_score=None if qwen_score is None else round(qwen_score, 4),
        llm_grade=llm_grade, llm_relevance=None if llm_rel is None else round(llm_rel, 4),
        relevance=round(relevance, 4), grade=grade, label_role=role,
        training_weight=round(weight, 3), teacher_agreement=agreement,
        is_designated_positive=is_designated_positive, false_negative_flag=fn_flag,
        masked=masked, evidence=evidence)
