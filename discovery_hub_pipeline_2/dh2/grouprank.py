"""
dh2.grouprank -- Diver-GroupRank-32B groupwise reranking (RASC Stage 4).

Pointwise reranking judges each document independently, so it cannot express "this patent
and that trial cover the same mechanism, but the trial is Phase 3 and the patent expires
next year." Groupwise scoring shows the model several documents at once and asks it to
score them together, which is where the distinctions that matter to a scout live: exact
mechanism, disease stage, modality, route, biomarker requirement, development maturity,
commercial usefulness.

STATUS: feature-flagged OFF (DH_ENABLE_GROUPRANK=0) and GATED. Strong published results,
unvalidated transfer to this corpus. It ships enabled only when it beats Qwen-only
ordering on independently judged low-overlap utility, parses >=99% of the time, and stays
inside the latency budget (config2.CascadeGates).

Design:
  * Serving       vLLM, BF16, tensor_parallel_size=4 (32B does not fit one 80GB card
                  with a 16K context and useful throughput).
  * Groups        random partitions of the Qwen top-40 into groups of 10.
  * Repeats       2 independent partitions; a document's score is its mean across
                  appearances. One partition makes a document's score depend on which
                  9 documents it happened to land with; averaging over independent
                  partitions is what makes the score a property of the document.
  * Parsing       strict. The returned id set must equal the requested id set exactly.
  * Failure       FAIL CLOSED -- any parse failure, id mismatch, or exception returns the
                  Qwen ordering untouched. A groupwise reranker that silently drops or
                  invents documents is worse than no groupwise reranker.

Determinism: partitions are seeded (CascadeConfig.group_rank_seed) so an eval is
reproducible; temperature is 0.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Sequence

from dh2 import config2 as C

SYSTEM_PROMPT = (
    "You are a pharmaceutical and biotechnology scouting analyst. You will be given a "
    "scouting need and a numbered set of documents. Score how practically useful each "
    "document is for satisfying that need."
)

RUBRIC = (
    "Score each document from 0 to 10 (integers only):\n"
    "  10-9  directly actionable: disease/stage, target/mechanism, and modality/route all "
    "match, and the asset is at a stage the scout can act on.\n"
    "  8-6   clearly relevant: same scientific area and intent, with a minor mismatch in "
    "stage, population, route, or maturity.\n"
    "  5-3   tangential: related target, pathway, or disease area, but it does not satisfy "
    "the stated need.\n"
    "  2-0   irrelevant.\n"
    "\n"
    "Compare the documents against EACH OTHER, not just against the need. When two "
    "documents both match, prefer the one with the more exact mechanism, the more "
    "appropriate disease stage, the required biomarker or population, and the greater "
    "development maturity or commercial usefulness.\n"
    "\n"
    "Legacy names, gene/protein aliases, drug development codes, class names, and patent "
    "terminology are semantic equivalents -- score meaning, never shared words. A document "
    "that reuses the need's vocabulary but does not meet the underlying scientific or "
    "commercial requirement scores low."
)


@dataclass
class GroupRankResult:
    """Per-document groupwise outcome. `appearances` is how many groups scored it."""
    scores: dict[str, float] = field(default_factory=dict)      # doc_id -> mean 0-10
    appearances: dict[str, int] = field(default_factory=dict)
    groups_total: int = 0
    groups_parsed: int = 0
    ok: bool = False                                            # False => use qwen_order

    @property
    def parse_success(self) -> float:
        return self.groups_parsed / self.groups_total if self.groups_total else 0.0


def make_groups(doc_ids: Sequence[str], group_size: int, repeats: int,
                seed: int) -> list[list[str]]:
    """`repeats` independent random partitions of doc_ids into groups of ~group_size.

    Each repeat shuffles the full list and chunks it, so every document appears exactly
    once per repeat (i.e. `repeats` times total) and group membership is independent
    across repeats. A trailing chunk smaller than group_size is kept rather than dropped:
    dropping it would silently deny some documents a score, and fail-closed logic would
    then discard the whole pass.
    """
    ids = list(doc_ids)
    if not ids:
        return []
    rng = random.Random(seed)
    groups: list[list[str]] = []
    for r in range(max(1, repeats)):
        shuffled = ids[:]
        rng.shuffle(shuffled)
        for i in range(0, len(shuffled), group_size):
            chunk = shuffled[i:i + group_size]
            if chunk:
                groups.append(chunk)
    return groups


def build_group_prompt(query: str, group: Sequence[str],
                       views: dict[str, str]) -> str:
    """Render one group as a single scoring prompt.

    Documents are labelled D1..Dn rather than by doc_id: raw ids like "us:10485802" leak
    the source and the identifier into the prompt, which is free lexical signal we are
    specifically trying not to give the model. The mapping back is positional.
    """
    lines = [RUBRIC, "", f"Scouting need: {query}", "", "Documents:"]
    for i, did in enumerate(group, start=1):
        view = (views.get(did) or "").strip() or "(no content available)"
        lines.append(f"\n[D{i}]\n{view}")
    labels = ", ".join(f'"D{i}"' for i in range(1, len(group) + 1))
    lines += [
        "",
        "Reply with ONLY a JSON object mapping every document label to its integer score, "
        "nothing else. No prose, no code fences, no explanation.",
        f'Required keys, exactly these and no others: {labels}',
        'Example format: {"D1": 7, "D2": 0, "D3": 10}',
    ]
    return "\n".join(lines)


def parse_answer_json(output: str, n_docs: int) -> dict[str, int]:
    """Strictly parse a group reply into {label: score}. Raises on anything unexpected."""
    if not output:
        raise ValueError("empty output")
    text = output.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in output: {output[:120]!r}")
    obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("parsed JSON is not an object")
    out: dict[str, int] = {}
    for k, v in obj.items():
        label = str(k).strip().upper()
        score = int(round(float(v)))
        if not 0 <= score <= 10:
            raise ValueError(f"score {score} out of range for {label}")
        out[label] = score
    expected = {f"D{i}" for i in range(1, n_docs + 1)}
    if set(out) != expected:
        raise ValueError(f"label mismatch: got {sorted(out)}, expected {sorted(expected)}")
    return out


class GroupRanker:
    """Diver-GroupRank-32B over vLLM. Loads lazily; TP=4 by default."""

    def __init__(self, model_name: str | None = None,
                 cfg: C.CascadeConfig = C.CASCADE,
                 tensor_parallel_size: int | None = None):
        self.model_name = model_name or C.GROUPRANK_MODEL
        self.cfg = cfg
        self.tp_size = tensor_parallel_size or cfg.group_rank_tp_size
        self._llm = None
        self._sampling = None

    def _load(self):
        if self._llm is None:
            from vllm import LLM, SamplingParams
            self._llm = LLM(
                model=self.model_name,
                dtype="bfloat16",
                tensor_parallel_size=self.tp_size,
                max_model_len=self.cfg.group_rank_max_model_len,
                gpu_memory_utilization=self.cfg.group_rank_gpu_util,
                trust_remote_code=True,
            )
            self._sampling = SamplingParams(
                temperature=0.0,          # scoring, not sampling
                top_p=1.0,
                max_tokens=256,           # a JSON map of <=10 int scores
            )
        return self._llm

    def _chat(self, prompts: list[str]) -> list[str]:
        """One batched vLLM generate call over all groups for a query."""
        llm = self._load()
        tok = llm.get_tokenizer()
        texts = []
        for p in prompts:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": p}]
            texts.append(tok.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True))
        outs = llm.generate(texts, self._sampling)
        return [o.outputs[0].text for o in outs]

    def rank(self, query: str, doc_ids: Sequence[str],
             views: dict[str, str]) -> GroupRankResult:
        """Score every doc_id via random groupwise passes. Never raises."""
        res = GroupRankResult()
        ids = list(doc_ids)
        if not ids:
            return res
        groups = make_groups(ids, self.cfg.group_rank_size,
                             self.cfg.group_rank_repeats, self.cfg.group_rank_seed)
        res.groups_total = len(groups)
        if not groups:
            return res

        try:
            prompts = [build_group_prompt(query, g, views) for g in groups]
            outputs = self._chat(prompts)
        except Exception as e:                                   # noqa: BLE001
            print(f"[grouprank] generation failed, falling back to Qwen order: {e}")
            return res

        totals: dict[str, float] = {}
        for group, out in zip(groups, outputs):
            try:
                scores = parse_answer_json(out, len(group))
            except Exception:                                    # noqa: BLE001
                continue                                         # this group is discarded
            res.groups_parsed += 1
            for i, did in enumerate(group, start=1):
                totals[did] = totals.get(did, 0.0) + float(scores[f"D{i}"])
                res.appearances[did] = res.appearances.get(did, 0) + 1

        # Every document must have been scored at least once, and the run must clear the
        # parse gate. Otherwise the ordering is partly Qwen's and partly GroupRank's,
        # which is a blend nobody evaluated -- so discard the whole pass.
        if len(res.appearances) != len(ids):
            missing = len(ids) - len(res.appearances)
            print(f"[grouprank] {missing} docs unscored; falling back to Qwen order")
            return res
        if res.parse_success < C.CASCADE_GATES.grouprank_min_parse_success:
            print(f"[grouprank] parse success {res.parse_success:.3f} < gate; "
                  f"falling back to Qwen order")
            return res

        res.scores = {d: totals[d] / res.appearances[d] for d in totals}
        res.ok = True
        return res
