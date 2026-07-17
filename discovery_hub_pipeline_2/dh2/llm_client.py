"""
dh2.llm_client -- GLM-4.6 (via z.ai) as the gray-zone relevance judge and, where the
spec calls for an external LLM teacher, the graded-label source.

z.ai is OpenAI-compatible:
  POST {base}/chat/completions            (base default https://api.z.ai/api/paas/v4)
  headers: {"Authorization": "Bearer <key>", "Content-Type": "application/json"}
  body:    {"model": ..., "messages": [{"role":"user","content": prompt}],
            "temperature": 0.1, "max_tokens": 512}          # thinking DISABLED for judging
  reply:   data["choices"][0]["message"]["content"]  (string)

Built for UNATTENDED runs:
  * retry/backoff on transient errors (429/5xx/timeout)
  * strict JSON extraction from the model's reply (tolerates prose/fences around JSON)
  * a graded_relevance() helper returning {grade:0-3, relevance:0-1, evidence:str}
  * on-disk cache so a re-run never re-pays for a pair
  * health_check() RETRIES (a single transient failure must not silently disable the
    judge for the whole run -- that happened once with the previous provider)

Judge settings: temperature is LOW (deterministic grading) and z.ai "thinking"/
"reasoning_effort" fields are intentionally omitted -- a 3-field JSON verdict does not
need chain-of-thought, and enabling it would multiply cost/latency over hundreds of
thousands of gray-zone pairs.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from dh2 import config2 as C


class LLMError(RuntimeError):
    pass


class TeacherLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None, timeout: int = 120,
                 max_retries: int = 6, cache_path: Path | None = None,
                 temperature: float | None = None, max_tokens: int = 512):
        self.api_key = api_key or C.TEACHER_LLM_API_KEY
        self.model = model or C.TEACHER_LLM_MODEL
        self.base_url = (base_url or C.TEACHER_LLM_BASE_URL).rstrip("/")
        self.url = f"{self.base_url}/chat/completions"
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = (C.TEACHER_LLM_TEMPERATURE if temperature is None
                            else temperature)
        self.max_tokens = max_tokens
        self.cache_path = cache_path
        self._cache: dict[str, Any] = {}
        self._cache_lock = threading.Lock()
        if cache_path and Path(cache_path).exists():
            for line in Path(cache_path).read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["k"]] = rec["v"]
        if not self.api_key:
            raise LLMError("DH2_TEACHER_LLM_API_KEY not set (needed for LLM judging). "
                           "Export it, or run the stage with --skip-llm to mask gray-zone.")

    # --- low level -------------------------------------------------------- #
    def _post(self, prompt: str) -> str:
        import requests
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": self.temperature,
                   "max_tokens": self.max_tokens,
                   # CRITICAL: GLM models default thinking ON. For a 3-field JSON verdict
                   # that wastes ~300 reasoning tokens/call (~10-45s latency). Force it OFF.
                   "thinking": {"type": "disabled"}}
        last = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(self.url, headers=headers, json=payload,
                                  timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise LLMError(f"transient {r.status_code}: {r.text[:160]}")
                r.raise_for_status()
                data = r.json()
                choices = data.get("choices", [])
                if not choices:
                    raise LLMError(f"no choices in reply: {str(data)[:200]}")
                msg = choices[0].get("message", {})
                text = (msg.get("content") or "").strip()
                if not text:
                    raise LLMError(f"empty content: {str(data)[:200]}")
                return text
            except Exception as e:                       # noqa: BLE001
                last = e
                # longer backoff so retries span minutes, riding out z.ai slow windows
                # (a 15s call under load can transiently exceed a short timeout)
                sleep = min(5 * (2 ** attempt), 60) + 0.5 * attempt
                time.sleep(sleep)
        raise LLMError(f"LLM failed after {self.max_retries} retries: {last}")

    # --- JSON extraction -------------------------------------------------- #
    @staticmethod
    def _extract_json(text: str) -> dict:
        t = text.strip()
        t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
        try:
            return json.loads(t)
        except Exception:
            m = re.search(r"\{.*\}", t, flags=re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
        raise LLMError(f"could not parse JSON from LLM reply: {text[:200]}")

    # --- caching ---------------------------------------------------------- #
    def _cache_key(self, kind: str, *parts: str) -> str:
        h = hashlib.sha256(("|".join((kind, self.model, *parts))).encode()).hexdigest()
        return h[:32]

    def _cache_put(self, k: str, v: Any) -> None:
        with self._cache_lock:
            self._cache[k] = v
            if self.cache_path:
                with Path(self.cache_path).open("a") as f:
                    f.write(json.dumps({"k": k, "v": v}) + "\n")

    def cache_has(self, k: str) -> bool:
        with self._cache_lock:
            return k in self._cache

    def cache_get(self, k: str) -> Any:
        with self._cache_lock:
            return self._cache.get(k)

    def cache_key_for(self, query: str, doc_title: str, doc_text: str,
                      fast: bool | None = None) -> str:
        """Public cache-key builder so a parallel prefetch phase can dedupe/check."""
        fast = C.TEACHER_LLM_FAST if fast is None else fast
        return self._cache_key("gradefast" if fast else "grade", query, doc_title,
                               doc_text[:800])

    # --- the graded-relevance judge (spec P0/P1 gray-zone) ---------------- #
    # Two rubrics: a FAST grade-only prompt (minimal output ~= 2-3s/call) and a full
    # prompt with an evidence string (~15s/call). Grade-only is the default for
    # high-volume gray-zone judging; evidence adds interpretability at ~5x the latency.
    _RUBRIC_FULL = (
        "You are a relevance judge for a pharmaceutical technology-scouting search "
        "engine. A scout issues a natural-language research interest; the system "
        "retrieves patents, clinical trials, academic papers, and SBIR awards. Judge "
        "how USEFUL the document is for the scout's interest, using ONLY the provided "
        "text. Grade on this scale:\n"
        "  3 = directly actionable (same need: disease/stage, mechanism/target, and "
        "modality/route all clearly match)\n"
        "  2 = clearly relevant (same therapeutic area and intent, minor mismatch)\n"
        "  1 = tangentially useful (related but off on a key dimension)\n"
        "  0 = irrelevant\n"
        "Do not reward mere keyword overlap. Do not assume facts not present. "
        "Respond with ONLY a JSON object and nothing else: "
        '{"grade": <0-3>, "relevance": <0.0-1.0>, "evidence": "<=200 chars"}'
    )
    _RUBRIC_FAST = (
        "Relevance judge for a pharma tech-scouting search engine. Judge how useful the "
        "document is for the scout's research interest, using ONLY the given text. "
        "Scale: 3=directly actionable (disease/stage, mechanism, modality all match); "
        "2=clearly relevant (same area+intent, minor mismatch); 1=tangential; 0=irrelevant. "
        "No keyword-only credit; no assumed facts. "
        'Reply with ONLY this JSON, nothing else: {"grade":<0-3>,"relevance":<0.0-1.0>}'
    )

    def graded_relevance(self, query: str, doc_title: str, doc_text: str,
                         source: str = "", fast: bool | None = None) -> dict:
        """Return {grade:int 0-3, relevance:float 0-1, evidence:str}. Cached by content.

        fast=True (default, from config) uses a grade-only prompt (~2-3s/call). fast=False
        adds an evidence string (~15s/call). The return schema is identical either way;
        evidence is "" in fast mode.
        """
        fast = C.TEACHER_LLM_FAST if fast is None else fast
        k = self._cache_key("grade" if not fast else "gradefast", query, doc_title,
                            doc_text[:800])
        if self.cache_has(k):
            return self.cache_get(k)
        rubric = self._RUBRIC_FAST if fast else self._RUBRIC_FULL
        prompt = (f"{rubric}\n\n"
                  f"Research interest: {query}\n"
                  f"Source type: {source or 'unknown'}\n"
                  f"Document title: {doc_title}\n"
                  f"Document text: {doc_text[:1500]}\n")
        # fast mode needs very few output tokens; full mode needs room for evidence
        saved_max = self.max_tokens
        try:
            if fast:
                self.max_tokens = 40
            raw = self._post(prompt)
        finally:
            self.max_tokens = saved_max
        # BUG #10 FIX: this used to swallow the parse failure and cache
        #     grade=1, relevance=0.5, evidence="[unparseable LLM reply masked] ..."
        # Nothing was masked. The string lied, and the row entered training and the qrels
        # as a real, judge-issued "tangential" verdict at weight 0.3. 42 occurrences
        # (0.08%) -- negligible in aggregate, which is exactly why it survived. It is
        # fabricated evidence in a label store whose entire value is that its labels are
        # traceable to a judge.
        # The verdict is still CACHED (unparseable replies are usually deterministic for
        # a given prompt; re-fetching re-pays for a reply that will fail again), but it
        # now carries unparseable=True, and teacher_merge masks on that flag.
        unparseable = False
        try:
            obj = self._extract_json(raw)
            grade = int(obj.get("grade", 0))
            grade = max(0, min(3, grade))
            rel = float(obj.get("relevance", grade / 3.0))
            rel = max(0.0, min(1.0, rel))
            evidence = str(obj.get("evidence", ""))[:200]
        except Exception:
            unparseable = True
            grade, rel = None, None
            evidence = f"[unparseable LLM reply] {raw[:80]}"
        out = {"grade": grade, "relevance": rel, "evidence": evidence,
               "unparseable": unparseable}
        self._cache_put(k, out)
        return out

    @staticmethod
    def is_unparseable(verdict: dict) -> bool:
        """True if a verdict must be masked rather than trusted.

        Also recognizes the LEGACY poisoned rows already sitting in llm_cache.jsonl --
        52,709 verdicts is the expensive artifact of this project and must not be
        re-fetched just to correct 42 rows, so the legacy marker string is detected on
        read instead. Retroactive, free.
        """
        if not isinstance(verdict, dict):
            return True
        if verdict.get("unparseable"):
            return True
        if verdict.get("grade") is None:
            return True
        ev = str(verdict.get("evidence", ""))
        return ev.startswith("[unparseable LLM reply")

    def health_check(self, attempts: int = 4) -> bool:
        """Confirm key/model/endpoint before a long run. RETRIES so one transient blip
        does not disable the judge for the entire run."""
        last_err = ""
        for i in range(attempts):
            try:
                out = self.graded_relevance(
                    "oral therapy for relapsed CLL",
                    "Oral drug X in Relapsed CLL",
                    "A trial of an oral agent in relapsed chronic lymphocytic leukemia.",
                    source="clinicaltrials")
                if isinstance(out, dict) and "grade" in out:
                    return True
            except Exception as e:                       # noqa: BLE001
                last_err = str(e)
                time.sleep(min(2 ** i, 15))
        print(f"[llm_client] health_check failed after {attempts} attempts: {last_err}")
        return False

    def as_cache_only(self):
        """Return a view that ONLY returns cached verdicts and raises for anything
        uncached (so a downstream merge masks it instead of making a new API call).
        Used in p0_3 Phase 2 after the parallel prefetch has filled the cache within
        budget -- prevents the serial merge from silently re-fetching skipped pairs."""
        outer = self

        class _CacheOnly:
            def graded_relevance(self, query, doc_title, doc_text, source=""):
                k = outer.cache_key_for(query, doc_title, doc_text)
                if outer.cache_has(k):
                    return outer.cache_get(k)
                raise RuntimeError("not_prefetched")  # -> merge_one masks this pair

        return _CacheOnly()

    def prefetch_parallel(self, jobs: list[dict], workers: int = 16,
                          budget: int = 0, progress_every: int = 100) -> dict:
        """Fill the cache by judging many pairs CONCURRENTLY.

        jobs: list of {"query","doc_title","doc_text","source"} dicts (the gray-zone pairs).
        workers: number of concurrent requests (z.ai has a fat latency tail, so overlap
                 many calls; 16-32 is a good range).
        budget: max NEW (uncached) calls; 0 = unlimited. Pairs beyond budget are skipped
                (left uncached) so the serial merge phase masks them.
        Returns stats. Safe to re-run: cached pairs are never re-fetched.

        Ordering note: results are written to the cache keyed by content, so worker order
        does not matter -- the downstream merge reads by key, not by arrival.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # dedupe by cache key and drop already-cached pairs up front
        todo = {}
        for j in jobs:
            k = self.cache_key_for(j["query"], j.get("doc_title", ""), j.get("doc_text", ""))
            if k in todo or self.cache_has(k):
                continue
            todo[k] = j
        todo_list = list(todo.values())
        if budget and len(todo_list) > budget:
            todo_list = todo_list[:budget]   # bound NEW calls for a proof run

        total = len(todo_list)
        done = {"ok": 0, "err": 0}
        lock = threading.Lock()

        def _one(j):
            try:
                self.graded_relevance(j["query"], j.get("doc_title", ""),
                                      j.get("doc_text", ""), source=j.get("source", ""))
                with lock:
                    done["ok"] += 1
            except Exception:                            # noqa: BLE001
                with lock:
                    done["err"] += 1
            with lock:
                n = done["ok"] + done["err"]
            if progress_every and n % progress_every == 0:
                print(f"[llm_client] prefetch {n}/{total} "
                      f"(ok={done['ok']} err={done['err']})", flush=True)

        if total == 0:
            return {"requested": len(jobs), "new_calls": 0, "cached_skipped": len(jobs),
                    "ok": 0, "err": 0}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_one, j) for j in todo_list]
            for _ in as_completed(futs):
                pass
        return {"requested": len(jobs), "new_calls": total,
                "cached_skipped": len(jobs) - total, "ok": done["ok"], "err": done["err"]}
