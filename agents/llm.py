"""
A minimal OpenAI-compatible chat client, pointed at glm-4.6 via z.ai.

WHY THIS EXISTS SEPARATELY FROM 08: `08_multiagent_rag.py::_explain_real_factory`
talks to the 1min.ai API shape (an `API-KEY` header, `/api/chat-with-ai`). z.ai
is a different shape entirely -- OpenAI-compatible: `POST {base}/chat/completions`
with `Authorization: Bearer {key}`. Copying 08's client would send the right
prompt to the wrong protocol.

THE CENTRAL DESIGN RULE: THIS CLIENT IS OPTIONAL.
`DH_LLM_API_KEY` is frequently absent -- in CI, in unit tests, on a fresh clone.
When it is, `.available` is False and `.chat`/`.chat_json` return None rather
than raising. Callers MUST have a deterministic fallback path and MUST say in
their output that the LLM did not run. Silently rendering a template and letting
it read as model output is the exact overclaim this codebase exists to avoid.

No secret is ever hardcoded or logged: the key is read from the environment and
never appears in an exception message or a repr.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import requests

DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4"
DEFAULT_MODEL = "glm-4.6"

# Retry only what a retry can fix: rate limits, server faults, and timeouts.
# A 401 is a wrong key -- retrying it just burns 3x the latency to fail anyway.
_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 2
_BACKOFF_BASE_S = 0.5

# glm-4.6 is a REASONING model, and its reasoning tokens bill against max_tokens.
# Left enabled, it spends the budget thinking and returns HTTP 200 with
# finish_reason="length" and content="" -- a successful-looking response carrying
# no answer. MEASURED on a synthesis-shaped prompt at max_tokens=1024:
#
#   thinking default : reasoning_tokens=1024, finish_reason=length, content=''
#   thinking disabled: reasoning_tokens=0,    finish_reason=stop,   content=<the answer>
#
# That empty string is not hypothetical: it silently drove every agent in this
# package onto its deterministic fallback on the first full demo run.
#
# None of the jobs here (evidence-gating, policy flagging, gap narration, strict-
# RAG synthesis) is a reasoning task -- each is extractive over supplied evidence.
# So thinking is OFF by default. Callers wanting it must pass thinking=True AND a
# max_tokens generous enough to cover reasoning + the answer.
# See agents/LLM_CONTRACT.md for the full measured contract.
_THINKING_DISABLED = {"type": "disabled"}

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


class LLMClient:
    """
    A bounded-retry chat client. Construct once and share; it is stateless apart
    from the requests.Session, and holds no conversation history.
    """

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None) -> None:
        self.base_url = (base_url or os.environ.get("DH_LLM_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("DH_LLM_MODEL") or DEFAULT_MODEL
        # May legitimately be None. That is a supported state, not an error.
        self._api_key = api_key or os.environ.get("DH_LLM_API_KEY") or None
        self._session = requests.Session()

    def __repr__(self) -> str:  # never leak the key through a traceback/log line
        return (f"LLMClient(base_url={self.base_url!r}, model={self.model!r}, "
                f"available={self.available})")

    @property
    def available(self) -> bool:
        """True iff an API key is present. Callers branch on this BEFORE calling."""
        return self._api_key is not None

    def chat(self, messages: list[dict], temperature: float = 0.0,
             max_tokens: int = 1024, timeout: int = 60, *,
             thinking: bool = False, response_format: dict | None = None
             ) -> str | None:
        """
        One completion. Returns the assistant text, or None if the LLM is
        unavailable, every attempt failed, or the model returned no text.

        An EMPTY completion is a failure, not an answer. glm-4.6 returns 200 with
        content="" when reasoning exhausts max_tokens (see _THINKING_DISABLED), and
        a caller that treats "" as valid renders a blank answer as if the model
        wrote it. Returning None routes it to the deterministic fallback, which
        labels itself honestly.

        temperature defaults to 0.0: this repo cares about reproducibility, and
        greedy decode is the closest an API model gets to it. (It is not a
        guarantee -- see discovery_hub/determinism.py on batch-invariance.)
        """
        if not self.available:
            return None

        payload: dict[str, Any] = {
            "model": self.model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens,
        }
        if not thinking:
            payload["thinking"] = _THINKING_DISABLED
        if response_format is not None:
            payload["response_format"] = response_format
        headers = {"Authorization": f"Bearer {self._api_key}",
                   "Content-Type": "application/json"}
        url = f"{self.base_url}/chat/completions"

        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = self._session.post(url, json=payload, headers=headers,
                                          timeout=timeout)
            except (requests.Timeout, requests.ConnectionError):
                if attempt < _MAX_RETRIES:
                    time.sleep(_BACKOFF_BASE_S * (2 ** attempt))
                    continue
                return None

            if resp.status_code == 200:
                try:
                    text = resp.json()["choices"][0]["message"]["content"]
                except (ValueError, KeyError, IndexError):
                    return None      # 200 with an unexpected body: not retryable
                # "" means the budget went to reasoning, or the model declined to
                # speak. Either way there is no answer here; say so with None.
                return text if (text or "").strip() else None
            if resp.status_code in _RETRY_STATUS and attempt < _MAX_RETRIES:
                time.sleep(_BACKOFF_BASE_S * (2 ** attempt))
                continue
            return None              # 401 and every other 4xx: retrying cannot help
        return None

    def chat_json(self, messages: list[dict], schema_hint: str,
                  temperature: float = 0.0, max_tokens: int = 1024,
                  timeout: int = 60) -> dict | None:
        """
        Ask for JSON and parse it defensively. Returns None -- never a partial or
        guessed dict -- when the model is unavailable or emits unparseable text.
        A None here means "no LLM verdict", which the caller must handle as such.

        `schema_hint` is appended as a system message describing the exact shape
        wanted; models comply far more reliably when the schema is stated.
        """
        instructed = list(messages) + [{
            "role": "system",
            "content": ("Respond with a single JSON object and nothing else. No "
                        f"markdown fences, no prose. Schema:\n{schema_hint}"),
        }]
        # response_format is the enforced contract; the schema_hint above is what
        # tells the model which SHAPE to fill. Both: JSON mode alone would happily
        # return a well-formed object with the wrong keys. _parse_json_loose stays
        # as the third layer -- verified to be unnecessary today, but a model
        # revision that reintroduces fences should degrade, not crash.
        text = self.chat(instructed, temperature=temperature,
                         max_tokens=max_tokens, timeout=timeout,
                         response_format={"type": "json_object"})
        return None if text is None else _parse_json_loose(text)


def _parse_json_loose(text: str) -> dict | None:
    """
    Parse JSON out of model prose. Tries, in order: the raw text; the text with
    markdown fences stripped; the outermost {...} span. Returns None on failure
    or on valid JSON that isn't an object -- a bare list or string is not the
    contract the caller asked for, and coercing it would invent structure.
    """
    for candidate in (text, _FENCE.sub("", text.strip()), _outermost_braces(text)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _outermost_braces(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else None
