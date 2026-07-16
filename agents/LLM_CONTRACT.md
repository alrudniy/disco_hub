# glm-4.6 / z.ai — VERIFIED call contract

Established by live calls against the real endpoint on 2026-07-16. Trust this over any assumption
about "it's OpenAI-compatible, it'll be fine". It mostly is — with one trap that silently returns
empty strings.

    POST {DH_LLM_BASE_URL}/chat/completions      # base = https://api.z.ai/api/paas/v4
    Authorization: Bearer {DH_LLM_API_KEY}
    Content-Type: application/json

Response text is at `choices[0].message.content`. Latency ~1-3 s for short calls.

## THE TRAP: glm-4.6 is a reasoning model

It emits `message.reasoning_content` alongside `message.content`, and **reasoning tokens are billed
against `max_tokens`**. If the budget runs out during reasoning, the API returns HTTP 200 with
`finish_reason: "length"` and **`content: ""`** — a successful-looking response carrying no answer.

Measured, same prompt ("Reply with exactly: OK"):

| call | reasoning_tokens | content | usage |
|---|---:|---|---:|
| `max_tokens: 20`, thinking default | 20 | `""`  ← silent empty | 20 completion |
| `max_tokens: 1000`, thinking default | 42 | `"OK"` | 45 completion |
| `max_tokens: 50`, `thinking: {"type":"disabled"}` | 0 | `"OK"` | 2 completion |

So: **send `thinking: {"type": "disabled"}`** for every structured/extractive task here (verification,
policy flagging, gap narration). None of them benefit from chain-of-thought, and disabling it makes the
call ~20x cheaper in completion tokens and removes the empty-content failure mode entirely.

If thinking is ever enabled deliberately, `max_tokens` must be generous, and the client must treat
empty `content` as a FAILURE (return None → caller falls back to the deterministic path), never as a
valid empty answer.

## Structured output works natively

`response_format: {"type": "json_object"}` returns clean, directly-parseable JSON — no markdown fences
in practice. Keep the defensive fence-stripping/brace-finding in `chat_json` anyway (belt and braces;
a model revision can regress this), but do not rely on prompt-begging alone.

Verified verifier-shaped call, evidence = one B7-H1 patent abstract, answer = one true claim + one
fabricated claim ("B7-H1 was approved by the FDA in 2019"):

```json
{"claims":[{"text":"B7-H1 is an immunoregulatory molecule.","status":"supported","doc_id":"uspto:US6803192B1"},
           {"text":"B7-H1 was approved by the FDA in 2019.","status":"unsupported","doc_id":null}],
 "verdict":"fail","unsupported_count":1}
```

The model returned `doc_id: null` for the unsupported claim — so **`doc_id` is nullable** and the
post-validation must handle `None` before checking membership in the retrieved evidence set.

`temperature: 0` is accepted and is what every agent here should use — this repo cares about
determinism (stage 09 verifies it). Note that temperature 0 still does not make an LLM bit-reproducible;
do not claim it does.

## Secrets

The key lives in `/home/alex/discovery_hub_pipeline/.env` (gitignored, mode 600), which `demo/env.sh`
sources when present. Never inline a key in code, a test, or a committed env file.
