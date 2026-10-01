# Security Policy

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue. Use
GitHub's private vulnerability reporting on this repository, or email the
maintainer.

## Credentials

**Never commit an API key.** The `.env` file is gitignored and `.env.example`
contains placeholders only.

The LLM arm reads three variables from the environment:

| Variable | Purpose |
|---|---|
| `LLM_BASE_URL` | OpenAI-compatible endpoint |
| `LLM_API_KEY` | Bearer token, sent as `Authorization: Bearer ...` |
| `LLM_MODEL` | Model identifier |

To keep this safe:

- Keep keys in `.env` or your shell environment, not in source.
- Rotate a key immediately if it is ever committed, pushed, or pasted into a
  chat, issue, or log. Once exposed, treat it as burned.
- Set `LLM_INPUT_PER_MTOK` / `LLM_OUTPUT_PER_MTOK` locally rather than hardcoding
  pricing in a commit.
- `LLMConfig.status()` never returns the key, so the API's `/api/health` is safe
  to expose on a shared network. Do not add the key to any response payload.

## Local-only by design

The server binds to `127.0.0.1` and has **no authentication**. Any client that
can reach the port can submit arbitrary text for classification and, when
baselines are enabled, spend GPU time and API credits.

Do not bind to `0.0.0.0` on a shared or untrusted network without putting
authentication in front of it. There is no CSRF protection, so treat the local
port as fully trusted.

## Data handling

Inference is local. Model weights and datasets are downloaded from Hugging Face.

Text sent to `/api/route` and `/api/schema-swap` is processed in-process and is
**never forwarded anywhere** — neither endpoint reaches the LLM client at all.

The one exception is `/api/panel`, and only when `llm` is among the systems you
select. That arm sends each item to whichever endpoint `LLM_BASE_URL` names,
under `Authorization: Bearer $LLM_API_KEY`. Nothing leaves the machine until you
request that run; the arm is opt-in per panel run, and an unconfigured arm
reports itself unavailable with the missing variable names rather than silently
skipping. Do not include confidential input in a panel run that has the LLM arm
enabled.

## Dependencies

`gliner2` and the model weights are Apache-2.0; see `NOTICE` for the full
attribution. Model weights are downloaded at run time and are not redistributed
here. Run `uv sync --extra dev` against a pinned `uv.lock` for a reproducible
environment. CI runs the offline suite on every push and additionally resolves
the pinned CUDA `torch` and asserts the `transformers` 4.x cap, because neither
dependency trap is visible from a CPU test run.