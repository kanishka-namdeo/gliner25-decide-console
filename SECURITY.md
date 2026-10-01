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

All inference is local. Model weights and datasets are downloaded from Hugging
Face; text sent to `/api/route` is processed in-process and is not forwarded
anywhere — with the single exception of the LLM comparison arm, which sends the
text to whichever endpoint `LLM_BASE_URL` names. **Anything you paste into the
Router tab is sent to that endpoint when the LLM arm is enabled.** Do not use
the LLM arm with confidential input.

## Dependencies

`gliner2` and the model weights are Apache-2.0; see `NOTICE` for the full
attribution. Model weights are downloaded at run time and are not redistributed
here. Run `uv sync` against a pinned `uv.lock` for a reproducible environment.