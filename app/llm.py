"""OpenAI-compatible LLM client for the comparison arm.

Speaks plain HTTP to /chat/completions rather than using the openai SDK, so any
compatible server works (OpenAI, OpenRouter, vLLM, Ollama, a gateway) with no
code change. Credentials come from the environment and are never logged.

Protocol, following arXiv:2608.20371: the model is shown the text and the bare
label list and must return one label. It is deliberately NOT given per-label
descriptions, because the encoder arm does get descriptions on banking77 and
handing only one side them would make the comparison meaningless.

Pricing is configurable because it changes constantly and quoting a stale number
in a UI is worse than asking the user for it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("llm")

SYSTEM_PROMPT = (
    "You are a text classification engine. Reply with exactly one label from the "
    "provided list and nothing else. No explanation, no punctuation, no quotes."
)

# Prices in USD per million tokens. Override via env; these are placeholders the
# UI displays as editable rather than asserting.
DEFAULT_INPUT_PER_MTOK = 0.0
DEFAULT_OUTPUT_PER_MTOK = 0.0

VALID_LABEL = re.compile(r"^[a-z0-9]+(?:[._\-][a-z0-9]+)*$")


@dataclass
class LLMResponse:
    label: str | None
    raw_text: str
    latency_ms: float
    valid: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None

    @property
    def cost_usd(self) -> float:
        cfg = LLMConfig.from_env()
        if not self.prompt_tokens and not self.completion_tokens:
            return 0.0
        return (
            (self.prompt_tokens or 0) / 1e6 * cfg.input_per_mtok
            + (self.completion_tokens or 0) / 1e6 * cfg.output_per_mtok
        )


def load_env_file(path: str | Path = ".env") -> bool:
    """Read LLM_* settings from a .env file into os.environ.

    Deliberately not a dependency: python-dotenv would be another pin to
    justify, and this is a dozen lines of straight-line parsing. Existing
    environment variables win, so a shell export overrides the file, which is
    what an operator expects when
    debugging a bad key.
    """
    env_path = Path(path)
    if not env_path.exists():
        return False
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
    return True


@dataclass
class LLMConfig:
    base_url: str | None
    api_key: str | None
    model: str | None
    input_per_mtok: float
    output_per_mtok: float
    timeout_s: float = 60.0
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "LLMConfig":
        # Load .env before reading, so a saved key works without the operator
        # having to export it in every shell. Environment wins over the file.
        load_env_file()
        base = os.getenv("LLM_BASE_URL", "").strip() or None
        return cls(
            base_url=base,
            api_key=os.getenv("LLM_API_KEY", "").strip() or None,
            model=os.getenv("LLM_MODEL", "").strip() or None,
            input_per_mtok=float(os.getenv("LLM_INPUT_PER_MTOK", DEFAULT_INPUT_PER_MTOK)),
            output_per_mtok=float(os.getenv("LLM_OUTPUT_PER_MTOK", DEFAULT_OUTPUT_PER_MTOK)),
            timeout_s=float(os.getenv("LLM_TIMEOUT_S", "60")),
        )

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def status(self) -> dict:
        """Never leak the key. Enough for the UI to explain why it is disabled."""
        missing = [
            field
            for field, value in (
                ("LLM_BASE_URL", self.base_url),
                ("LLM_API_KEY", self.api_key),
                ("LLM_MODEL", self.model),
            )
            if not value
        ]
        return {
            "configured": self.configured,
            "model": self.model,
            "base_url": self.base_url,
            "missing_env": missing,
            "input_per_mtok": self.input_per_mtok,
            "output_per_mtok": self.output_per_mtok,
            "reason": None
            if self.configured
            else f"set {', '.join(missing)} in .env to enable the LLM arm",
        }


def build_prompt(text: str, labels: list[str]) -> str:
    return f"Labels:\n{json.dumps(labels)}\n\nText:\n{text}\n\nLabel:"


class LLMClassifier:
    """Thin async client. One instance per app; httpx handles pooling."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        self._client: httpx.AsyncClient | None = None

    async def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=(self.config.base_url or "").rstrip("/"),
                timeout=self.config.timeout_s,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _extract_label(self, content: str, labels: list[str]) -> tuple[str | None, bool]:
        """Pull one label out of the reply and confirm it is really in the list.

        Tolerates the cosmetic variations models actually produce: surrounding
        quotes or code fences, trailing prose on the line, a capitalised
        `Refund_not_showing_up`, or a hyphenated `card-lost`. Those are the same
        answer, and counting them wrong would understate the LLM arm.

        A reply that is *not* a variant of a real label stays invalid: inventing
        a label outside the schema is a genuine error, not a formatting quirk.

        Also accepts a position number ("3"), since some models prefer that.
        """
        lookup = {label.lower(): label for label in labels}
        text = content.strip().strip("`").strip()
        text = text.splitlines()[0].strip() if text else ""
        if not text:
            return None, False

        # Normalise separators and case: "Card-Lost" -> "card_lost".
        key = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

        if key in lookup:
            return lookup[key], True

        # Tolerate a short lead-in ("The answer is card_pin_change") by asking
        # whether the reply *ends* with a label at a separator boundary. Keeping
        # compound labels intact matters: splitting on "_" would turn
        # card_pin_change into card/pin/change and never match it. Requiring the
        # boundary still refuses open-ended prose, because a sentence that merely
        # ends in a label word has no separator in front of it.
        for lower in sorted(lookup, key=len, reverse=True):
            if key.endswith("_" + lower):
                return lookup[lower], True

        if key.isdigit():
            idx = int(key)
            if 0 <= idx < len(labels):
                return labels[idx], True

        return None, False

    async def classify(
        self, text: str, labels: list[str], system: str = SYSTEM_PROMPT
    ) -> LLMResponse:
        if not self.config.configured:
            return LLMResponse(
                label=None,
                raw_text="",
                latency_ms=0.0,
                valid=False,
                error="LLM arm not configured",
            )

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": build_prompt(text, labels)},
            ],
            "temperature": 0,
            "max_tokens": 16,
        }

        client = await self.client()
        last_error: str | None = None
        for attempt in range(self.config.max_retries):
            t0 = time.perf_counter()
            try:
                resp = await client.post("/chat/completions", json=payload)
                if resp.status_code >= 400:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    # 4xx other than 429 will not fix themselves.
                    if resp.status_code < 500 and resp.status_code != 429:
                        break
                    await asyncio_sleep(2**attempt)
                    continue

                body = resp.json()
                content = body["choices"][0]["message"]["content"] or ""
                usage = body.get("usage") or {}
                label, valid = self._extract_label(content, labels)
                return LLMResponse(
                    label=label,
                    raw_text=content[:200],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    valid=valid,
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    error=None if valid else f"reply not in label set: {content[:80]!r}",
                )
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                await asyncio_sleep(2**attempt)

        return LLMResponse(
            label=None,
            raw_text="",
            latency_ms=0.0,
            valid=False,
            error=last_error or "unknown error",
        )

    async def probe(self) -> dict:
        """Cheapest possible round trip, to confirm the endpoint answers."""
        status = self.config.status()
        if not status["configured"]:
            return {**status, "reachable": False}
        try:
            resp = await self.classify("hello", ["greeting", "farewell"])
            return {
                **status,
                "reachable": True,
                "model_echo_ok": resp.label is not None or resp.error is None,
                "sample_latency_ms": resp.latency_ms,
            }
        except Exception as exc:  # noqa: BLE001
            return {**status, "reachable": False, "error": f"{type(exc).__name__}: {exc}"}


async def asyncio_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)