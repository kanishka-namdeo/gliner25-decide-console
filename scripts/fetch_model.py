"""Resumable model fetcher.

uv and huggingface_hub both restart an interrupted transfer from zero. On a slow
or flaky link a 2 GB file never completes, so we download to a local directory
ourselves with HTTP range resume, then point `from_pretrained` at that path.

Files land in models/<repo-name>/ and are validated by size, so re-running is
cheap and safe: anything already complete is skipped.

    python scripts/fetch_model.py                  # default checkpoint
    python scripts/fetch_model.py fastino/gliner2.5-base-v1
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

MODELS_ROOT = Path("models")

# Explicit file lists: everything from_pretrained touches, nothing else.
FILE_SETS: dict[str, list[str]] = {
    "fastino/GLiNER2.5-Decide": [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "encoder_config/config.json",
    ],
    "fastino/gliner2.5-base-v1": [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ],
}

MAX_ATTEMPTS = 40
CHUNK = 1 << 20  # progress granularity only


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def remote_size(url: str, client: httpx.Client) -> int | None:
    try:
        r = client.head(url, follow_redirects=True, timeout=30.0)
        return int(r.headers["content-length"]) if r.status_code == 200 else None
    except Exception:
        return None


def download(url: str, dest: Path, expected: int | None, client: httpx.Client) -> bool:
    """Download with range resume. Returns True on completion."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if expected and dest.exists() and dest.stat().st_size == expected:
        print(f"  skip (complete) {dest.name} [{human(expected)}]")
        return True

    attempt = 0
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        have = dest.stat().st_size if dest.exists() else 0
        if expected and have >= expected:
            break

        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with client.stream("GET", url, headers=headers, timeout=60.0, follow_redirects=True) as r:
                if r.status_code not in (200, 206):
                    print(f"  attempt {attempt}: HTTP {r.status_code}, retrying")
                    time.sleep(min(2 ** attempt, 30))
                    continue

                mode = "ab" if (r.status_code == 206 and have) else "wb"
                start_time, start_bytes = time.time(), have
                with dest.open(mode) as fh:
                    for chunk in r.iter_bytes(CHUNK):
                        fh.write(chunk)
                        have += len(chunk)
                        if expected and time.time() - start_time > 2:
                            rate = (have - start_bytes) / (time.time() - start_time)
                            pct = 100.0 * have / expected if expected else 0
                            print(
                                f"\r  {dest.name}: {pct:5.1f}%  "
                                f"{human(have)}/{human(expected) if expected else '?'}  "
                                f"{human(rate)}/s",
                                end="",
                                flush=True,
                            )
                            start_time, start_bytes = time.time(), have
            print()
            if not expected or dest.stat().st_size == expected:
                return True
            print(f"  attempt {attempt}: short read, resuming")
        except Exception as exc:  # noqa: BLE001 - any network fault is retryable
            print(f"\n  attempt {attempt}: {type(exc).__name__}: {exc}")
            print(f"  resuming from {human(dest.stat().st_size if dest.exists() else 0)}")
        time.sleep(min(2 ** attempt, 30))

    return expected is None or dest.exists()


def main() -> int:
    repo = sys.argv[1] if len(sys.argv) > 1 else "fastino/GLiNER2.5-Decide"
    if repo not in FILE_SETS:
        print(f"No file set for {repo}. Known: {', '.join(FILE_SETS)}")
        return 1

    target = MODELS_ROOT / repo.split("/")[-1]
    target.mkdir(parents=True, exist_ok=True)
    print(f"fetching {repo} -> {target}")

    failures = []
    with httpx.Client() as client:
        for name in FILE_SETS[repo]:
            url = f"https://huggingface.co/{repo}/resolve/main/{name}"
            dest = target / name
            size = remote_size(url, client)
            print(f"\n{name}  (remote: {human(size) if size else 'unknown'})")
            if not download(url, dest, size, client):
                failures.append(name)

    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        print("Re-run to resume from where it stopped.")
        return 1

    print(f"\nComplete. Load with:\n  from_pretrained(r'{target.as_posix()}')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())