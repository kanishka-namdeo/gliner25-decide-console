"""Environment gate: run before writing any application code.

Checks that the interpreter, the CUDA build, the GPU architecture and the
transformers pin are all correct. Exits non-zero on the first hard failure so
CI or a pre-commit hook can enforce it.
"""

import sys

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -> {detail}" if detail else ""))


def main() -> int:
    print("=" * 72)
    print("ENVIRONMENT GATE")
    print("=" * 72)

    # --- interpreter -----------------------------------------------------
    check("python is 3.12.x", sys.version_info[:2] == (3, 12), sys.version.split()[0])
    check(
        "running inside the project .venv",
        sys.prefix != sys.base_prefix,
        f"prefix={sys.prefix}",
    )

    import torch

    print(f"\ntorch            {torch.__version__}")
    print(f"transformers     {_version('transformers')}")
    print(f"gliner2          {_version('gliner2')}")
    print(f"numpy            {_version('numpy')}")

    # --- check 1: CUDA build, not the CPU-only wheel ----------------------
    cuda_ver = torch.version.cuda
    check(
        "1. torch built against CUDA",
        cuda_ver is not None,
        f"torch.version.cuda={cuda_ver}",
    )

    # --- check 2: CUDA runtime available ---------------------------------
    available = torch.cuda.is_available()
    check("2. torch.cuda.is_available()", available)
    if not available:
        return _report()

    # --- check 3: correct device ------------------------------------------
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    check("3. expected GPU present", "2070" in name, f"{name} sm_{cap[0]}{cap[1]}")

    # --- check 4: the real Turing gate ------------------------------------
    # If sm_75 is absent from the compiled arch list, kernels are unavailable
    # for this GPU and the first matmul fails with "no kernel image".
    archs = torch.cuda.get_arch_list()
    check("4. sm_75 in compiled arch list", "sm_75" in archs, ", ".join(archs))

    # --- check 5: fp16 works on this device ------------------------------
    # bf16 requires sm_80+, so fp16 is the only option here (Turing).
    try:
        a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
        out = a @ a
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(out).all())
        check("5. fp16 matmul on GPU is finite", finite, f"device={out.device}, dtype={out.dtype}")
    except Exception as exc:  # noqa: BLE001 - gate reports, never raises
        check("5. fp16 matmul on GPU is finite", False, repr(exc))

    # --- check 6: transformers 4.x (gliner2 2.0.0 caps transformers<5) ---
    tv = _version("transformers")
    check("6. transformers is 4.x", tv.startswith("4."), tv)

    return _report()


def _version(pkg: str) -> str:
    try:
        from importlib.metadata import version

        return version(pkg)
    except Exception:  # noqa: BLE001
        return "not installed"


def _report() -> int:
    print("\n" + "=" * 72)
    failed = [name for name, ok, _ in RESULTS if not ok]
    if failed:
        print(f"ENVIRONMENT GATE FAILED ({len(failed)}/{len(RESULTS)} checks)")
        for name in failed:
            print(f"  - {name}")
        print("=" * 72)
        return 1
    print(f"ENVIRONMENT GATE PASSED ({len(RESULTS)}/{len(RESULTS)} checks)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())