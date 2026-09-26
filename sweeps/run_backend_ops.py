"""Backend operator correctness receipt.

Runs test-backend-ops and binds, in one artifact: the executable hash, the sibling ggml-hip.dll
hash, the exact argv, the exit code, the parsed pass/fail/unsupported counts, and the full raw
log. Everything is captured at run time from the files the launcher actually loads, so a reader
can re-derive the result without trusting this script's prose.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "serve"))

from common import (  # noqa: E402
    ACTIVE_DLL,
    atomic_write_json,
    capture_gpu_process_memory,
    ensure_no_llama_processes,
    environment,
    file_provenance,
    utc_now,
)

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PASSED = re.compile(r"^\s*(\d+)/(\d+) tests passed\s*$", re.M)
BACKENDS = re.compile(r"^\s*(\d+)/(\d+) backends passed\s*$", re.M)
OK_ROW = re.compile(r"^\s+\S.*\): OK\s*$", re.M)
UNSUPPORTED_ROW = re.compile(r"^\s+\S.*\): not supported \[", re.M)
FAILED_ROW = re.compile(r"^\s+\S.*\): FAILED", re.M)


def sibling_dll(binary):
    dll = Path(binary).parent / "ggml-hip.dll"
    return file_provenance(dll) if dll.is_file() else None


def main(argv=None):
    parser = argparse.ArgumentParser(description="test-backend-ops receipt with bound provenance.")
    parser.add_argument("--binary", type=Path,
                        default=ACTIVE_DLL.parent / "test-backend-ops.exe")
    parser.add_argument("--op", default="MUL_MAT")
    parser.add_argument("--param", default="ptq1_0")
    parser.add_argument("--device", default="ROCm0")
    parser.add_argument("--expected-dll-sha256", default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"backend-ops-{args.label}.log"
    receipt_path = out_dir / f"backend-ops-{args.label}.json"

    binary_info = file_provenance(args.binary)
    dll_info = sibling_dll(args.binary)
    if args.expected_dll_sha256:
        got = (dll_info or {}).get("sha256")
        if got != args.expected_dll_sha256:
            raise SystemExit(f"ggml-hip.dll sha256 {got} != expected {args.expected_dll_sha256}")

    cli = [str(args.binary), "-o", args.op, "-p", args.param, "-b", args.device]
    preflight = ensure_no_llama_processes()
    occupancy_before = capture_gpu_process_memory()

    completed = subprocess.run(cli, env=environment(), capture_output=True, text=True,
                              cwd=str(Path(args.binary).parent))
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    plain = ANSI.sub("", completed.stdout)

    passed = PASSED.search(plain)
    backends = BACKENDS.search(plain)
    receipt = {
        "schema_version": 1,
        "recorded_at_utc": utc_now(),
        "label": args.label,
        "benchmark_environment": "desktop-active",
        "argv": cli,
        "returncode": completed.returncode,
        "log_path": str(log_path),
        "log_sha256": file_provenance(log_path)["sha256"],
        "binary": binary_info,
        "ggml_hip_dll": dll_info,
        "ggml_hip_dll_after": sibling_dll(args.binary),
        "tests_passed": int(passed.group(1)) if passed else None,
        "tests_total": int(passed.group(2)) if passed else None,
        "backends_passed": int(backends.group(1)) if backends else None,
        "backends_total": int(backends.group(2)) if backends else None,
        "ok_rows": len(OK_ROW.findall(plain)),
        "unsupported_rows": len(UNSUPPORTED_ROW.findall(plain)),
        "failed_rows": len(FAILED_ROW.findall(plain)),
        "preflight_llama_processes": preflight,
        "background_occupancy": {"before": occupancy_before, "after": None},
        "stderr_tail": completed.stderr[-4000:],
    }
    try:
        receipt["background_occupancy"]["after"] = capture_gpu_process_memory()
    except Exception as exc:
        receipt["background_occupancy"]["after_error"] = f"{type(exc).__name__}: {exc}"
    atomic_write_json(receipt_path, receipt)

    ok = (receipt["failed_rows"] == 0 and receipt["returncode"] == 0
          and receipt["tests_passed"] == receipt["tests_total"]
          and receipt["tests_total"])
    print(f"{args.label}: exit={receipt['returncode']} "
          f"{receipt['tests_passed']}/{receipt['tests_total']} passed, "
          f"{receipt['unsupported_rows']} unsupported, {receipt['failed_rows']} failed "
          f"-> {'PASS' if ok else 'NOT-PASS'}")
    print(f"receipt={receipt_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
