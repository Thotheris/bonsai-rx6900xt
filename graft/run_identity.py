"""Run strict greedy identity checks on one grafted MTP file, off vs on."""

import argparse
import copy
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "serve"))

from common import (  # noqa: E402
    ACTIVE_DLL,
    BINARY,
    MODEL_SPECS,
    atomic_write_json,
    capture_gpu_process_memory,
    ensure_no_llama_processes,
    environment,
    file_provenance,
    managed_server,
    request_json,
    server_argv,
    utc_now,
)
from bit_identical import compare_receipts, validate_completion  # noqa: E402

IDENTITY_CONTEXT = 32768


def read_prompts(path):
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 3 or any(not line.strip() for line in lines):
        raise ValueError("identity prompt file must contain exactly three non-empty lines")
    return lines


def completion_request(prompt):
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("identity prompt must be a non-empty string")
    return {
        "prompt": prompt,
        "n_predict": 128,
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "seed": 42,
        "cache_prompt": False,
        "return_tokens": True,
        "stream": False,
        "response_fields": ["content", "tokens", "timings", "stop_type", "truncated"],
    }


def capture_prompt(base_url, prompt, label, provenance, *, timeout=600.0):
    request = completion_request(prompt)
    response = validate_completion(request_json(base_url, "/completion", request, timeout=timeout))
    if len(response["tokens"]) != request["n_predict"]:
        raise ValueError("identity completion must generate exactly 128 tokens")
    return {
        "schema_version": 1,
        "recorded_at_utc": utc_now(),
        "label": label,
        "provenance": copy.deepcopy(provenance),
        "request": request,
        "prompt_utf8_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "response": response,
    }


def run_arm(base_url, prompts, label, provenance, *, timeout=600.0):
    return {
        "schema_version": 1,
        "label": label,
        "recorded_at_utc": utc_now(),
        "receipts": [
            capture_prompt(base_url, prompt, label, provenance, timeout=timeout)
            for prompt in prompts
        ],
    }


def compare_arms(left_arm, right_arm):
    left = left_arm.get("receipts", [])
    right = right_arm.get("receipts", [])
    if len(left) != 3 or len(right) != 3:
        raise ValueError("identity arms must each contain three prompt receipts")
    return [compare_receipts(a, b) for a, b in zip(left, right)]


def _parser():
    parser = argparse.ArgumentParser(
        description="Compare greedy MTP-off and MTP-on output on the same grafted GGUF.",
    )
    parser.add_argument("--model", required=True, choices=("pq2-mtp", "ptq1-mtp"))
    parser.add_argument("--context", type=int, default=IDENTITY_CONTEXT)
    parser.add_argument("--port", type=int)
    parser.add_argument("--binary", type=Path, default=BINARY)
    parser.add_argument("--hip-dll", type=Path, default=ACTIVE_DLL)
    parser.add_argument("--expected-hip-dll-sha256")
    parser.add_argument("--prompts", type=Path, default=ROOT / "prompts" / "identity.txt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "identity")
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    return parser


def _record_after(runtime):
    """Occupancy after the run is evidence, not a reason to drop token receipts."""
    try:
        runtime["background_occupancy"]["after"] = capture_gpu_process_memory()
    except Exception as exc:
        runtime["background_occupancy"]["after_error"] = f"{type(exc).__name__}: {exc}"


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.context <= 0 or args.context > IDENTITY_CONTEXT:
        raise SystemExit(f"--context must be between 1 and {IDENTITY_CONTEXT}")

    spec = MODEL_SPECS[args.model]
    port = args.port or spec["port"]
    prompts = read_prompts(args.prompts)
    model_info = file_provenance(spec["path"], spec["sha256"])
    binary_info = file_provenance(args.binary)
    hip_info = file_provenance(args.hip_dll, args.expected_hip_dll_sha256)
    prompts_info = file_provenance(args.prompts)
    prefix = "pq2" if args.model == "pq2-mtp" else "ptq1"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    common_provenance = {
        "model": model_info,
        "binary": binary_info,
        "hip_dll": hip_info,
        "prompts_file": prompts_info,
        "context_tokens": args.context,
    }

    off_argv = server_argv(spec["path"], args.context, port, spec="none", binary=args.binary)
    off_provenance = dict(common_provenance, server_argv=off_argv, mtp="off")
    off_log = output_dir / f"{prefix}-off.server.log"
    off_runtime = {
        "benchmark_environment": "desktop-active",
        "preflight_llama_processes": ensure_no_llama_processes(),
        "background_occupancy": {"before": capture_gpu_process_memory(), "after": None},
    }
    with managed_server(
        off_argv,
        environment(),
        port,
        off_log,
        startup_timeout=args.startup_timeout,
    ) as running:
        off_provenance["server"] = {key: running[key] for key in ("pid", "base_url", "started_at_utc", "health")}
        off = run_arm(running["base_url"], prompts, f"{prefix}-off", off_provenance, timeout=args.request_timeout)
        repeat = run_arm(
            running["base_url"],
            prompts,
            f"{prefix}-off-repeat",
            off_provenance,
            timeout=args.request_timeout,
        )

    _record_after(off_runtime)
    off["runtime"] = copy.deepcopy(off_runtime)
    repeat["runtime"] = copy.deepcopy(off_runtime)
    atomic_write_json(output_dir / f"{prefix}-off.json", off)
    atomic_write_json(output_dir / f"{prefix}-off-repeat.json", repeat)
    self_comparison = compare_arms(off, repeat)
    if not all(item["identical"] for item in self_comparison):
        failure = {
            "schema_version": 1,
            "model": args.model,
            "self_comparison": self_comparison,
            "status": "invalid-self-compare",
        }
        atomic_write_json(output_dir / f"{prefix}-comparison.json", failure)
        raise SystemExit("frozen-binary self-comparison failed; MTP arm was not run")

    on_argv = server_argv(spec["path"], args.context, port, spec="draft-mtp", binary=args.binary)
    on_provenance = dict(common_provenance, server_argv=on_argv, mtp="on")
    on_log = output_dir / f"{prefix}-mtp.server.log"
    on_runtime = {
        "benchmark_environment": "desktop-active",
        "preflight_llama_processes": ensure_no_llama_processes(),
        "background_occupancy": {"before": capture_gpu_process_memory(), "after": None},
    }
    with managed_server(
        on_argv,
        environment(),
        port,
        on_log,
        startup_timeout=args.startup_timeout,
    ) as running:
        on_provenance["server"] = {key: running[key] for key in ("pid", "base_url", "started_at_utc", "health")}
        on = run_arm(running["base_url"], prompts, f"{prefix}-mtp", on_provenance, timeout=args.request_timeout)

    _record_after(on_runtime)
    on["runtime"] = on_runtime
    atomic_write_json(output_dir / f"{prefix}-mtp.json", on)
    mtp_comparison = compare_arms(off, on)
    summary = {
        "schema_version": 1,
        "recorded_at_utc": utc_now(),
        "model": args.model,
        "model_sha256": model_info["sha256"],
        "hip_dll_sha256": hip_info["sha256"],
        "benchmark_environment": "desktop-active",
        "benchmark_policy": {
            "background_gpu_memory_is_recorded_not_threshold_blocked": True,
            "missing_or_empty_token_receipts_are_failures": True,
        },
        "self_comparison": self_comparison,
        "mtp_off_vs_on": mtp_comparison,
        "identical": all(item["identical"] for item in mtp_comparison),
    }
    atomic_write_json(output_dir / f"{prefix}-comparison.json", summary)
    print(f"self_identical=true mtp_identical={str(summary['identical']).lower()}")
    return 0 if summary["identical"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
