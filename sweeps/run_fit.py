"""Launch one exact-context fit probe and write an auditable receipt."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "serve"))

from common import (  # noqa: E402
    ACTIVE_DLL,
    BINARY,
    MODEL_SPECS,
    WINDOW,
    atomic_write_json,
    capture_gpu_process_memory,
    ensure_no_llama_processes,
    environment,
    file_provenance,
    managed_server,
    parse_gpu_process_memory,
    server_argv,
    utc_now,
)
from fit import decide  # noqa: E402


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def parse_counter_output(text):
    return parse_gpu_process_memory(text)


def collect_gpu_counter_samples(pid, *, count=3, interval=1.0, runner=subprocess.run):
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("pid must be a positive integer")
    if count <= 0 or interval < 0:
        raise ValueError("counter count must be positive and interval non-negative")
    all_rows = []
    for index in range(count):
        snapshot = capture_gpu_process_memory(runner=runner)
        for source in snapshot["rows"]:
            if source["pid"] != pid:
                continue
            row = dict(source)
            row["recorded_at_utc"] = snapshot["recorded_at_utc"]
            row["sample_index"] = index
            all_rows.append(row)
        if index + 1 < count:
            time.sleep(interval)
    return all_rows


def load_residency_evidence(path, model_sha256, context):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid residency evidence file: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("residency evidence must be a schema_version 1 object")
    if value.get("classification") not in {"device-resident", "spill"}:
        raise ValueError("residency evidence classification must be device-resident or spill")
    if value.get("model_sha256") != model_sha256:
        raise ValueError("residency evidence model SHA-256 does not match the probed model")
    if value.get("context_tokens") != context:
        raise ValueError("residency evidence context does not match the probe")
    if not isinstance(value.get("method"), str) or not value["method"].strip():
        raise ValueError("residency evidence requires a non-empty method")
    if not isinstance(value.get("source"), str) or not value["source"].strip():
        raise ValueError("residency evidence requires a non-empty source")
    if not isinstance(value.get("observed_at_utc"), str) or not value["observed_at_utc"].strip():
        raise ValueError("residency evidence requires observed_at_utc")
    return value


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Probe one model/context. Windows shared-memory counters are recorded but "
            "the result stays inconclusive without independent residency evidence."
        ),
    )
    parser.add_argument("--model", required=True, choices=tuple(MODEL_SPECS))
    parser.add_argument("--context", type=int, default=WINDOW)
    parser.add_argument("--mtp", choices=("auto", "off", "on"), default="auto")
    parser.add_argument("--port", type=int)
    parser.add_argument("--binary", type=Path, default=BINARY)
    parser.add_argument("--hip-dll", type=Path, default=ACTIVE_DLL)
    parser.add_argument("--expected-hip-dll-sha256")
    parser.add_argument("--residency-evidence", type=Path)
    parser.add_argument("--counter-samples", type=int, default=3)
    parser.add_argument("--counter-interval", type=float, default=1.0)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.context <= 0:
        raise SystemExit("--context must be positive")
    spec = MODEL_SPECS[args.model]
    port = args.port or spec["port"]
    if args.mtp == "auto":
        server_spec = spec["default_spec"]
    else:
        server_spec = "draft-mtp" if args.mtp == "on" else "none"
    if args.model in {"pq2", "ptq1"} and server_spec != "none":
        raise SystemExit(f"the ungrafted {args.model} model has no MTP-on fit arm")

    model_info = file_provenance(spec["path"], spec["sha256"])
    binary_info = file_provenance(args.binary)
    hip_info = file_provenance(args.hip_dll, args.expected_hip_dll_sha256)
    evidence = None
    if args.residency_evidence:
        evidence = load_residency_evidence(args.residency_evidence, model_info["sha256"], args.context)

    command = server_argv(spec["path"], args.context, port, spec=server_spec, binary=args.binary)
    label = f"{args.model}-{'on' if server_spec == 'draft-mtp' else 'off'}-{args.context}"
    output = (args.output or ROOT / "results" / "fit" / f"{label}.json").resolve()
    receipt = {
        "schema_version": 1,
        "recorded_at_utc": utc_now(),
        "label": label,
        "provenance": {
            "model": model_info,
            "binary": binary_info,
            "hip_dll": hip_info,
            "server_argv": command,
            "context_tokens": args.context,
            "mtp": "on" if server_spec == "draft-mtp" else "off",
        },
        "residency_evidence": evidence,
        "benchmark_environment": "desktop-active",
        "benchmark_policy": {
            "background_gpu_memory_is_recorded_not_threshold_blocked": True,
            "actual_oom_or_unhealthy_server_is_a_failure": True,
            "shared_counters_alone_do_not_establish_residency": True,
        },
        "background_occupancy": {"before": None, "after": None},
        "counter_samples": [],
    }

    log_path = output.with_suffix(".server.log")
    try:
        receipt["preflight_llama_processes"] = ensure_no_llama_processes()
        receipt["background_occupancy"]["before"] = capture_gpu_process_memory()
        with managed_server(
            command,
            environment(),
            port,
            log_path,
            startup_timeout=args.startup_timeout,
        ) as running:
            receipt["server"] = {key: running[key] for key in ("pid", "base_url", "started_at_utc", "health")}
            receipt["counter_samples"] = collect_gpu_counter_samples(
                running["pid"],
                count=args.counter_samples,
                interval=args.counter_interval,
            )
            live_exit_code = running["process"].poll()
        receipt["background_occupancy"]["after"] = capture_gpu_process_memory()
    except Exception as exc:
        if receipt["background_occupancy"]["after"] is None:
            try:
                receipt["background_occupancy"]["after"] = capture_gpu_process_memory()
            except Exception as snapshot_exc:
                receipt["background_occupancy"]["after_error"] = (
                    f"{type(snapshot_exc).__name__}: {snapshot_exc}"
                )
        receipt["decision"] = "exit"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(output, receipt)
        print(f"decision=exit receipt={output}", file=sys.stderr)
        return 2

    shared_values = [
        row["cooked_value"] for row in receipt["counter_samples"] if row["metric"] == "shared"
    ]
    shared_max = max(shared_values, default=0.0)
    residency = evidence["classification"] if evidence else None
    receipt["shared_bytes_max"] = shared_max
    receipt["decision"] = decide(
        live_exit_code,
        shared_max,
        health_ok=live_exit_code is None,
        residency=residency,
    )
    atomic_write_json(output, receipt)
    print(f"decision={receipt['decision']} shared_bytes_max={shared_max:g} receipt={output}")
    if receipt["decision"] == "exit":
        return 2
    if receipt["decision"] == "inconclusive":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
