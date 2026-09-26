"""Same-session paired tg128, frozen versus candidate: interleaved then order-reversed."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "serve"))

from common import (  # noqa: E402
    MODEL_SPECS,
    ROOT,
    atomic_write_json,
    capture_gpu_process_memory,
    ensure_no_llama_processes,
    environment,
    file_provenance,
    utc_now,
)


def sibling_dll(binary):
    """The ggml-hip.dll that the llama-bench.exe launcher next to it actually loads."""
    dll = Path(binary).parent / "ggml-hip.dll"
    return file_provenance(dll) if dll.is_file() else None


def llama_cpp_commit():
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT / "_work" / "llama.cpp"), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip() or None
    except Exception:
        return None


def evaluator_provenance():
    """Bind the measuring code itself, so a receipt names the revision that produced it."""
    return {"script": file_provenance(Path(__file__)), "llama_cpp_commit": llama_cpp_commit()}


def bench_argv(binary, model, depth):
    return [
        str(binary), "-m", str(model), "-p", "512", "-n", "128", "-d", str(depth),
        "-r", "1", "-o", "json", "-ngl", "999", "-dev", "ROCm0", "-sm", "none",
        "-fa", "on", "-b", "2048", "-ub", "512", "-t", "8", "-ctk", "q4_0", "-ctv", "q4_0",
    ]


def parse_tg128(text):
    """Return (tg128, pp512) from llama-bench -o json output.

    llama-bench emits separate rows for the prompt test (n_prompt=512, n_gen=0) and the
    generation test (n_prompt=0, n_gen=128), both at the requested depth.
    """
    rows = json.loads(text)
    tg = pp = None
    for row in rows:
        if int(row.get("n_prompt", -1)) == 512 and int(row.get("n_gen", -1)) == 0:
            pp = float(row["avg_ts"])
        if int(row.get("n_prompt", -1)) == 0 and int(row.get("n_gen", -1)) == 128:
            tg = float(row["avg_ts"])
    if tg is None or pp is None:
        raise ValueError("llama-bench output has no split 512/128 row pair")
    return tg, pp


def run_one(binary, model, depth, label, receipt_dir, expected=None):
    argv = bench_argv(binary, model, depth)
    # Bind every artifact at run time, from the files the launcher is about to load.
    bound = {
        "binary": file_provenance(Path(binary)),
        "ggml_hip_dll": sibling_dll(binary),
        "model": file_provenance(Path(model)),
        "evaluator": evaluator_provenance(),
    }
    if expected:
        for field, want in expected.items():
            got = (bound.get(field) or {}).get("sha256")
            if got != want:
                raise RuntimeError(f"{label} {field} sha256 {got} != expected {want}")
    completed = subprocess.run(
        argv, env=environment(), capture_output=True, text=True, cwd=str(Path(binary).parent)
    )
    raw = {"label": label, "recorded_at_utc": utc_now(), "argv": argv,
           "returncode": completed.returncode, "stdout": completed.stdout,
           "stderr": completed.stderr, "bound_provenance": bound,
           "expected_provenance": expected}
    if completed.returncode != 0:
        atomic_write_json(receipt_dir / f"{label}.json", raw)
        raise RuntimeError(f"{label} llama-bench failed with exit {completed.returncode}")
    tg128, pp512 = parse_tg128(completed.stdout)
    raw["tg128"] = tg128
    raw["pp512"] = pp512
    raw["bound_provenance_after"] = {
        "binary": file_provenance(Path(binary)),
        "ggml_hip_dll": sibling_dll(binary),
    }
    atomic_write_json(receipt_dir / f"{label}.json", raw)
    return tg128


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Paired tg128, baseline versus candidate: interleaved then order-reversed."
    )
    parser.add_argument("--model", default="ptq1-mtp", choices=tuple(MODEL_SPECS))
    parser.add_argument("--depth", type=int, default=32768)
    parser.add_argument("--baseline-binary", "--frozen-binary", dest="baseline_binary",
                        type=Path, required=True)
    parser.add_argument("--baseline-label", default="frozen",
                        help="what the baseline slot is; 'frozen' or 'control'")
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", default=None)
    parser.add_argument("--expected-candidate-dll-sha256", default=None)
    parser.add_argument("--expected-baseline-dll-sha256", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    spec = MODEL_SPECS[args.model]
    receipt_dir = args.output_dir.resolve()
    receipt_dir.mkdir(parents=True, exist_ok=True)
    model_info = file_provenance(spec["path"], spec["sha256"])
    baseline_info = file_provenance(args.baseline_binary)
    candidate_info = file_provenance(args.candidate_binary, args.expected_candidate_sha256)

    preflight = ensure_no_llama_processes()
    occupancy_before = capture_gpu_process_memory()

    label_baseline = args.baseline_label
    order = [(label_baseline, "candidate")] * 3 + [("candidate", label_baseline)] * 3
    binaries = {label_baseline: args.baseline_binary, "candidate": args.candidate_binary}
    expected = {
        label_baseline: {"binary": baseline_info["sha256"],
                         "ggml_hip_dll": args.expected_baseline_dll_sha256},
        "candidate": {"binary": candidate_info["sha256"],
                      "ggml_hip_dll": args.expected_candidate_dll_sha256},
    }
    for slot in expected:
        expected[slot] = {k: v for k, v in expected[slot].items() if v is not None}
    passes = []
    for index, pair in enumerate(order, start=1):
        measured = {}
        for slot in pair:
            label = f"pass{index}-{slot}"
            measured[slot] = run_one(binaries[slot], spec["path"], args.depth, label,
                                     receipt_dir, expected[slot])
        delta = (measured["candidate"] - measured[label_baseline]) / measured[label_baseline] * 100.0
        passes.append({"pass": index, "order": list(pair), "tg128": measured,
                       "delta_percent": delta})
        print(f"pass {index} order={pair} {label_baseline}={measured[label_baseline]:.4f} "
              f"candidate={measured['candidate']:.4f} delta={delta:+.3f}%", flush=True)

    deltas = [item["delta_percent"] for item in passes]
    summary = {
        "schema_version": 2,
        "recorded_at_utc": utc_now(),
        "benchmark_environment": "desktop-active",
        "model": args.model,
        "model_provenance": model_info,
        "depth_tokens": args.depth,
        "n_prompt": 512,
        "n_gen": 128,
        "repetitions_per_pass": 1,
        "mtp": "off",
        "comparison": f"candidate versus {label_baseline}",
        "baseline_label": label_baseline,
        "baseline_binary": baseline_info,
        "baseline_ggml_hip_dll": sibling_dll(args.baseline_binary),
        "candidate_binary": candidate_info,
        "candidate_ggml_hip_dll": sibling_dll(args.candidate_binary),
        "evaluator": evaluator_provenance(),
        "preflight_llama_processes": preflight,
        "background_occupancy": {"before": occupancy_before, "after": None},
        "passes": passes,
        "delta_percent_samples": deltas,
        "delta_percent_median": statistics.median(deltas),
        "delta_percent_min": min(deltas),
        "delta_percent_max": max(deltas),
        "keep_threshold_percent": 5.0,
        "keep": statistics.median(deltas) >= 5.0,
    }
    try:
        summary["background_occupancy"]["after"] = capture_gpu_process_memory()
    except Exception as exc:
        summary["background_occupancy"]["after_error"] = f"{type(exc).__name__}: {exc}"
    atomic_write_json(receipt_dir / "paired-summary.json", summary)
    print(f"median delta {summary['delta_percent_median']:+.3f}% "
          f"keep={summary['keep']} (min {summary['delta_percent_min']:+.3f}%, "
          f"max {summary['delta_percent_max']:+.3f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
