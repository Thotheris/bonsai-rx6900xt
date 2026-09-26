"""Run hash-bound pp512/tg128 fill curves for a source-built Bonsai model."""

import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "serve"))
sys.path.insert(0, str(ROOT / "graft"))

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
    request_json,
    server_argv,
    utc_now,
)
from bit_identical import validate_completion  # noqa: E402
from fill_curve import CHECKPOINTS, PP, REPS, TG, headline, row_from_timings  # noqa: E402

LLAMA_ROOT = ROOT / "_work" / "llama.cpp"


def _compact_request(request):
    if not isinstance(request, dict) or not isinstance(request.get("prompt"), list):
        return request
    prompt = request["prompt"]
    compact = {key: value for key, value in request.items() if key != "prompt"}
    compact["prompt_n"] = len(prompt)
    compact["prompt_head"] = prompt[:2]
    compact["prompt_tail"] = prompt[-2:]
    return compact


def _integer_tokens(value, name):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must contain a non-empty token list")
    if any(isinstance(token, bool) or not isinstance(token, int) for token in value):
        raise ValueError(f"{name} tokens must be integer ids")
    return value


def tokenize_fill_tokens(base_url, *, request_fn=request_json, timeout=120.0):
    """Return a pad id and a distinct EOS sentinel for cache setup."""
    props = request_fn(base_url, "/props", timeout=timeout)
    eos_text = props.get("eos_token")
    if not isinstance(eos_text, str) or not eos_text:
        raise ValueError("/props did not provide a non-empty eos_token")

    pad_response = request_fn(
        base_url,
        "/tokenize",
        {"content": " hello", "add_special": False},
        timeout=timeout,
    )
    eos_response = request_fn(
        base_url,
        "/tokenize",
        {"content": eos_text, "add_special": False},
        timeout=timeout,
    )
    pad_tokens = _integer_tokens(pad_response.get("tokens"), "/tokenize pad response")
    eos_tokens = _integer_tokens(eos_response.get("tokens"), "/tokenize EOS response")
    pad = pad_tokens[-1]
    if pad in eos_tokens:
        raise ValueError("the selected pad token is an EOS token id")
    return pad, eos_tokens[-1]


def tokenize_pad(base_url, *, request_fn=request_json, timeout=120.0):
    """Compatibility helper returning only the validated pad id."""
    return tokenize_fill_tokens(base_url, request_fn=request_fn, timeout=timeout)[0]


def _base_request(prompt, n_predict, *, cache_prompt):
    return {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "seed": 42,
        "cache_prompt": cache_prompt,
        "ignore_eos": True,
        "return_tokens": True,
        "stream": False,
        "response_fields": [
            "content",
            "tokens",
            "timings",
            "truncated",
            "stop_type",
            "tokens_cached",
            "tokens_evaluated",
        ],
    }


def fill_request(depth, pad, sentinel):
    if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
        raise ValueError("fill depth must be a positive integer")
    if any(isinstance(token, bool) or not isinstance(token, int) for token in (pad, sentinel)):
        raise ValueError("pad and sentinel must be integer token ids")
    if pad == sentinel:
        raise ValueError("fill sentinel must differ from pad")
    # llama.cpp checkpoints 4 tokens before the prompt end and the next
    # request restores that checkpoint, not the full common prefix.
    # ponytail: tail length is that source constant (server-context.cpp
    # checkpoint_offsets). Change it if that offset changes.
    tail = 4
    return _base_request([pad] * depth + [sentinel] * tail, 0, cache_prompt=False)


def measure_request(depth, pad):
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError("measurement depth must be a non-negative integer")
    if isinstance(pad, bool) or not isinstance(pad, int):
        raise ValueError("pad must be an integer token id")
    return _base_request([pad] * (depth + PP), TG, cache_prompt=depth > 0)


def _validate_fill_response(response, depth):
    response = validate_completion(response)
    # Fill prompt is [pad]*depth + 4 sentinels so the restored checkpoint lands on depth.
    expected_prompt_n = depth + 4
    if response["timings"]["cache_n"] != 0:
        raise ValueError("fill request unexpectedly reused cached prompt tokens")
    if response["timings"]["prompt_n"] != expected_prompt_n:
        raise ValueError("fill request did not process the sentinel-extended depth")
    if response["timings"]["predicted_n"] != 1 or len(response["tokens"]) != 1:
        raise ValueError("local n_predict=0 semantics changed; expected one sampled token")
    return response


def measure_depth(base_url, depth, pad, sentinel, window, *, request_fn=request_json, timeout=3600.0, on_step=None):
    """Fill one depth, then score exactly pp512/tg128."""
    if depth + PP + TG > window:
        raise ValueError("depth plus pp512/tg128 sample exceeds fitted window")
    if depth > 0 and (isinstance(sentinel, bool) or not isinstance(sentinel, int) or sentinel == pad):
        raise ValueError("fill sentinel must be an integer token id different from pad")

    fill_response = None
    if depth > 0:
        fill_response = _validate_fill_response(
            request_fn(base_url, "/completion", fill_request(depth, pad, sentinel), timeout=timeout),
            depth,
        )
        if on_step is not None:
            on_step("fill", fill_response)

    response = validate_completion(
        request_fn(base_url, "/completion", measure_request(depth, pad), timeout=timeout)
    )
    if len(response["tokens"]) != TG:
        raise ValueError(f"measurement token receipt must contain exactly {TG} ids")
    for key in ("tokens_cached", "tokens_evaluated", "truncated"):
        if key not in response:
            raise ValueError(f"measurement response is missing {key}")

    row = row_from_timings(
        depth,
        response["timings"],
        window=window,
        truncated=response.get("truncated"),
        tokens_cached=response.get("tokens_cached"),
        tokens_evaluated=response.get("tokens_evaluated"),
    )
    result = {
        "depth": depth,
        "fill_request": fill_request(depth, pad, sentinel) if depth > 0 else None,
        "fill_response": fill_response,
        "measurement_request": measure_request(depth, pad),
        "measurement_response": response,
        "row": row,
    }
    if on_step is not None:
        on_step("measure", result)
    return result


def _line_number(text, marker):
    index = text.find(marker)
    if index < 0:
        return None
    return text.count("\n", 0, index) + 1


def inspect_source_semantics(llama_root=LLAMA_ROOT):
    """Verify the local endpoint semantics the fill protocol relies on."""
    llama_root = Path(llama_root)
    context_path = llama_root / "tools" / "server" / "server-context.cpp"
    common_path = llama_root / "tools" / "server" / "server-common.cpp"
    context = context_path.read_text(encoding="utf-8")
    common = common_path.read_text(encoding="utf-8")

    full_prompt_marker = "if (n_past == slot.task->n_tokens() && n_past > 0)"
    decrement_marker = "n_past--;"
    add_token_marker = "slot.add_token(result);"
    budget_marker = "if (slot.stats.n_gen > 0 && slot.has_next_token && !slot.has_budget())"
    timing_markers = (
        '{"cache_n",                n_prompt_cached}',
        '{"prompt_n",               n_prompt_processed}',
        '{"predicted_n",            n_gen}',
    )

    full_line = _line_number(context, full_prompt_marker)
    decrement_line = _line_number(context, decrement_marker)
    add_line = _line_number(context, add_token_marker)
    budget_line = _line_number(context, budget_marker)
    result = {
        "cached_full_prompt_decrements_n_past": (
            full_line is not None and decrement_line is not None and full_line < decrement_line
        ),
        "n_predict_zero_samples_before_budget_check": (
            add_line is not None and budget_line is not None and add_line < budget_line
        ),
        "timing_fields_map_to_cached_processed_generated": all(marker in common for marker in timing_markers),
        "evidence_lines": {
            "full_prompt_check": full_line,
            "n_past_decrement": decrement_line,
            "sample_recorded": add_line,
            "budget_check": budget_line,
        },
        "files": [file_provenance(context_path), file_provenance(common_path)],
    }
    if not all(
        result[key]
        for key in (
            "cached_full_prompt_decrements_n_past",
            "n_predict_zero_samples_before_budget_check",
            "timing_fields_map_to_cached_processed_generated",
        )
    ):
        raise RuntimeError("local llama-server endpoint semantics no longer match the fill protocol")
    return result


def _reject_json_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def load_fit_receipt(path, model_sha256, hip_dll_sha256, context):
    """Require a model/context/DLL-bound fit decision before timing."""
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid fit receipt: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("fit receipt must be a schema_version 1 object")
    if value.get("decision") != "fit":
        raise ValueError("fill timing requires a fit receipt whose decision is fit")
    try:
        provenance = value["provenance"]
        receipt_model = provenance["model"]["sha256"]
        receipt_dll = provenance["hip_dll"]["sha256"]
        receipt_context = provenance["context_tokens"]
    except (KeyError, TypeError) as exc:
        raise ValueError("fit receipt is missing provenance") from exc
    if receipt_model != model_sha256:
        raise ValueError("fit receipt model SHA-256 does not match")
    if receipt_dll != hip_dll_sha256:
        raise ValueError("fit receipt HIP DLL SHA-256 does not match")
    if receipt_context != context:
        raise ValueError("fit receipt context does not match")
    return value


def _parser():
    parser = argparse.ArgumentParser(
        description="Measure three pp512/tg128 reps at the Bonsai fill checkpoints.",
    )
    parser.add_argument("--model", required=True, choices=tuple(MODEL_SPECS))
    parser.add_argument("--context", required=True, type=int)
    parser.add_argument("--mtp", choices=("auto", "off", "on"), default="auto")
    parser.add_argument("--port", type=int)
    parser.add_argument("--binary", type=Path, default=BINARY)
    parser.add_argument("--hip-dll", type=Path, default=ACTIVE_DLL)
    parser.add_argument("--expected-hip-dll-sha256")
    parser.add_argument("--exploratory", action="store_true",
                        help="record desktop-active timings without a fit receipt; never a resident headline")
    parser.add_argument("--depths", default="",
                        help="comma-separated depths; default is the standard checkpoint list")
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument(
        "--fit-receipt",
        type=Path,
        help="required unless --exploratory: schema_version 1 receipt with decision=fit",
    )
    parser.add_argument("--llama-root", type=Path, default=LLAMA_ROOT)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--output", type=Path)
    return parser


def _selected_spec(model_name, mtp):
    spec = MODEL_SPECS[model_name]
    if mtp == "auto":
        selected = spec["default_spec"]
    else:
        selected = "draft-mtp" if mtp == "on" else "none"
    if model_name in {"pq2", "ptq1"} and selected != "none":
        raise ValueError(f"the ungrafted {model_name} model has no MTP-on arm")
    return selected


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.context <= 0 or args.context > WINDOW:
        raise SystemExit(f"--context must be between 1 and {WINDOW}")
    if args.startup_timeout <= 0 or args.request_timeout <= 0:
        raise SystemExit("timeouts must be positive")
    if not args.exploratory and args.fit_receipt is None:
        raise SystemExit("strict fill requires --fit-receipt with decision=fit")
    if args.exploratory:
        if isinstance(args.reps, bool) or args.reps < 1:
            raise SystemExit("--reps must be a positive integer")
        reps = args.reps
    else:
        if args.reps != REPS:
            raise SystemExit("strict fill uses 3 reps")
        reps = REPS
    if args.depths:
        try:
            selected_depths = tuple(int(part) for part in args.depths.split(",") if part != "")
        except ValueError as exc:
            raise SystemExit("--depths must be comma-separated integers") from exc
        if not selected_depths or any(depth < 0 for depth in selected_depths):
            raise SystemExit("--depths must be non-negative integers")
    else:
        selected_depths = CHECKPOINTS

    try:
        selected_spec = _selected_spec(args.model, args.mtp)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    spec = MODEL_SPECS[args.model]
    port = args.port or spec["port"]
    model_info = file_provenance(spec["path"], spec["sha256"])
    binary_info = file_provenance(args.binary)
    hip_info = file_provenance(args.hip_dll, args.expected_hip_dll_sha256)
    fit_receipt = None
    fit_receipt_info = None
    if not args.exploratory:
        fit_receipt_info = file_provenance(args.fit_receipt)
        fit_receipt = load_fit_receipt(
            args.fit_receipt,
            model_info["sha256"],
            hip_info["sha256"],
            args.context,
        )
    source_semantics = inspect_source_semantics(args.llama_root)
    command = server_argv(spec["path"], args.context, port, spec=selected_spec, binary=args.binary)
    arm = f"{args.model}-{'on' if selected_spec == 'draft-mtp' else 'off'}"
    output = (args.output or ROOT / "results" / "fill" / f"{arm}.json").resolve()

    receipt = {
        "schema_version": 1,
        "recorded_at_utc": utc_now(),
        "status": "running",
        "arm": arm,
        "claim": "residency-unverified" if args.exploratory else "resident-fit-required",
        "resident_headline": False if args.exploratory else None,
        "benchmark_environment": "desktop-active",
        "benchmark_policy": {
            "background_gpu_memory_is_recorded_not_threshold_blocked": True,
            "actual_oom_or_unhealthy_server_is_a_failure": True,
            "three_valid_repetitions_are_required_for_a_median": True,
        },
        "protocol": {
            "checkpoints": list(CHECKPOINTS),
            "window_tokens": args.context,
            "pp_tokens": PP,
            "tg_tokens": TG,
            "repetitions": reps,
            "headline": "not-a-resident-headline" if args.exploratory else "median-of-exactly-three-valid-reps",
        },
        "provenance": {
            "model": model_info,
            "binary": binary_info,
            "hip_dll": hip_info,
            "fit_receipt_file": fit_receipt_info,
            "fit_decision": None if args.exploratory else fit_receipt["decision"],
            "fit_residency_evidence": None if args.exploratory else copy.deepcopy(fit_receipt.get("residency_evidence")),
            "server_argv": command,
            "source_semantics": source_semantics,
        },
        "repetitions": [],
        "headlines": [],
    }

    eligible = [depth for depth in selected_depths if depth + PP + TG <= args.context]
    atomic_write_json(output, receipt)
    try:
        for rep_index in range(reps):
            rep_receipt = {
                "rep": rep_index + 1,
                "recorded_at_utc": utc_now(),
                "background_occupancy": {"before": None, "after": None},
                "server": None,
                "pad_token_id": None,
                "depths": [],
            }
            receipt["repetitions"].append(rep_receipt)
            atomic_write_json(output, receipt)
            rep_receipt["preflight_llama_processes"] = ensure_no_llama_processes()
            rep_receipt["background_occupancy"]["before"] = capture_gpu_process_memory()
            log_path = output.with_name(f"{output.stem}.rep-{rep_index + 1}.server.log")
            try:
                with managed_server(
                    command,
                    environment(),
                    port,
                    log_path,
                    startup_timeout=args.startup_timeout,
                ) as running:
                    pad, sentinel = tokenize_fill_tokens(
                        running["base_url"],
                        request_fn=request_json,
                        timeout=args.request_timeout,
                    )
                    rep_receipt["server"] = {
                        key: copy.deepcopy(running[key])
                        for key in ("pid", "base_url", "started_at_utc", "health")
                    }
                    rep_receipt["pad_token_id"] = pad
                    rep_receipt["sentinel_token_id"] = sentinel
                    atomic_write_json(output, receipt)
                    for depth in eligible:
                        def on_step(phase, payload, depth=depth):
                            if phase == "fill":
                                rep_receipt["open_fill"] = {"depth": depth, "response": payload}
                            atomic_write_json(output, receipt)

                        result = measure_depth(
                            running["base_url"],
                            depth,
                            pad,
                            sentinel,
                            args.context,
                            request_fn=request_json,
                            timeout=args.request_timeout,
                            on_step=on_step,
                        )
                        saved = dict(result)
                        saved["fill_request"] = _compact_request(result.get("fill_request"))
                        saved["measurement_request"] = _compact_request(result.get("measurement_request"))
                        rep_receipt["depths"].append(saved)
                        rep_receipt["open_fill"] = None
                        atomic_write_json(output, receipt)
            finally:
                try:
                    rep_receipt["background_occupancy"]["after"] = capture_gpu_process_memory()
                except Exception as snapshot_exc:
                    rep_receipt["background_occupancy"]["after_error"] = (
                        f"{type(snapshot_exc).__name__}: {snapshot_exc}"
                    )
                atomic_write_json(output, receipt)
    except Exception as exc:
        receipt["status"] = "partial" if args.exploratory else "invalid"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(output, receipt)
        print(f"status={receipt['status']} receipt={output}", file=sys.stderr)
        return 2

    for depth in selected_depths:
        if depth not in eligible:
            receipt["headlines"].append({
                "depth": depth,
                "status": "omitted",
                "reason": "depth plus pp512/tg128 exceeds window",
                "resident_headline": False,
            })
            continue
        rows = [
            rep["depths"][eligible.index(depth)]["row"]
            for rep in receipt["repetitions"]
        ]
        summary = headline(rows)
        if args.exploratory:
            summary.update({
                "depth": depth,
                "status": "residency-unverified" if summary["valid"] else "invalid",
                "resident_headline": False,
            })
        else:
            summary.update({"depth": depth, "status": "valid" if summary["valid"] else "invalid"})
        receipt["headlines"].append(summary)

    complete = all(
        row["status"] != "omitted" and
        (row["valid"] if not args.exploratory else row["valid_reps"] == reps)
        for row in receipt["headlines"]
    )
    if args.exploratory:
        receipt["status"] = "residency-unverified" if complete else "invalid"
        receipt["resident_headline"] = False
    else:
        receipt["status"] = "valid" if complete else "invalid"
        receipt["resident_headline"] = complete
    atomic_write_json(output, receipt)
    print(f"status={receipt['status']} receipt={output}")
    return 0 if receipt["status"] in {"valid", "residency-unverified"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
