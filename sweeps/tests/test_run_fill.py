import hashlib
import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_fill  # noqa: E402
from run_fill import (  # noqa: E402
    fill_request,
    load_fit_receipt,
    measure_depth,
    measure_request,
    tokenize_fill_tokens,
    tokenize_pad,
)


class _Requester:
    def __init__(self):
        self.calls = []

    def __call__(self, base_url, route, payload=None, **_kwargs):
        self.calls.append((base_url, route, payload))
        if route == "/props":
            return {"eos_token": "<eos>"}
        if route == "/tokenize":
            if payload["content"] == " hello":
                return {"tokens": [10, 11]}
            return {"tokens": [99]}
        if route == "/completion" and payload["n_predict"] == 0:
            # This fork samples one token despite n_predict=0. The runner must
            # retain but not score this response.
            return {
                "content": "x",
                "tokens": [1],
                "timings": {
                    "cache_n": 0,
                    "prompt_n": len(payload["prompt"]),
                    "prompt_per_second": 300.0,
                    "predicted_n": 1,
                    "predicted_per_second": 40.0,
                },
            }
        if route == "/completion":
            depth = len(payload["prompt"]) - 512
            return {
                "content": "x" * 128,
                "tokens": list(range(128)),
                "timings": {
                    "cache_n": depth,
                    "prompt_n": 512,
                    "prompt_per_second": 300.0,
                    "predicted_n": 128,
                    "predicted_per_second": 40.0,
                },
                "truncated": False,
                "tokens_cached": depth + 512,
                "tokens_evaluated": depth + 512,
            }
        raise AssertionError(route)


def test_tokenize_pad_rejects_eos_and_returns_last_id():
    requester = _Requester()
    assert tokenize_fill_tokens("http://server", request_fn=requester) == (11, 99)
    requester = _Requester()
    assert tokenize_pad("http://server", request_fn=requester) == 11
    assert requester.calls[0][1] == "/props"
    assert requester.calls[1][1] == "/tokenize"
    assert requester.calls[2][2]["content"] == "<eos>"

    class EosRequester(_Requester):
        def __call__(self, base_url, route, payload=None, **kwargs):
            result = super().__call__(base_url, route, payload, **kwargs)
            if route == "/tokenize" and payload["content"] == "<eos>":
                return {"tokens": [11]}
            return result

    try:
        tokenize_pad("http://server", request_fn=EosRequester())
    except ValueError as exc:
        assert "EOS" in str(exc)
    else:
        raise AssertionError("EOS cannot be used as the pad token")


def test_fill_and_measure_requests_pin_protocol():
    fill = fill_request(8, 11, 99)
    assert fill["prompt"] == [11] * 8 + [99] * 4
    assert fill["n_predict"] == 0
    assert fill["cache_prompt"] is False

    depth0 = measure_request(0, 11)
    depth8 = measure_request(8, 11)
    assert depth0["cache_prompt"] is False
    assert depth8["cache_prompt"] is True
    assert depth8["n_predict"] == 128
    assert depth8["ignore_eos"] is True
    assert len(depth8["prompt"]) == 8 + 512


def test_measure_depth_ignores_fill_token_and_scores_exact_counts():
    requester = _Requester()
    result = measure_depth("http://server", 8, 11, 99, 262144, request_fn=requester)
    assert result["row"]["valid"] is True
    assert result["fill_response"]["timings"]["predicted_n"] == 1
    assert [call[2]["n_predict"] for call in requester.calls] == [0, 128]

    requester = _Requester()
    result = measure_depth("http://server", 0, 11, 99, 262144, request_fn=requester)
    assert result["row"]["valid"] is True
    assert result["fill_response"] is None
    assert len(requester.calls) == 1


def test_measure_depth_rejects_missing_tokens_and_nonfinite_timing():
    class BadRequester(_Requester):
        def __init__(self, mode):
            super().__init__()
            self.mode = mode

        def __call__(self, base_url, route, payload=None, **kwargs):
            response = super().__call__(base_url, route, payload, **kwargs)
            if route == "/completion" and payload["n_predict"] == 128:
                if self.mode == "tokens":
                    del response["tokens"]
                else:
                    response["timings"]["predicted_per_second"] = float("nan")
            return response

    for mode in ("tokens", "timing"):
        try:
            measure_depth("http://server", 0, 11, 99, 262144, request_fn=BadRequester(mode))
        except ValueError as exc:
            assert "tokens" in str(exc) or "finite" in str(exc)
        else:
            raise AssertionError(f"invalid {mode} receipt was accepted")

    class MissingAuditRequester(_Requester):
        def __call__(self, base_url, route, payload=None, **kwargs):
            response = super().__call__(base_url, route, payload, **kwargs)
            if route == "/completion" and payload["n_predict"] == 128:
                del response["tokens_cached"]
            return response

    try:
        measure_depth("http://server", 0, 11, 99, 262144, request_fn=MissingAuditRequester())
    except ValueError as exc:
        assert "tokens_cached" in str(exc)
    else:
        raise AssertionError("missing cached-token audit field was accepted")




def test_fill_requires_a_fit_receipt_bound_to_model_context_and_dll():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fit.json"
        base = {
            "schema_version": 1,
            "decision": "inconclusive",
            "provenance": {
                "model": {"sha256": "a" * 64},
                "hip_dll": {"sha256": "b" * 64},
                "context_tokens": 262144,
            },
        }
        path.write_text(json.dumps(base), encoding="utf-8")
        try:
            load_fit_receipt(path, "a" * 64, "b" * 64, 262144)
        except ValueError as exc:
            assert "fit" in str(exc)
        else:
            raise AssertionError("an inconclusive fit receipt must not authorize a fill run")

        base["decision"] = "fit"
        path.write_text(json.dumps(base), encoding="utf-8")
        receipt = load_fit_receipt(path, "a" * 64, "b" * 64, 262144)
        assert receipt["decision"] == "fit"


def test_fill_runner_records_desktop_active_occupancy_for_each_rep():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model = tmp / "model.gguf"
        binary = tmp / "llama-server.exe"
        dll = tmp / "ggml-hip.dll"
        output = tmp / "fill.json"
        fit_receipt = tmp / "fit.json"
        model.write_bytes(b"model")
        binary.write_bytes(b"binary")
        dll.write_bytes(b"dll")
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        spec = {
            "path": model,
            "sha256": digest(model),
            "port": 8123,
            "default_spec": "none",
        }
        fit_receipt.write_text(json.dumps({
            "schema_version": 1,
            "decision": "fit",
            "provenance": {
                "model": {"sha256": digest(model)},
                "hip_dll": {"sha256": digest(dll)},
                "context_tokens": 32768,
            },
        }), encoding="utf-8")
        snapshot = {
            "recorded_at_utc": "now",
            "classification": "desktop-active",
            "rows": [{
                "metric": "dedicated", "instance": "pid_42", "pid": 42,
                "process_name": "browser", "cooked_value": float(2 * 1024**3),
            }],
        }

        @contextmanager
        def fake_server(*_args, **_kwargs):
            yield {
                "pid": 99,
                "base_url": "http://127.0.0.1:8123",
                "started_at_utc": "start",
                "health": {"status": "ok"},
            }

        def fake_measure(_url, depth, _pad, _sentinel, _window, **_kwargs):
            row = {
                "depth": depth,
                "valid": True,
                "pp512": 100.0,
                "tg128": 20.0,
            }
            return {"depth": depth, "row": row}

        with (
            mock.patch.object(run_fill, "MODEL_SPECS", {"pq2": spec}),
            mock.patch.object(run_fill, "environment", return_value={}),
            mock.patch.object(run_fill, "ensure_no_llama_processes", return_value=[]),
            mock.patch.object(run_fill, "managed_server", fake_server),
            mock.patch.object(run_fill, "tokenize_fill_tokens", return_value=(11, 99)),
            mock.patch.object(run_fill, "measure_depth", side_effect=fake_measure),
            mock.patch.object(run_fill, "inspect_source_semantics", return_value={"verified": True}),
            mock.patch.object(run_fill, "capture_gpu_process_memory", return_value=snapshot),
        ):
            code = run_fill.main([
                "--model", "pq2",
                "--context", "32768",
                "--binary", str(binary),
                "--hip-dll", str(dll),
                "--expected-hip-dll-sha256", digest(dll),
                "--fit-receipt", str(fit_receipt),
                "--output", str(output),
            ])
            exploratory_output = tmp / "exploratory.json"
            exploratory_code = run_fill.main([
                "--model", "pq2",
                "--context", "32768",
                "--binary", str(binary),
                "--hip-dll", str(dll),
                "--exploratory",
                "--reps", "1",
                "--depths", "0,32768",
                "--output", str(exploratory_output),
            ])
            invalid_output = tmp / "invalid-sample.json"
            with mock.patch.object(run_fill, "measure_depth", return_value={
                "depth": 0, "row": {"depth": 0, "valid": False, "pp512": None, "tg128": None},
            }):
                invalid_code = run_fill.main([
                    "--model", "pq2",
                    "--context", "32768",
                    "--binary", str(binary),
                    "--hip-dll", str(dll),
                    "--exploratory",
                    "--reps", "1",
                    "--depths", "0",
                    "--output", str(invalid_output),
                ])

        assert code == 2
        receipt = json.loads(output.read_text(encoding="utf-8"))
        assert receipt["status"] == "invalid"
        assert receipt["resident_headline"] is False
        assert exploratory_code == 2
        assert json.loads(exploratory_output.read_text(encoding="utf-8"))["status"] == "invalid"
        assert invalid_code == 2
        assert json.loads(invalid_output.read_text(encoding="utf-8"))["status"] == "invalid"
        assert receipt["benchmark_environment"] == "desktop-active"
        assert len(receipt["repetitions"]) == 3
        assert all(rep["background_occupancy"]["before"]["rows"] for rep in receipt["repetitions"])
        assert all(rep["background_occupancy"]["after"]["classification"] == "desktop-active"
                   for rep in receipt["repetitions"])
        assert receipt["headlines"][0]["valid_reps"] == 3
        assert receipt["headlines"][1]["status"] == "omitted"


if __name__ == "__main__":
    test_tokenize_pad_rejects_eos_and_returns_last_id()
    test_fill_and_measure_requests_pin_protocol()
    test_measure_depth_ignores_fill_token_and_scores_exact_counts()
    test_local_server_source_semantics_are_recorded()
    print("ok")
