import hashlib
import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_fit  # noqa: E402
from run_fit import load_residency_evidence, parse_counter_output  # noqa: E402


def test_counter_parser_retains_raw_dedicated_and_shared_readings():
    raw = json.dumps([
        {"metric": "dedicated", "instance": "pid_42_luid_0", "pid": 42,
         "process_name": "llama-server", "cooked_value": 100},
        {"metric": "shared", "instance": "pid_42_luid_0", "pid": 42,
         "process_name": "llama-server", "cooked_value": 5.5},
    ])
    samples = parse_counter_output(raw)
    assert samples == [
        {"metric": "dedicated", "instance": "pid_42_luid_0", "pid": 42,
         "process_name": "llama-server", "cooked_value": 100.0},
        {"metric": "shared", "instance": "pid_42_luid_0", "pid": 42,
         "process_name": "llama-server", "cooked_value": 5.5},
    ]


def test_counter_parser_rejects_nonfinite_values():
    try:
        parse_counter_output('[{"metric":"shared","instance":"pid_1","cooked_value":NaN}]')
    except ValueError as exc:
        assert "JSON" in str(exc) or "finite" in str(exc)
    else:
        raise AssertionError("nonfinite counter readings must fail")


def test_residency_evidence_is_bound_to_model_and_context():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evidence.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "classification": "device-resident",
            "model_sha256": "a" * 64,
            "context_tokens": 262144,
            "method": "external allocator audit",
            "source": "prerequisite-lane/report.json",
            "observed_at_utc": "2026-09-21T00:00:00Z",
        }), encoding="utf-8")
        evidence = load_residency_evidence(path, "a" * 64, 262144)
        assert evidence["classification"] == "device-resident"
        try:
            load_residency_evidence(path, "b" * 64, 262144)
        except ValueError as exc:
            assert "model" in str(exc)
        else:
            raise AssertionError("cross-model evidence must fail")


def test_residency_evidence_requires_method_and_source():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evidence.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "classification": "spill",
            "model_sha256": "a" * 64,
            "context_tokens": 262144,
            "method": "",
            "source": "",
            "observed_at_utc": "2026-09-21T00:00:00Z",
        }), encoding="utf-8")
        try:
            load_residency_evidence(path, "a" * 64, 262144)
        except ValueError as exc:
            assert "method" in str(exc) or "source" in str(exc)
        else:
            raise AssertionError("unattributed evidence must fail")


@pytest.mark.parametrize(
    ("exit_code", "expected_code", "expected_decision"),
    [(None, 3, "inconclusive"), (0, 2, "exit"), (1, 2, "exit")],
)
def test_fit_receipt_records_desktop_active_occupancy_before_and_after(
    exit_code, expected_code, expected_decision
):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model = tmp / "model.gguf"
        binary = tmp / "llama-server.exe"
        dll = tmp / "ggml-hip.dll"
        output = tmp / "fit.json"
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
        snapshots = [
            {"recorded_at_utc": "before", "classification": "desktop-active", "rows": [
                {"metric": "dedicated", "instance": "pid_42", "pid": 42,
                 "process_name": "browser", "cooked_value": float(2 * 1024**3)},
            ]},
            {"recorded_at_utc": "after", "classification": "desktop-active", "rows": []},
        ]

        class Process:
            def poll(self):
                return exit_code

        @contextmanager
        def fake_server(*_args, **_kwargs):
            yield {
                "pid": 99,
                "base_url": "http://127.0.0.1:8123",
                "started_at_utc": "start",
                "health": {"status": "ok"},
                "process": Process(),
            }

        with (
            mock.patch.object(run_fit, "MODEL_SPECS", {"pq2": spec}),
            mock.patch.object(run_fit, "environment", return_value={}),
            mock.patch.object(run_fit, "ensure_no_llama_processes", return_value=[]),
            mock.patch.object(run_fit, "managed_server", fake_server),
            mock.patch.object(run_fit, "collect_gpu_counter_samples", return_value=[]),
            mock.patch.object(run_fit, "capture_gpu_process_memory", side_effect=snapshots),
        ):
            code = run_fit.main([
                "--model", "pq2",
                "--context", "32768",
                "--binary", str(binary),
                "--hip-dll", str(dll),
                "--expected-hip-dll-sha256", digest(dll),
                "--counter-samples", "1",
                "--counter-interval", "0",
                "--output", str(output),
            ])

        assert code == expected_code
        receipt = json.loads(output.read_text(encoding="utf-8"))
        assert receipt["decision"] == expected_decision
        assert receipt["benchmark_environment"] == "desktop-active"
        assert receipt["background_occupancy"]["before"]["rows"][0]["cooked_value"] > 1024**3
        assert receipt["background_occupancy"]["after"]["recorded_at_utc"] == "after"


if __name__ == "__main__":
    test_counter_parser_retains_raw_dedicated_and_shared_readings()
    test_counter_parser_rejects_nonfinite_values()
    test_residency_evidence_is_bound_to_model_and_context()
    test_residency_evidence_requires_method_and_source()
    print("ok")
