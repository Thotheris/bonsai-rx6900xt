import hashlib
import json
import sys
import tempfile
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_identity  # noqa: E402
from run_identity import capture_prompt, completion_request, read_prompts  # noqa: E402


class _Handler(BaseHTTPRequestHandler):
    bodies = []
    tokens = [7, 8] * 64

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        type(self).bodies.append((self.path, body))
        tokens = type(self).tokens
        response = {
            "content": "ok",
            "tokens": tokens,
            "timings": {
                "cache_n": 0,
                "prompt_n": 3,
                "prompt_per_second": 12.0,
                "predicted_n": len(tokens),
                "predicted_per_second": 4.0,
            },
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args):
        pass


def _provenance():
    return {
        "model": {"path": "same-mtp.gguf", "sha256": "a" * 64},
        "binary": {"path": "llama-server.exe", "sha256": "b" * 64},
        "hip_dll": {"path": "ggml-hip.dll", "sha256": "c" * 64},
        "server_argv": ["llama-server.exe"],
    }


def test_completion_request_matches_frozen_protocol():
    request = completion_request("hello")
    assert request["prompt"] == "hello"
    assert request["n_predict"] == 128
    assert request["temperature"] == 0
    assert request["top_k"] == 1
    assert request["top_p"] == 1
    assert request["seed"] == 42
    assert request["cache_prompt"] is False
    assert request["return_tokens"] is True
    assert request["stream"] is False


def test_capture_prompt_posts_completion_and_records_provenance():
    _Handler.bodies = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        receipt = capture_prompt(
            f"http://127.0.0.1:{server.server_port}",
            "hello",
            "pq2-off",
            _provenance(),
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert _Handler.bodies[0][0] == "/completion"
    assert _Handler.bodies[0][1]["return_tokens"] is True
    assert receipt["schema_version"] == 1
    assert receipt["label"] == "pq2-off"
    assert receipt["provenance"]["model"]["sha256"] == "a" * 64
    assert receipt["response"]["tokens"] == [7, 8] * 64


def test_read_prompts_rejects_blank_or_wrong_count():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prompts.txt"
        path.write_text("one\ntwo\n", encoding="utf-8")
        try:
            read_prompts(path)
        except ValueError as exc:
            assert "three" in str(exc)
        else:
            raise AssertionError("identity fixture must have exactly three prompts")


def test_short_completion_is_not_an_identity_pass():
    response = {
        "content": "ok",
        "tokens": [7, 8],
        "timings": {
            "cache_n": 0,
            "prompt_n": 3,
            "prompt_per_second": 12.0,
            "predicted_n": 2,
            "predicted_per_second": 4.0,
        },
    }
    with mock.patch.object(run_identity, "request_json", return_value=response):
        try:
            capture_prompt("http://127.0.0.1:8123", "hello", "pq2-off", _provenance())
        except ValueError as exc:
            assert "exactly 128 tokens" in str(exc)
        else:
            raise AssertionError("matching EOS-short samples must not pass identity")


def test_cli_fails_when_mtp_arm_diverges():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model, binary, dll = (root / name for name in ("model.gguf", "llama-server.exe", "ggml-hip.dll"))
        for path in (model, binary, dll):
            path.write_bytes(path.name.encode())
        prompts = root / "prompts.txt"
        prompts.write_text("one\ntwo\nthree\n", encoding="utf-8")
        spec = {
            "path": model,
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "port": 8123,
            "default_spec": "draft-mtp",
        }

        @contextmanager
        def fake_server(*_args, **_kwargs):
            yield {"pid": 99, "base_url": "http://127.0.0.1:8123",
                   "started_at_utc": "now", "health": {"status": "ok"}}

        def fake_arm(_url, texts, label, provenance, **_kwargs):
            token = 2 if label.endswith("-mtp") else 1
            return {"receipts": [
                {
                    "provenance": provenance,
                    "request": completion_request(text),
                    "response": {
                        "content": "same",
                        "tokens": [token] * 128,
                        "timings": {
                            "cache_n": 0, "prompt_n": 1, "prompt_per_second": 1.0,
                            "predicted_n": 128, "predicted_per_second": 1.0,
                        },
                    },
                }
                for text in texts
            ]}

        with (
            mock.patch.object(run_identity, "MODEL_SPECS", {"pq2-mtp": spec}),
            mock.patch.object(run_identity, "managed_server", fake_server),
            mock.patch.object(run_identity, "run_arm", side_effect=fake_arm),
            mock.patch.object(run_identity, "ensure_no_llama_processes", return_value=[]),
            mock.patch.object(run_identity, "capture_gpu_process_memory", return_value={"rows": []}),
            mock.patch.object(run_identity, "environment", return_value={}),
        ):
            code = run_identity.main([
                "--model", "pq2-mtp", "--binary", str(binary), "--hip-dll", str(dll),
                "--prompts", str(prompts), "--output-dir", str(root / "output"),
            ])
        summary = json.loads((root / "output" / "pq2-comparison.json").read_text(encoding="utf-8"))
        assert code != 0
        assert summary["identical"] is False


if __name__ == "__main__":
    test_completion_request_matches_frozen_protocol()
    test_capture_prompt_posts_completion_and_records_provenance()
    test_read_prompts_rejects_blank_or_wrong_count()
    print("ok")
