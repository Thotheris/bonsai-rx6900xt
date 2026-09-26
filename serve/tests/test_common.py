import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    capture_gpu_process_memory,
    ensure_no_llama_processes,
    file_provenance,
    managed_server,
    parse_gpu_process_memory,
    request_json,
    server_argv,
)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"path": self.path, "body": body}).encode("utf-8"))

    def log_message(self, *_args):
        pass


class _FakeProcess:
    def __init__(self):
        self.pid = 1234
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True
        return 0

    def kill(self):
        self.killed = True


def test_server_argv_has_explicit_mtp_mode():
    base = server_argv(Path("model.gguf"), 32768, 8080, spec="none", binary=Path("server.exe"))
    mtp = server_argv(Path("model.gguf"), 32768, 8080, spec="draft-mtp", binary=Path("server.exe"))
    assert "--spec-type" not in base
    assert mtp[:-4] == base
    assert mtp[-4:] == ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"]


def test_file_provenance_verifies_expected_sha256():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "artifact.bin"
        path.write_bytes(b"abc")
        result = file_provenance(path, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        assert result["size_bytes"] == 3
        assert result["sha256"].startswith("ba7816bf")
        try:
            file_provenance(path, "0" * 64)
        except ValueError as exc:
            assert "SHA-256" in str(exc)
        else:
            raise AssertionError("wrong hashes must fail")


def test_request_json_round_trips_against_mock_http():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        assert request_json(base_url, "/health") == {"status": "ok"}
        result = request_json(base_url, "/completion", {"n_predict": 128})
        assert result == {"path": "/completion", "body": {"n_predict": 128}}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_managed_server_waits_for_health_and_stops_process():
    fake = _FakeProcess()
    calls = []

    def popen_factory(argv, **kwargs):
        calls.append((argv, kwargs))
        return fake

    def health_waiter(base_url, process, timeout):
        assert base_url == "http://127.0.0.1:8123"
        assert process is fake
        assert timeout == 7
        return {"status": "ok"}

    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "server.log"
        with managed_server(
            ["server.exe", "--port", "8123"],
            {},
            8123,
            log,
            startup_timeout=7,
            popen_factory=popen_factory,
            health_waiter=health_waiter,
        ) as running:
            assert running["pid"] == 1234
            assert running["health"] == {"status": "ok"}
        assert log.exists()

    assert calls[0][0] == ["server.exe", "--port", "8123"]
    assert fake.terminated is True
    assert fake.waited is True
    assert fake.killed is False


def test_existing_llama_process_is_a_gate_not_killed():
    class Completed:
        returncode = 0
        stdout = json.dumps({"name": "llama-server", "pid": 77, "path": "server.exe"})
        stderr = ""

    def runner(_argv, **_kwargs):
        return Completed()

    try:
        ensure_no_llama_processes(runner=runner)
    except RuntimeError as exc:
        assert "77" in str(exc)
        assert "llama-server" in str(exc)
    else:
        raise AssertionError("an existing llama process must block a new harness server")


def test_gpu_occupancy_records_large_desktop_process_without_blocking():
    raw = json.dumps([
        {
            "metric": "dedicated",
            "instance": "pid_42_luid_0_phys_0",
            "pid": 42,
            "process_name": "browser",
            "cooked_value": 2 * 1024**3,
        },
        {
            "metric": "shared",
            "instance": "pid_42_luid_0_phys_0",
            "pid": 42,
            "process_name": "browser",
            "cooked_value": 256,
        },
    ])
    parsed = parse_gpu_process_memory(raw)
    assert parsed[0]["cooked_value"] == float(2 * 1024**3)
    assert parsed[0]["pid"] == 42

    class Completed:
        returncode = 0
        stdout = raw
        stderr = ""

    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return Completed()

    snapshot = capture_gpu_process_memory(runner=runner)
    assert snapshot["classification"] == "desktop-active"
    assert snapshot["rows"] == parsed
    assert calls[0][0][0] == "powershell.exe"


def test_gpu_occupancy_rejects_nonfinite_or_unattributed_rows():
    bad_rows = [
        '[{"metric":"shared","instance":"pid_1","pid":1,"process_name":"x","cooked_value":NaN}]',
        '[{"metric":"shared","instance":"pid_1","process_name":"x","cooked_value":1}]',
    ]
    for raw in bad_rows:
        try:
            parse_gpu_process_memory(raw)
        except ValueError as exc:
            assert "GPU" in str(exc) or "JSON" in str(exc)
        else:
            raise AssertionError("invalid GPU occupancy row was accepted")


if __name__ == "__main__":
    test_server_argv_has_explicit_mtp_mode()
    test_file_provenance_verifies_expected_sha256()
    test_request_json_round_trips_against_mock_http()
    test_managed_server_waits_for_health_and_stops_process()
    print("ok")
