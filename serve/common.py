"""Shared launch, HTTP, hashing, and lifecycle helpers for Bonsai runs."""

import contextlib
import hashlib
import json
import math
import os
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = Path(os.environ.get("BONSAI_WORKSPACE", ROOT.parent))
BINARY = ROOT / "_work" / "build-gfx1030" / "bin" / "llama-server.exe"
ACTIVE_DLL = BINARY.with_name("ggml-hip.dll")
MODEL_DIR = Path(os.environ.get("BONSAI_MODEL_DIR", LAB / "models"))
PQ2 = MODEL_DIR / "Ternary-Bonsai-2-27B-PQ2_0.gguf"
PTQ1 = MODEL_DIR / "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
PQ2_MTP = MODEL_DIR / "Ternary-Bonsai-2-27B-PQ2_0-mtp.gguf"
PTQ1_MTP = MODEL_DIR / "Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf"
DEFAULT_CONTEXT = 32768
WINDOW = 262144

MODEL_SPECS = {
    "ptq1": {
        "path": PTQ1,
        "sha256": "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3",
        "port": 8084,
        "default_spec": "none",
    },
    "pq2": {
        "path": PQ2,
        "sha256": "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1",
        "port": 8081,
        "default_spec": "none",
    },
    "pq2-mtp": {
        "path": PQ2_MTP,
        "sha256": os.environ.get("BONSAI_PQ2_MTP_SHA256", "e949d4277f7a6c1c1b5906b87689d3165e57ff7d13a63d42ea2c8d88d7d36c7e"),
        "port": 8082,
        "default_spec": "draft-mtp",
    },
    "ptq1-mtp": {
        "path": PTQ1_MTP,
        "sha256": os.environ.get("BONSAI_PTQ1_MTP_SHA256", "6b806ab2379ea4431b5d7657fc5d678d382c771bcd6b27e17e2ff30b990ea84a"),
        "port": 8083,
        "default_spec": "draft-mtp",
    },
}


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path, expected_sha256=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if expected_sha256 is not None:
        expected = str(expected_sha256).lower()
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError("expected SHA-256 must be 64 lowercase hexadecimal characters")
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": actual}


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_process_rows(text):
    if not text or not text.strip():
        return []
    try:
        value = json.loads(text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("process-list output is not strict JSON") from exc
    rows = value if isinstance(value, list) else [value]
    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("process-list row must be an object")
        name = row.get("name")
        pid = row.get("pid")
        path = row.get("path")
        if not isinstance(name, str) or not name:
            raise ValueError("process-list row requires a name")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("process-list row requires a positive pid")
        if path is not None and not isinstance(path, str):
            raise ValueError("process-list path must be a string or null")
        parsed.append({"name": name, "pid": pid, "path": path})
    return parsed


def ensure_no_llama_processes(*, runner=subprocess.run):
    """Fail closed if another llama server/bench exists; never kill it."""
    script = (
        "@(Get-Process llama-server,llama-bench -ErrorAction SilentlyContinue | "
        "ForEach-Object {[pscustomobject]@{name=$_.ProcessName;pid=[int]$_.Id;path=$_.Path}}) | "
        "ConvertTo-Json -Compress"
    )
    completed = runner(
        ["powershell.exe", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"llama process preflight failed: {completed.stderr.strip()}")
    rows = parse_process_rows(completed.stdout)
    if rows:
        detail = ", ".join(f"{row['name']} pid={row['pid']}" for row in rows)
        raise RuntimeError(f"existing llama process blocks the run: {detail}")
    return rows


def parse_gpu_process_memory(text):
    """Parse strict PID-attributed Windows GPU process-memory rows."""
    if not text or not text.strip():
        return []
    try:
        value = json.loads(text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("GPU process-memory output is not strict JSON") from exc
    rows = value if isinstance(value, list) else [value]
    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("GPU process-memory row must be an object")
        metric = row.get("metric")
        instance = row.get("instance")
        pid = row.get("pid")
        process_name = row.get("process_name")
        cooked = row.get("cooked_value")
        if metric not in {"dedicated", "shared"}:
            raise ValueError("GPU process-memory metric must be dedicated or shared")
        if not isinstance(instance, str) or not instance:
            raise ValueError("GPU process-memory instance must be non-empty")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("GPU process-memory row requires a positive pid")
        if not isinstance(process_name, str):
            raise ValueError("GPU process-memory row requires process_name")
        if isinstance(cooked, bool) or not isinstance(cooked, (int, float)):
            raise ValueError("GPU process-memory cooked_value must be numeric")
        cooked = float(cooked)
        if not math.isfinite(cooked) or cooked < 0:
            raise ValueError("GPU process-memory cooked_value must be finite and non-negative")
        parsed.append({
            "metric": metric,
            "instance": instance,
            "pid": pid,
            "process_name": process_name,
            "cooked_value": cooked,
        })
    return parsed


def capture_gpu_process_memory(*, runner=subprocess.run, attempts=4):
    """Capture all desktop GPU users without killing or threshold-gating them."""
    script = (
        "$rows=@(); "
        "foreach($item in @(@{Metric='dedicated';Name='Dedicated Usage'},"
        "@{Metric='shared';Name='Shared Usage'})){"
        "$samples=(Get-Counter (\"\\GPU Process Memory(*)\\\"+$item.Name) -ErrorAction Stop).CounterSamples;"
        "$samples | Where-Object {$_.CookedValue -gt 0} | ForEach-Object {"
        "$m=[regex]::Match($_.InstanceName,'(?:^|_)pid_(\\d+)(?:_|$)');"
        "if($m.Success){$processId=[int]$m.Groups[1].Value;"
        "$proc=Get-Process -Id $processId -ErrorAction SilentlyContinue;"
        "$rows += [pscustomobject]@{metric=$item.Metric;instance=$_.InstanceName;pid=$processId;"
        "process_name=if($proc){$proc.ProcessName}else{''};cooked_value=[double]$_.CookedValue}}}};"
        "$rows | ConvertTo-Json -Compress"
    )
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    last_error = "GPU process-memory command failed"
    for attempt in range(attempts):
        completed = runner(
            ["powershell.exe", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            return {
                "recorded_at_utc": utc_now(),
                "classification": "desktop-active",
                "rows": parse_gpu_process_memory(completed.stdout),
            }
        last_error = completed.stderr.strip() or last_error
        if attempt + 1 < attempts:
            time.sleep(0.5)
    raise RuntimeError(f"GPU process-memory command failed: {last_error}")


def environment(base=None):
    env = dict(os.environ if base is None else base)
    venv_value = env.get("BONSAI_ROCM_VENV")
    if not venv_value:
        raise SystemExit("Set BONSAI_ROCM_VENV to the Windows Python environment containing _rocm_sdk_devel.")
    venv = Path(venv_value)
    sdk = venv / "Lib" / "site-packages" / "_rocm_sdk_devel"
    if not sdk.is_dir():
        raise SystemExit(f"TheRock SDK missing: {sdk}")
    env["PATH"] = os.pathsep.join([
        str(venv / "Scripts"),
        str(sdk / "bin"),
        str(sdk / "lib" / "llvm" / "bin"),
        env.get("PATH", ""),
    ])
    env["HIP_VISIBLE_DEVICES"] = env.get("BONSAI_HIP_DEVICE", "1")
    env.pop("GGML_VK_VISIBLE_DEVICES", None)
    return env


def server_argv(model: Path, context: int, port: int, *, spec="none", binary=BINARY):
    if isinstance(context, bool) or int(context) <= 0:
        raise ValueError("context must be a positive integer")
    if isinstance(port, bool) or not 1 <= int(port) <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if spec not in {"none", "draft-mtp"}:
        raise ValueError("spec must be 'none' or 'draft-mtp'")
    argv = [
        str(binary), "-m", str(model), "--alias", "bonsai-27b",
        "--host", "127.0.0.1", "--port", str(int(port)),
        "-c", str(int(context)), "-np", "1", "-ngl", "999",
        "--device", "ROCm0", "--split-mode", "none",
        "--fit", "off", "--no-context-shift",
        "-fa", "on", "-b", "2048", "-ub", "512", "-t", "8",
        "-ctk", "q4_0", "-ctv", "q4_0",
        "--temp", "0", "--top-k", "1", "--top-p", "1", "--seed", "42",
    ]
    if spec == "draft-mtp":
        argv.extend(["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"])
    return argv


def _reject_json_constant(value):
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


def request_json(base_url, route, payload=None, *, timeout=120.0):
    url = base_url.rstrip("/") + "/" + route.lstrip("/")
    data = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON from {url}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON from {url} must be an object")
    return value


def wait_for_health(base_url, process, timeout=180.0, interval=0.25):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"llama-server exited before health check (exit {code})")
        try:
            health = request_json(base_url, "/health", timeout=min(2.0, interval + 1.0))
            if health.get("status") == "ok":
                return health
            last_error = RuntimeError(f"unexpected health response: {health!r}")
        except (OSError, RuntimeError, ValueError) as exc:
            last_error = exc
        time.sleep(interval)
    raise TimeoutError(f"server was not healthy within {timeout:g}s: {last_error}")


def stop_process(process, timeout=15.0):
    if process.poll() is not None:
        return process.poll()
    process.terminate()
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=timeout)


@contextlib.contextmanager
def managed_server(
    argv,
    env,
    port,
    log_path,
    *,
    startup_timeout=180.0,
    popen_factory=subprocess.Popen,
    health_waiter=wait_for_health,
):
    """Start one server, wait for health, and always stop it."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{int(port)}"
    with log_path.open("ab") as log_handle:
        process = popen_factory(
            list(argv),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        started_at = utc_now()
        try:
            health = health_waiter(base_url, process, startup_timeout)
            yield {
                "pid": process.pid,
                "base_url": base_url,
                "argv": list(argv),
                "started_at_utc": started_at,
                "health": health,
                "process": process,
                "log_path": str(log_path.resolve()),
            }
        finally:
            stop_process(process)
