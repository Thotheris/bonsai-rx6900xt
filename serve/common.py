import os
from pathlib import Path

LAB = Path(r"D:/projects/Bonsai")
BINARY = LAB / "llama.cpp" / "build-hip-original" / "bin" / "llama-server.exe"
FROZEN_DLL = LAB / "llama.cpp" / "build-hip-original" / "bin" / "ggml-hip.dll"
FROZEN_SHA256 = "7920de6a4b6f843da72f9f258c8e8e58aa006a99874fb267f796fb17e37bb6fc"
PQ2 = LAB / "models" / "Ternary-Bonsai-27B-PQ2_0.gguf"
PQ2_MTP = LAB / "models" / "Ternary-Bonsai-27B-PQ2_0-mtp.gguf"
PTQ1_MTP = LAB / "models" / "Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf"
WINDOW = 262144


def environment():
    env = os.environ.copy()
    venv = Path(env.get("BONSAI_ROCM_VENV", r"D:/llama.cpp/.venv"))
    sdk = venv / "Lib" / "site-packages" / "_rocm_sdk_devel"
    if not sdk.is_dir():
        raise SystemExit(f"TheRock SDK missing: {sdk}")
    env["PATH"] = os.pathsep.join([
        str(venv / "Scripts"), str(sdk / "bin"), str(sdk / "lib" / "llvm" / "bin"), env.get("PATH", ""),
    ])
    env["HIP_VISIBLE_DEVICES"] = env.get("BONSAI_HIP_DEVICE", "1")
    env.pop("GGML_VK_VISIBLE_DEVICES", None)
    return env


def server_argv(model: Path, context: int, port: int):
    return [
        str(BINARY), "-m", str(model), "--alias", "bonsai-27b",
        "--host", "127.0.0.1", "--port", str(port),
        "-c", str(context), "-np", "1", "-ngl", "999",
        "--device", "ROCm0", "--split-mode", "none",
        "-fa", "on", "-b", "2048", "-ub", "512", "-t", "8",
        "-ctk", "q4_0", "-ctv", "q4_0",
        "--temp", "0", "--top-k", "1", "--top-p", "1", "--seed", "42",
    ]
