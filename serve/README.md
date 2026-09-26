# RX 6900 XT source-built serve lines

Build the pinned PrismML fork plus the HIP PTQ1_0 patch with
`kernel/build-staging.cmd` before using these scripts. The build writes
`_work/build-gfx1030/bin/llama-server.exe`; no binary is distributed here.
Set `BONSAI_ROCM_VENV` to your TheRock-enabled Python environment and,
if needed, `BONSAI_MODEL_DIR` to your downloaded GGUF directory. Device
`ROCm0` is selected after `HIP_VISIBLE_DEVICES=1` (override that physical
selection with `BONSAI_HIP_DEVICE` if your adapter ordering differs).

The default context is 32768, not an asserted resident limit; the prior
262144 fit probe was inconclusive. Cache uses q4_0 K/V with one slot.
Each launcher checks the exact model SHA-256 before starting:

| command | current Bonsai 2 model | mode |
| --- | --- | --- |
| `python serve/16gb-ptq1.py` | `PTQ1_0.gguf` | MTP off |
| `python serve/16gb-pq2.py` | `PQ2_0.gguf` | MTP off |
| `python serve/16gb-ptq1-mtp.py` | `PTQ1_0-mtp.gguf` | MTP on |
| `python serve/16gb-pq2-mtp.py` | `PQ2_0-mtp.gguf` | MTP on; verify identity and speed before use |

The `-mtp` files are locally grafted artifacts, not part of the official
model download. Pass an explicit context as the first positional argument,
for example `python serve/16gb-pq2.py 65536`. Do not infer verified VRAM
residency or speculative speedup from a server starting successfully.

For PQ2 MTP, merge the **current** Bonsai 2 PQ2_0 GGUF with an extracted
Qwen3.8-27B MTP head using the linked upstream `graft/tools/merge.py`.
This machine's head was 812,322,560 bytes (SHA-256
`021e81b06a079e7fe67e29036b0e9b972f030f7c7a1728c9aa99044298d3d0c8`);
its Bonsai 2 PQ2 graft has SHA-256
`e949d4277f7a6c1c1b5906b87689d3165e57ff7d13a63d42ea2c8d88d7d36c7e`.
Stripping that graft reproduced the downloaded PQ2 release byte-for-byte
(SHA-256 `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`).
For a different donor head, explicitly set `BONSAI_PQ2_MTP_SHA256` to
the hash of your newly generated GGUF; likewise set
`BONSAI_PTQ1_MTP_SHA256` for a different PTQ1 graft. A hash override
changes the expected artifact, not the requirement to measure MTP identity.

```text
python <bonsai2-small-gpu>/graft/tools/merge.py <model-dir>/Ternary-Bonsai-2-27B-PQ2_0.gguf <model-dir>/qwen38-27b-nextn-head.gguf <model-dir>/Ternary-Bonsai-2-27B-PQ2_0-mtp.gguf
python graft/run_identity.py --model pq2-mtp
```

Identity requires three complete 128-token greedy prompts with identical
token IDs and UTF-8 output for MTP off/on. A short completion or a mismatch
fails the command; the receipts remain local under ignored `results/`.
