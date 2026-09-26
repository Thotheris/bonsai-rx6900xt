# Bonsai 2 on RX 6900 XT (staging)

Source-built Windows HIP recipe for **both Bonsai 2 27B packings**, PTQ1_0 and PQ2_0, on an AMD Radeon RX 6900 XT (gfx1030). Inspired by [bonsai2-small-gpu](https://github.com/sudoingX/bonsai2-small-gpu); its NVIDIA CUDA kernel patch is **not** the AMD implementation. The project does not redistribute GGUFs, runtime binaries, or machine-specific benchmark receipts.

## Build from source

Requirements: Windows x64, RX 6900 XT, Git, CMake, Ninja, an x64 Visual Studio Developer Command Prompt, and a Python environment containing TheRock's `_rocm_sdk_devel` package. The fork is pinned to [PrismML llama.cpp commit `23d0d71502d690230131f22b1393ac41d6b4406e`](https://github.com/PrismML-Eng/llama.cpp/commit/23d0d71502d690230131f22b1393ac41d6b4406e). `kernel/build-staging.cmd` clones that revision into ignored `_work/llama.cpp`, applies `kernel/patches/0001-rdna2-ptq1-hip-mmvq.patch`, and builds `llama-server`, `llama-bench`, and `test-backend-ops` under ignored `_work/build-gfx1030/`. The patch contains the validated native HIP ternary dot path, two-row PTQ1 tile with bounded final-row reads, and PTQ1/F16 support; it does not modify PQ2 decode.

In the x64 Developer Command Prompt:

```bat
set BONSAI_ROCM_VENV=<path-to-your-TheRock-python-environment>
kernel\build-staging.cmd
```

No frozen or precompiled binary is pinned as a runtime dependency. The source revision **is** pinned so that the patch fails rather than silently applying to an incompatible fork. Other GPUs and Linux are not tested.

## Download the current Bonsai 2 release

```text
hf download prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-PTQ1_0.gguf --local-dir <model-dir>
hf download prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-PQ2_0.gguf --local-dir <model-dir>
```

Set `BONSAI_MODEL_DIR=<model-dir>` (default: a `models` directory alongside this repository). The official [model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) lists both packings and the Apache-2.0 model license. The release files are checked by each launcher before starting:

| packing | bytes | SHA-256 |
| --- | ---: | --- |
| Bonsai 2 PTQ1_0 | 5,946,648,928 | `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` |
| Bonsai 2 PQ2_0 | 7,206,168,928 | `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1` |

**Different model:** older `Ternary-Bonsai-27B-PQ2_0.gguf` measurements are not Bonsai 2 PQ2_0 results. The older file was 7,165,121,600 bytes with SHA-256 `e4781999f1997ef97ce0c58d05750835acc999d18d83ee6489ba7ac7b14cb5f6`.

## Serve

```text
python serve/16gb-ptq1.py
python serve/16gb-pq2.py
```

Run **one server at a time**. The default context is 32768, q4_0 K/V, one slot, `ROCm0` after `HIP_VISIBLE_DEVICES=1`; set `BONSAI_HIP_DEVICE` if physical adapter numbering differs. Pass a context integer to override the default. The source-built server has no verified resident 262144-token configuration. See `serve/README.md` for optional grafted MTP launchers and the no-speedup-until-measured rule.

For a grafted PTQ1 MTP file, follow the [upstream head-extraction and merge recipe](https://github.com/sudoingX/bonsai2-small-gpu/blob/main/graft/recipe.txt) and use the Bonsai 2 PTQ1_0 base. MTP on/off greedy identity and throughput are separate gates: a valid server start proves neither.

## Evidence and limits

On the measured RX 6900 XT, a *locally built* two-row PTQ1 HIP candidate passed 78/78 backend operations and 3×128-token greedy identity against the historical frozen control with MTP both on and off. Six reversed-order depth-32768 MTP-on server pairs measured a median **+26.932%** versus that control with identical 128-token IDs and UTF-8 output in every pair. Six depth-32768 MTP-off `llama-bench` pairs measured a median **+22.231%**. These results belong to the **specific prior candidate DLL** (SHA-256 `b077258e2656a50a0000ab61fe32b4c192033e8159e9f412242466aa5b59266a`), not to every rebuild of the source patch. Desktop residency was unverified and the raw receipts remain local pending privacy review.

The **fresh source-built** DLL from this recipe (SHA-256
`dde7a42df68cdbeeb23654d4cc02e9fb2e45e56b8634904c98aebc0b768cb648`)
passed 78/78 PTQ1 backend cases (zero unsupported) and 49/49 supported
PQ2 backend cases (33 F16-input cases were reported unsupported by both
ROCm0 and CPU; zero failures). A real server loaded the current Bonsai 2
PQ2 GGUF and returned an eight-token completion. On the grafted files,
three 128-token greedy prompts were identical with MTP off and on for
each format. The fresh PTQ1 run also matched the frozen control's token
IDs and UTF-8 output in all three prompts, both with MTP off and on.

The original PQ2 depth curve and its long-context MTP slowdown used the *older* model named above. A single exploratory depth-32768 sample on the new graft measured 34.48 tok/s MTP-off and 32.69 tok/s MTP-on; one repetition per arm fails the three-repetition headline gate, and residency was unverified. Do not claim a Bonsai 2 PQ2 MTP speedup, production throughput, or 262144-token fit from these observations. Contributors can reproduce their own [hash-bound measurements](CONTRIBUTING.md), but should not compare unmatched desktop sessions or publish identifying process lists.

## Attribution

This repository's own code is Apache-2.0 (`LICENSE`). The pinned PrismML/ggml runtime is upstream MIT-licensed; its source is fetched rather than vendored. The Bonsai 2 model files stay with PrismML. The graft workflow is credited to `sudoingX/bonsai2-small-gpu` and is linked, not copied; its README describes Apache-2.0 licensing, but no root LICENSE file was found when this staging recipe was prepared. Confirm any third-party redistribution terms before vendoring their tools.
