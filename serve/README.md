# serve lines

All lines use the frozen HIP binary in `build-hip-original` until `kernel/LEDGER.md` says a candidate was kept.
Cache is q4_0. Context is the value in `sweeps/rx6900xt.md` under `window_tokens`, not a guess.
One slot. Device ROCm0 after `HIP_VISIBLE_DEVICES=1`.

| script | file | spec |
| --- | --- | --- |
| `16gb-pq2.py` | Ternary-Bonsai-27B-PQ2_0.gguf | none |
| `16gb-pq2-mtp.py` | Ternary-Bonsai-27B-PQ2_0-mtp.gguf | `--spec-type draft-mtp --spec-draft-n-max 1` |
| `16gb-ptq1-mtp.py` | Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf | `--spec-type draft-mtp --spec-draft-n-max 1` |
