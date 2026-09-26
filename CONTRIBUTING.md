# add your card

One row per card, same method, so rows compare. Your row is not comparable to another card's
row if the protocol differs, so say which protocol you ran.

1. Build the pinned source with `kernel/build-staging.cmd` and serve with the
   matching script in `serve/`. Record the runtime commit, applied patch hash,
   model SHA-256, and actual `ggml-hip.dll` SHA-256. PTQ1_0 and PQ2_0 are
   different GGUF formats of the same Bonsai 2 release. Do not submit results
   from the older `Ternary-Bonsai-27B-PQ2_0.gguf` as Bonsai 2 measurements.
2. Run the fill curve within a measured context (32768 is the starter value;
   204800 was explored on an earlier build, without verified residency):

   ```text
   python sweeps/run_fill.py --model <ptq1|pq2|ptq1-mtp|pq2-mtp> --context 32768 \
     --mtp <off|on> --exploratory --depths 0,16384 --reps 3 \
     --output <your-private-results>/<arm>.json
   ```

   `--model` takes the serve spec name, not a GGUF path (`serve/common.py` `MODEL_SPECS`).
   Use `--mtp off` for ungrafted `ptq1` and `pq2`; only grafted models have an MTP-on arm.
   Every selected depth must fit `depth + 512 + 128 <= context` and must produce
   valid samples, or the command exits non-zero. `--exploratory` makes
   `claim: residency-unverified` and `resident_headline: false` explicit;
   without it `run_fill.py` demands a `--fit-receipt` whose `decision` is `fit`.
   Do not invent a fit receipt.
3. Note the served VRAM at your context. This repo records per-PID dedicated/shared counters
   before and after each run rather than a single peak, because a per-process dedicated counter
   is not device total.
4. Open a PR that adds `sweeps/<your card>.md` with: card, VRAM, driver, binary and
   `ggml-hip.dll` sha256, the fill table verbatim from your receipts, served dedicated bytes, and
   one sentence on anything odd. Numbers only from runs you did yourself.

Rules that are not negotiable:

- A headline with wrong `cache_n`/`prompt_n`/`predicted_n`, an unhealthy server, an omitted
  depth, or fewer than three valid reps is `invalid` or `unknown`, never a zero-speed result.
- Publish raw min/max and the sample count. Three samples are not a confidence interval.
- Never trim or replace an inconvenient rep. Keep the receipt and say the median rule was used.
- Absolute tok/s does not transfer between sessions. Only same-session paired deltas do; if you
  compare a candidate to a baseline, interleave the passes and then reverse the order.
- `desktop-active` measurements are permitted and must be labelled as such. Do not kill
  unrelated applications and do not claim an idle GPU.

Never publish local GGUFs, DLLs, raw machine receipts, absolute paths, credentials,
or identifying GPU-process lists with a sweep PR. Share a sanitized summary and
reproduction instructions; retain hash-bound raw receipts privately.