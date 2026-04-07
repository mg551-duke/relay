# Native BPGD + Sinter progress patch

This patch set keeps the repository's original architecture:

- the heavy message-passing kernels stay in Rust,
- the decoders are exposed through the existing Python bindings,
- and Stim/Sinter integration stays in `src/relay_bp/stim/sinter/decoders.py`.

## What is added

### New Sinter-exposed decoder names

The following names are now registered alongside the existing aliases:

- `regular_BP`
- `damped_BP`
- `mem_BP`
- `relay_BP`
- `BPGD`
- `BPGD-relay_BP`

The original names remain available too:

- `msl-bp`
- `mem-bp`
- `relay-bp`

### Native statistics

`DecodeResult.extra` now carries native stage information for BPGD:

- `fallback_used`
- `converged_stage_index`
- `stage_names`
- `stage_iterations`
- `stage_converged`

Those values are exposed through the Python `DecodeResult` getters and can be
persisted by the Sinter wrapper to JSONL sidecar files.

### Damped BP

`MinSumDecoderConfig` now has an optional `c_damp`. The min-sum update blends
new check-to-variable messages with the previous iteration's messages using the
same convention discussed in the project chat:

```text
new_msg = c_damp * new_msg + (1 - c_damp) * old_msg
```

### Native BPGD controller

`crates/relay_bp/src/bp/bpgd.rs` implements a native wrapper around the existing
Rust min-sum decoder. The current implementation takes a deliberately surgical
approach:

- the Tanner graph stays fixed,
- decimated variables are clamped by overwriting their log-priors with
  `±infinity`,
- and optional Relay fallback is also kept native.

This is much closer to the original repository architecture than the earlier
Python-side benchmark loop, while still being small enough to debug iteratively.

## Files of interest

- `crates/relay_bp/src/bp/min_sum.rs`
- `crates/relay_bp/src/bp/bpgd.rs`
- `crates/relay_bp_py/src/bp/bpgd.rs`
- `crates/relay_bp_py/src/decoder.rs`
- `src/relay_bp/stim/sinter/decoders.py`
- `scripts/run_gross_144_12_12_memory_X_sinter.py`
- `configs/decoder_specs_gross_144_12_12_memory_X_sinter.json`
- `sbatch/run_gross_144_12_12_memory_X_sinter_array.sbatch`

## Running the Gross [[144,12,12]] memory-X experiment

Use the new Sinter driver:

```bash
python scripts/run_gross_144_12_12_memory_X_sinter.py \
  --circuits-dir tests/testdata/bicycle_bivariate \
  --decoder-config configs/decoder_specs_gross_144_12_12_memory_X_sinter.json \
  --outdir results_gross_144_12_12_memory_X_sinter \
  --filename-contains "circuit=bicycle_bivariate_144_12_12_memory_X" \
  --max-shots 10000 \
  --print-progress
```

Outputs:

- `*_resume.csv`: incremental Sinter save/resume file
- `*_summary.csv`: consolidated Sinter stats
- `*_flat_summary.csv`: easier-to-read flat CSV
- `details/*.jsonl`: per-shot detailed decode records carrying the native BPGD
  stage/fallback statistics

## Important note

This patch is a strong native-first step, but it is still a work in progress.
The current BPGD wrapper clamps variables by changing priors instead of
physically reducing the graph. That keeps the patch surgical and should still
be far more efficient than a pure-Python controller, but it is not yet the most
aggressive graph-reduction implementation one could build.
