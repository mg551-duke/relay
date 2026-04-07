# Native Rust port notes for damping and BPGD

The standalone experiment layer is ready to run as-is, but if you later decide to move `damped_BP` and `BPGD` into the native Rust decoder stack, the relevant repo locations are:

- `crates/relay_bp/src/bp/min_sum.rs`
- `crates/relay_bp_py/src/decoder.rs`
- `src/relay_bp/stim/sinter/decoders.py`

## Damped BP

### Current state

The native min-sum decoder already exposes:

- `alpha`
- `alpha_iteration_scaling_factor`
- `gamma0`

but not a check-to-variable damping coefficient.

### Minimal native change

1. Extend the Rust min-sum config struct with a new optional field, for example:

```rust
pub c_damp: Option<f64>
```

2. Add a second edge-message buffer to the decoder state, storing the previous check-to-variable messages.

3. Right after the fresh check-to-variable update is computed, replace it with:

```rust
msg_new = c_damp * msg_new + (1.0 - c_damp) * msg_old;
```

4. Copy the damped message into the stored old-message buffer.

5. Expose the new constructor keyword through the Python bindings.

6. Add a new Python-side alias and sinter wrapper name:

- `damped_BP`

## Native BPGD

A native BPGD port is most straightforward if implemented as a wrapper decoder that repeatedly:

1. runs the existing min-sum decoder for `T` iterations,
2. checks convergence,
3. fixes one variable,
4. reduces the problem,
5. repeats until `R` rounds are exhausted,
6. then optionally hands the reduced problem to another native decoder.

The key bookkeeping fields to expose back to Python are:

- total iterations
- stage-by-stage iterations
- stage-by-stage convergence flags
- number of fallback usages
- final fixed-variable set if a later decoder needs it

That is why the standalone overlay already uses the reduced-problem convention. The same convention can be ported into the Rust layer without changing the experiment naming or CSV layout.
