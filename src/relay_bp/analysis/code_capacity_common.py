from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csc_matrix

import relay_bp


@dataclass(frozen=True)
class CodeCapacityProblem:
    path: Path
    hx: np.ndarray
    hz: csc_matrix
    lz: np.ndarray
    error_priors: np.ndarray
    metadata: dict[str, Any]

    @property
    def n_bits(self) -> int:
        return int(self.hz.shape[1])

    @property
    def n_logicals(self) -> int:
        return int(self.lz.shape[0])


@dataclass(frozen=True)
class DecodeEvaluation:
    converged: bool
    iterations: int
    logical_signature: np.ndarray
    logical_weight: int
    logical_success: bool
    exact_recovery: bool
    decoding: np.ndarray
    residual: np.ndarray


def default_export_code_paths(repo_root: Path) -> dict[str, Path]:
    export_dir = repo_root / "examples" / "notebook_data" / "stim_decoder_npz_exports"
    return {
        "gross": export_dir / "gross_from_stim_Z_basis_fault_projection.npz",
        "two_gross": export_dir / "two_gross_from_stim_Z_basis_fault_projection.npz",
        "surface13": export_dir / "surface13_from_stim_Z_basis_fault_projection.npz",
    }


def load_code_capacity_problem(
    path: Path | str,
    *,
    default_prior_error_rate: float = 0.1,
) -> CodeCapacityProblem:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    hx = np.asarray(data["Hx"], dtype=np.uint8) % 2
    hz_dense = np.asarray(data["Hz"], dtype=np.uint8) % 2
    lz = np.asarray(data["Lz"], dtype=np.uint8) % 2
    error_priors = np.asarray(
        data["error_priors"]
        if "error_priors" in data.files
        else np.full(hz_dense.shape[1], float(default_prior_error_rate), dtype=np.float64),
        dtype=np.float64,
    ).reshape(-1)
    if error_priors.size != hz_dense.shape[1]:
        raise ValueError(
            f"error_priors length {error_priors.size} does not match Hz columns {hz_dense.shape[1]}."
        )
    metadata: dict[str, Any] = {}
    if "metadata_json" in data.files:
        raw = np.asarray(data["metadata_json"]).item()
        metadata = json.loads(str(raw))
    return CodeCapacityProblem(
        path=path,
        hx=hx,
        hz=csc_matrix(hz_dense),
        lz=lz,
        error_priors=error_priors,
        metadata=metadata,
    )


def with_uniform_error_rate(
    problem: CodeCapacityProblem,
    error_rate: float,
) -> CodeCapacityProblem:
    """Return a shallow copy of the problem with uniform decoder priors."""
    return CodeCapacityProblem(
        path=problem.path,
        hx=problem.hx,
        hz=problem.hz,
        lz=problem.lz,
        error_priors=np.full(problem.n_bits, float(error_rate), dtype=np.float64),
        metadata={
            **problem.metadata,
            "uniform_error_rate": float(error_rate),
        },
    )


def _all_half_subsets(support_bits: tuple[int, ...]) -> list[tuple[int, ...]]:
    from itertools import combinations

    if len(support_bits) == 0 or len(support_bits) % 2 != 0:
        return []
    half_weight = len(support_bits) // 2
    return [tuple(sorted(subset)) for subset in combinations(support_bits, half_weight)]


def enumerate_half_stabilizer_cases(
    problem: CodeCapacityProblem,
    *,
    selected_row_weights: set[int] | None = None,
    max_cases: int | None = None,
    shuffle_seed: int | None = None,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    case_id = 0
    hz_dense = problem.hz.toarray().astype(np.uint8) % 2
    for row_idx in range(problem.hx.shape[0]):
        support_bits = tuple(np.flatnonzero(problem.hx[row_idx]).astype(int).tolist())
        row_weight = len(support_bits)
        if row_weight == 0 or row_weight % 2 != 0:
            continue
        if selected_row_weights is not None and row_weight not in selected_row_weights:
            continue
        for half_subset_idx, flipped_bits in enumerate(_all_half_subsets(support_bits)):
            error = np.zeros(problem.n_bits, dtype=np.uint8)
            error[list(flipped_bits)] = 1
            syndrome = np.asarray((hz_dense @ error) % 2, dtype=np.uint8).reshape(-1)
            cases.append(
                {
                    "case_id": int(case_id),
                    "row_idx": int(row_idx),
                    "row_weight": int(row_weight),
                    "half_subset_idx": int(half_subset_idx),
                    "support_bits": support_bits,
                    "flipped_bits": tuple(int(bit) for bit in flipped_bits),
                    "error": error,
                    "syndrome": syndrome,
                    "syndrome_weight": int(syndrome.sum()),
                }
            )
            case_id += 1
    if shuffle_seed is not None and cases:
        rng = np.random.default_rng(int(shuffle_seed))
        rng.shuffle(cases)
    if max_cases is not None:
        cases = cases[: int(max_cases)]
    return cases


def build_half_stabilizer_case_from_seed(
    problem: CodeCapacityProblem,
    core_seed: int,
) -> dict[str, Any]:
    row_weights = problem.hx.sum(axis=1).astype(int)
    eligible_rows = np.flatnonzero((row_weights > 0) & ((row_weights % 2) == 0))
    if eligible_rows.size == 0:
        raise ValueError("No even-weight Hx row exists for a half-stabilizer case.")
    rng = np.random.default_rng(int(core_seed))
    row_idx = int(rng.choice(eligible_rows))
    support_bits = tuple(np.flatnonzero(problem.hx[row_idx]).astype(int).tolist())
    flipped_bits = tuple(
        sorted(
            rng.choice(
                np.asarray(support_bits, dtype=int),
                size=len(support_bits) // 2,
                replace=False,
            ).astype(int).tolist()
        )
    )
    error = np.zeros(problem.n_bits, dtype=np.uint8)
    error[list(flipped_bits)] = 1
    syndrome = np.asarray((problem.hz.toarray() @ error) % 2, dtype=np.uint8).reshape(-1)
    return {
        "case_id": int(core_seed),
        "row_idx": row_idx,
        "row_weight": len(support_bits),
        "half_subset_idx": 0,
        "support_bits": support_bits,
        "flipped_bits": flipped_bits,
        "error": error,
        "syndrome": syndrome,
        "syndrome_weight": int(syndrome.sum()),
    }


def sample_random_error_cases(
    problem: CodeCapacityProblem,
    *,
    num_samples: int,
    error_rate: float,
    base_seed: int,
) -> list[dict[str, Any]]:
    hz_dense = problem.hz.toarray().astype(np.uint8) % 2
    cases = []
    for sample_idx in range(int(num_samples)):
        rng = np.random.default_rng(int(base_seed) + sample_idx)
        error = (rng.random(problem.n_bits) < float(error_rate)).astype(np.uint8)
        syndrome = np.asarray((hz_dense @ error) % 2, dtype=np.uint8).reshape(-1)
        cases.append(
            {
                "sample_idx": int(sample_idx),
                "error": error,
                "error_weight": int(error.sum()),
                "syndrome": syndrome,
                "syndrome_weight": int(syndrome.sum()),
            }
        )
    return cases


def make_min_sum_tracer(
    problem: CodeCapacityProblem,
    *,
    max_iter: int,
    alpha: float | None,
    gamma0: float | None,
) -> relay_bp.MinSumBPDecoderTraceF64:
    return relay_bp.MinSumBPDecoderTraceF64(
        problem.hz,
        error_priors=problem.error_priors,
        max_iter=int(max_iter),
        alpha=alpha,
        gamma0=gamma0,
    )


def evaluate_decode_result(
    *,
    result: Any,
    error: np.ndarray,
    lz: np.ndarray,
) -> DecodeEvaluation:
    decoding = np.asarray(result.decoding, dtype=np.uint8)
    residual = np.bitwise_xor(decoding, np.asarray(error, dtype=np.uint8))
    logical_signature = np.asarray((lz @ residual) % 2, dtype=np.uint8).reshape(-1)
    logical_weight = int(logical_signature.sum())
    return DecodeEvaluation(
        converged=bool(result.success),
        iterations=int(result.iterations),
        logical_signature=logical_signature,
        logical_weight=logical_weight,
        logical_success=logical_weight == 0,
        exact_recovery=bool(np.array_equal(decoding, error)),
        decoding=decoding,
        residual=residual,
    )


def _finish_trace_decode(tracer: relay_bp.MinSumBPDecoderTraceF64, syndrome: np.ndarray) -> Any:
    result = tracer.snapshot(np.asarray(syndrome, dtype=np.uint8))
    while tracer.current_iteration < result.max_iter and not result.success:
        result = tracer.run_iteration(np.asarray(syndrome, dtype=np.uint8))
    return result


def decode_with_plain_bp(
    tracer: relay_bp.MinSumBPDecoderTraceF64,
    syndrome: np.ndarray,
    *,
    n_bits: int,
) -> Any:
    return decode_with_memory_and_edge_damping(
        tracer,
        syndrome,
        n_bits=n_bits,
        memory_strengths=None,
        edge_damping_messages=None,
    )


def decode_with_memory_strengths(
    tracer: relay_bp.MinSumBPDecoderTraceF64,
    syndrome: np.ndarray,
    memory_strengths: np.ndarray,
) -> Any:
    return decode_with_memory_and_edge_damping(
        tracer,
        syndrome,
        n_bits=int(np.asarray(memory_strengths, dtype=np.float64).shape[0]),
        memory_strengths=memory_strengths,
        edge_damping_messages=None,
    )


def decode_with_edge_damping(
    tracer: relay_bp.MinSumBPDecoderTraceF64,
    syndrome: np.ndarray,
    edge_damping_messages: np.ndarray,
    *,
    n_bits: int,
) -> Any:
    return decode_with_memory_and_edge_damping(
        tracer,
        syndrome,
        n_bits=n_bits,
        memory_strengths=None,
        edge_damping_messages=edge_damping_messages,
    )


def decode_with_edge_message_weights(
    tracer: relay_bp.MinSumBPDecoderTraceF64,
    syndrome: np.ndarray,
    edge_message_weights: np.ndarray,
    *,
    n_bits: int,
) -> Any:
    return decode_with_memory_edge_damping_and_edge_weights(
        tracer,
        syndrome,
        n_bits=n_bits,
        memory_strengths=None,
        edge_damping_messages=None,
        edge_message_weights=edge_message_weights,
    )


def decode_with_memory_and_edge_damping(
    tracer: relay_bp.MinSumBPDecoderTraceF64,
    syndrome: np.ndarray,
    *,
    n_bits: int,
    memory_strengths: np.ndarray | None,
    edge_damping_messages: np.ndarray | None,
) -> Any:
    return decode_with_memory_edge_damping_and_edge_weights(
        tracer,
        syndrome,
        n_bits=n_bits,
        memory_strengths=memory_strengths,
        edge_damping_messages=edge_damping_messages,
        edge_message_weights=None,
    )


def decode_with_memory_edge_damping_and_edge_weights(
    tracer: relay_bp.MinSumBPDecoderTraceF64,
    syndrome: np.ndarray,
    *,
    n_bits: int,
    memory_strengths: np.ndarray | None,
    edge_damping_messages: np.ndarray | None,
    edge_message_weights: np.ndarray | None,
) -> Any:
    tracer.reset()
    if edge_damping_messages is None:
        tracer.clear_explicit_c_damp_messages()
    else:
        tracer.set_explicit_c_damp_messages(np.asarray(edge_damping_messages, dtype=np.float64))
    supports_explicit_edge_weights = hasattr(
        tracer,
        "set_explicit_edge_message_weights",
    ) and hasattr(
        tracer,
        "clear_explicit_edge_message_weights",
    )
    if edge_message_weights is None:
        if supports_explicit_edge_weights:
            tracer.clear_explicit_edge_message_weights()
    else:
        if not supports_explicit_edge_weights:
            raise RuntimeError(
                "The loaded relay_bp extension does not expose explicit edge-message-weight methods. "
                "Rebuild/install the package so the Python bindings match the current source tree."
            )
        tracer.set_explicit_edge_message_weights(np.asarray(edge_message_weights, dtype=np.float64))
    if memory_strengths is None:
        tracer.set_memory_strengths(np.zeros(int(n_bits), dtype=np.float64))
    else:
        tracer.set_memory_strengths(np.asarray(memory_strengths, dtype=np.float64))
    return _finish_trace_decode(tracer, syndrome)


def build_edge_damping_messages(
    check_matrix: csc_matrix,
    *,
    interval: tuple[float, float],
    coeff_seed: int,
    support_bits: tuple[int, ...] | list[int] | None = None,
    outside_value: float = 1.0,
) -> np.ndarray:
    low, high = float(interval[0]), float(interval[1])
    if not (0.0 <= low <= high <= 1.0):
        raise ValueError(f"Damping interval must stay within [0, 1]; got {interval}.")
    if not (0.0 <= float(outside_value) <= 1.0):
        raise ValueError(f"outside_value must be in [0, 1]; got {outside_value}.")
    check_matrix = check_matrix.tocsc()
    messages = np.full(check_matrix.nnz, float(outside_value), dtype=np.float64)
    if support_bits is None:
        target_columns = np.arange(check_matrix.shape[1], dtype=int)
    else:
        target_columns = np.unique(np.asarray(support_bits, dtype=int))
    if target_columns.size == 0:
        return messages
    positions = []
    indptr = check_matrix.indptr
    for col_idx in target_columns:
        if col_idx < 0 or col_idx >= check_matrix.shape[1]:
            raise ValueError(f"support bit {col_idx} is out of bounds for {check_matrix.shape[1]} columns.")
        start = int(indptr[col_idx])
        end = int(indptr[col_idx + 1])
        if end > start:
            positions.append(np.arange(start, end, dtype=int))
    if not positions:
        return messages
    target_positions = np.concatenate(positions)
    if np.isclose(low, high):
        messages[target_positions] = low
    else:
        rng = np.random.default_rng(int(coeff_seed))
        messages[target_positions] = rng.uniform(low, high, size=target_positions.size)
    return messages


def build_edge_message_weights(
    check_matrix: csc_matrix,
    *,
    interval: tuple[float, float],
    coeff_seed: int,
    support_bits: tuple[int, ...] | list[int] | None = None,
    outside_value: float = 1.0,
) -> np.ndarray:
    low, high = float(interval[0]), float(interval[1])
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError(f"Edge-weight interval endpoints must be finite; got {interval}.")
    if low > high:
        raise ValueError(f"Edge-weight interval low must be <= high; got {interval}.")
    if not np.isfinite(float(outside_value)):
        raise ValueError(f"outside_value must be finite; got {outside_value}.")
    check_matrix = check_matrix.tocsc()
    weights = np.full(check_matrix.nnz, float(outside_value), dtype=np.float64)
    if support_bits is None:
        target_columns = np.arange(check_matrix.shape[1], dtype=int)
    else:
        target_columns = np.unique(np.asarray(support_bits, dtype=int))
    if target_columns.size == 0:
        return weights
    positions = []
    indptr = check_matrix.indptr
    for col_idx in target_columns:
        if col_idx < 0 or col_idx >= check_matrix.shape[1]:
            raise ValueError(f"support bit {col_idx} is out of bounds for {check_matrix.shape[1]} columns.")
        start = int(indptr[col_idx])
        end = int(indptr[col_idx + 1])
        if end > start:
            positions.append(np.arange(start, end, dtype=int))
    if not positions:
        return weights
    target_positions = np.concatenate(positions)
    if np.isclose(low, high):
        weights[target_positions] = low
    else:
        rng = np.random.default_rng(int(coeff_seed))
        weights[target_positions] = rng.uniform(low, high, size=target_positions.size)
    return weights
