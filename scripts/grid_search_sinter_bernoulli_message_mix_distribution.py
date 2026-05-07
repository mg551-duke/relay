#!/usr/bin/env python3
"""Crude grid search for Bernoulli relay message-mix weights."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import relay_bp
import stim  # type: ignore

from relay_bp.stim.sinter.check_matrices import CheckMatrices
from relay_bp.stim.sinter.runner import (
    _filter_detectors_by_basis,
    _normalize_basis_filter,
    find_circuits,
)

PARAMETER_NAMES = [
    "fresh_negative",
    "fresh_positive",
    "fresh_p_positive",
    "previous_negative",
    "previous_positive",
    "previous_p_positive",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuits-dir", type=Path, required=True)
    parser.add_argument("--circuit-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--filename-contains", type=str, default="")
    parser.add_argument("--basis-filter", type=str, default="none")
    parser.add_argument("--max-circuits", type=int, default=1)
    parser.add_argument("--max-shots", type=int, default=50_000)
    parser.add_argument("--shot-batch-size", type=int, default=256)
    parser.add_argument("--target-logical-errors", type=int, default=10)
    parser.add_argument("--distribution-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--decomposed-hyperedges", action="store_true")
    parser.add_argument("--allow-undecomposed-hyperedges", action="store_true")
    parser.add_argument("--no-prune-decided-errors", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--no-resume", action="store_true")

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma0", type=_optional_float, default=None)
    parser.add_argument("--random-decimation-candidates", type=int, default=1)
    parser.add_argument("--pre-iter", type=int, default=80)
    parser.add_argument("--num-sets", type=int, default=60)
    parser.add_argument("--set-max-iter", type=int, default=60)
    parser.add_argument("--no-relay-posteriors", action="store_true")
    parser.add_argument("--stop-nconv", type=int, default=5)
    parser.add_argument(
        "--stopping-criterion", choices=["pre_iter", "nconv", "all"], default="nconv"
    )
    parser.add_argument("--decimation-pre-iter", type=int, default=0)
    parser.add_argument("--initial-decimation-percentage", type=float, default=0.0)
    parser.add_argument("--r-low", type=int, default=0)
    parser.add_argument("--t-low", type=int, default=3)
    parser.add_argument("--r-high", type=int, default=0)
    parser.add_argument("--t-high", type=int, default=5)

    parser.add_argument("--fresh-negative-values", type=str, default="-0.1,-0.02,0.0")
    parser.add_argument("--fresh-positive-values", type=str, default="0.7,0.85,1.0")
    parser.add_argument(
        "--fresh-probability-values", type=str, default="0.9,0.97,0.995"
    )
    parser.add_argument(
        "--previous-negative-values", type=str, default="-0.25,-0.1,0.0"
    )
    parser.add_argument("--previous-positive-values", type=str, default="0.0,0.1,0.25")
    parser.add_argument(
        "--previous-probability-values", type=str, default="0.25,0.5,0.75"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    circuit_path = _select_circuit(args)
    circuit = stim.Circuit.from_file(circuit_path)
    basis_filter = _normalize_basis_filter(args.basis_filter)
    if basis_filter != "none":
        circuit = _filter_detectors_by_basis(circuit, basis_filter)

    check_matrices = CheckMatrices.from_dem(
        circuit.detector_error_model(),
        decomposed_hyperedges=_decomposed_hyperedges_arg(args),
        prune_decided_errors=not args.no_prune_decided_errors,
        threshold=args.threshold,
    )
    detectors, observables = _sample_adjusted_shots(
        circuit,
        check_matrices,
        shots=args.max_shots,
        seed=args.seed + 100_000,
    )

    grid = _parameter_grid(args)
    candidates = [
        (candidate_index, params)
        for candidate_index, params in enumerate(grid)
        if candidate_index % args.num_shards == args.shard_index
    ]
    if not candidates:
        raise SystemExit(
            f"Shard {args.shard_index}/{args.num_shards} has no candidates."
        )

    config_payload = _config_payload(
        args=args,
        circuit_path=circuit_path,
        basis_filter=basis_filter,
        full_grid_size=len(grid),
        shard_candidate_count=len(candidates),
    )
    _write_json(args.output_dir / "grid_search_config.json", config_payload)

    results_csv = args.output_dir / "grid_results.csv"
    evaluated = (
        _evaluated_candidate_indices(results_csv) if not args.no_resume else set()
    )
    rows = (
        _load_rows(results_csv) if results_csv.exists() and not args.no_resume else []
    )
    best_row = _best_row(rows)

    print(
        json.dumps(
            {
                "full_grid_size": len(grid),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "shard_candidate_count": len(candidates),
                "already_evaluated": len(evaluated),
                "max_shots": args.max_shots,
                "target_logical_errors": args.target_logical_errors,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for local_index, (candidate_index, params) in enumerate(candidates):
        if candidate_index in evaluated:
            continue
        row = _evaluate_candidate(
            args=args,
            check_matrices=check_matrices,
            detectors=detectors,
            observables=observables,
            params=params,
            candidate_index=candidate_index,
            local_index=local_index,
        )
        rows.append(row)
        _append_row(results_csv, row)
        best_row = _best_row([best_row, row] if best_row is not None else [row])
        if best_row is not None:
            _write_json(args.output_dir / "best_params.json", _best_payload(best_row))
        print(json.dumps(row, sort_keys=True), flush=True)

    sorted_rows = sorted(rows, key=_rank_key)
    _write_rows(args.output_dir / "grid_results_sorted.csv", sorted_rows)
    if sorted_rows:
        _write_json(args.output_dir / "best_params.json", _best_payload(sorted_rows[0]))
        print(json.dumps(_best_payload(sorted_rows[0]), indent=2, sort_keys=True))


def _evaluate_candidate(
    *,
    args: argparse.Namespace,
    check_matrices: CheckMatrices,
    detectors: np.ndarray,
    observables: np.ndarray,
    params: dict[str, float],
    candidate_index: int,
    local_index: int,
) -> dict[str, Any]:
    logical_failures = 0
    converged = 0
    iterations = 0
    trials = 0
    stopped_early = False

    for repeat_idx in range(args.distribution_repeats):
        repeat_seed = (
            int(args.seed) + 5_000_003 * int(candidate_index) + 10_007 * int(repeat_idx)
        )
        decoder = relay_bp.RelayedBPGDDecoderF64(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            alpha=None if args.alpha == 0.0 else args.alpha,
            gamma0=args.gamma0,
            random_decimation_candidates=args.random_decimation_candidates,
            pre_iter=args.pre_iter,
            num_sets=args.num_sets,
            set_max_iter=args.set_max_iter,
            message_mix_bernoulli=_params_tuple(params),
            relay_posteriors=not args.no_relay_posteriors,
            stop_nconv=args.stop_nconv,
            stopping_criterion=args.stopping_criterion,
            logging=False,
            seed=repeat_seed,
            decimation_pre_iter=args.decimation_pre_iter,
            initial_decimation_percentage=args.initial_decimation_percentage,
            r_low=args.r_low,
            t_low=args.t_low,
            r_high=args.r_high,
            t_high=args.t_high,
        )
        observable_decoder = relay_bp.ObservableDecoderRunner(
            decoder,
            check_matrices.observables_matrix,
            include_decode_result=True,
        )
        for batch_start in range(0, detectors.shape[0], args.shot_batch_size):
            batch_end = min(batch_start + args.shot_batch_size, detectors.shape[0])
            details = observable_decoder.decode_observables_detailed_batch(
                detectors[batch_start:batch_end],
                parallel=args.parallel,
                progress_bar=False,
            )
            for detail, truth in zip(
                details,
                observables[batch_start:batch_end],
                strict=True,
            ):
                predicted = np.asarray(detail.observables, dtype=np.uint8)
                if not np.array_equal(predicted, truth):
                    logical_failures += 1
                if bool(detail.converged):
                    converged += 1
                iterations += int(detail.iterations)
                trials += 1
                if logical_failures >= args.target_logical_errors:
                    stopped_early = True
                    break
            if stopped_early:
                break
        if stopped_early:
            break

    trials_f64 = float(max(trials, 1))
    row = {
        "candidate_index": int(candidate_index),
        "local_index": int(local_index),
        **params,
        "shots": int(detectors.shape[0]),
        "repeats": int(args.distribution_repeats),
        "trials": int(trials),
        "logical_failures": int(logical_failures),
        "target_logical_errors": int(args.target_logical_errors),
        "stopped_early": bool(stopped_early),
        "logical_failure_rate": float(logical_failures / trials_f64),
        "converged": int(converged),
        "convergence_rate": float(converged / trials_f64),
        "mean_iterations": float(iterations / trials_f64),
    }
    return row


def _parameter_grid(args: argparse.Namespace) -> list[dict[str, float]]:
    values = [
        _parse_float_values(args.fresh_negative_values, "fresh-negative-values"),
        _parse_float_values(args.fresh_positive_values, "fresh-positive-values"),
        _parse_float_values(args.fresh_probability_values, "fresh-probability-values"),
        _parse_float_values(args.previous_negative_values, "previous-negative-values"),
        _parse_float_values(args.previous_positive_values, "previous-positive-values"),
        _parse_float_values(
            args.previous_probability_values, "previous-probability-values"
        ),
    ]
    grid = []
    for combination in itertools.product(*values):
        params = dict(zip(PARAMETER_NAMES, map(float, combination), strict=True))
        _validate_message_mix_params(params)
        grid.append(params)
    return grid


def _validate_message_mix_params(params: dict[str, float]) -> None:
    if params["fresh_negative"] > 0.0:
        raise ValueError("fresh_negative must be <= 0.")
    if params["previous_negative"] > 0.0:
        raise ValueError("previous_negative must be <= 0.")
    if params["fresh_positive"] < 0.0:
        raise ValueError("fresh_positive must be >= 0.")
    if params["previous_positive"] < 0.0:
        raise ValueError("previous_positive must be >= 0.")
    if not 0.0 <= params["fresh_p_positive"] <= 1.0:
        raise ValueError("fresh_p_positive must be in [0, 1].")
    if not 0.0 <= params["previous_p_positive"] <= 1.0:
        raise ValueError("previous_p_positive must be in [0, 1].")
    if not all(np.isfinite(list(params.values()))):
        raise ValueError(f"Message-mix params must be finite: {params}")


def _params_tuple(
    params: dict[str, float],
) -> tuple[float, float, float, float, float, float]:
    return tuple(float(params[name]) for name in PARAMETER_NAMES)  # type: ignore[return-value]


def _parse_float_values(text: str, label: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"--{label} must contain at least one value.")
    return values


def _select_circuit(args: argparse.Namespace) -> Path:
    if args.circuit_file is not None:
        path = args.circuit_file
        if not path.is_absolute():
            path = args.circuits_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Selected circuit does not exist: {path}")
        return path

    paths = find_circuits(
        args.circuits_dir,
        args.filename_contains,
        max_circuits=args.max_circuits,
    )
    if len(paths) != 1:
        raise SystemExit(
            "Message-mix grid search expects exactly one circuit per run. "
            f"Matched {len(paths)} circuits; use --filename-contains or --max-circuits 1."
        )
    return paths[0]


def _decomposed_hyperedges_arg(args: argparse.Namespace) -> bool | None:
    if args.decomposed_hyperedges and args.allow_undecomposed_hyperedges:
        raise SystemExit(
            "Use at most one of --decomposed-hyperedges and --allow-undecomposed-hyperedges."
        )
    if args.decomposed_hyperedges:
        return True
    if args.allow_undecomposed_hyperedges:
        return False
    return None


def _sample_adjusted_shots(
    circuit: stim.Circuit,
    check_matrices: CheckMatrices,
    *,
    shots: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        sampler = circuit.compile_detector_sampler(seed=int(seed))
    except TypeError:
        sampler = circuit.compile_detector_sampler()
    detectors, observables = sampler.sample(
        shots=int(shots),
        separate_observables=True,
        bit_packed=False,
    )
    detectors = np.asarray(detectors, dtype=np.uint8)
    observables = np.asarray(observables, dtype=np.uint8)
    if check_matrices.syndrome_bias is not None:
        detectors = (
            detectors + np.asarray(check_matrices.syndrome_bias, dtype=np.uint8)
        ) % 2
    if check_matrices.observables_bias is not None:
        observables = (
            observables + np.asarray(check_matrices.observables_bias, dtype=np.uint8)
        ) % 2
    return detectors, observables


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_shots <= 0:
        raise ValueError("--max-shots must be positive.")
    if args.shot_batch_size <= 0:
        raise ValueError("--shot-batch-size must be positive.")
    if args.target_logical_errors <= 0:
        raise ValueError("--target-logical-errors must be positive.")
    if args.distribution_repeats <= 0:
        raise ValueError("--distribution-repeats must be positive.")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards).")


def _config_payload(
    *,
    args: argparse.Namespace,
    circuit_path: Path,
    basis_filter: str,
    full_grid_size: int,
    shard_candidate_count: int,
) -> dict[str, Any]:
    payload = vars(args).copy()
    payload["circuits_dir"] = str(args.circuits_dir)
    payload["output_dir"] = str(args.output_dir)
    payload["selected_circuit"] = str(circuit_path)
    payload["basis_filter_applied"] = basis_filter
    payload["full_grid_size"] = int(full_grid_size)
    payload["shard_candidate_count"] = int(shard_candidate_count)
    return _json_ready(payload)


def _evaluated_candidate_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            int(row["candidate_index"])
            for row in csv.DictReader(handle)
            if row.get("candidate_index")
        }


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [_coerce_row(row) for row in csv.DictReader(handle)]


def _coerce_row(row: dict[str, str]) -> dict[str, Any]:
    integer_fields = {
        "candidate_index",
        "local_index",
        "shots",
        "repeats",
        "trials",
        "logical_failures",
        "target_logical_errors",
        "converged",
    }
    bool_fields = {"stopped_early"}
    result: dict[str, Any] = {}
    for key, value in row.items():
        if key in integer_fields:
            result[key] = int(value)
        elif key in bool_fields:
            result[key] = value.lower() == "true"
        else:
            result[key] = float(value)
    return result


def _append_row(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames() -> list[str]:
    return [
        "candidate_index",
        "local_index",
        *PARAMETER_NAMES,
        "shots",
        "repeats",
        "trials",
        "logical_failures",
        "target_logical_errors",
        "stopped_early",
        "logical_failure_rate",
        "converged",
        "convergence_rate",
        "mean_iterations",
    ]


def _best_row(rows: Iterable[dict[str, Any] | None]) -> dict[str, Any] | None:
    valid_rows = [row for row in rows if row is not None]
    if not valid_rows:
        return None
    return min(valid_rows, key=_rank_key)


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(row["logical_failure_rate"]),
        float(row["mean_iterations"]),
        -float(row["convergence_rate"]),
        int(row["candidate_index"]),
    )


def _best_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_index": int(row["candidate_index"]),
        "params": {name: float(row[name]) for name in PARAMETER_NAMES},
        "metrics": {
            "shots": int(row["shots"]),
            "repeats": int(row["repeats"]),
            "trials": int(row["trials"]),
            "logical_failures": int(row["logical_failures"]),
            "max_logical_failures": int(row["target_logical_errors"]),
            "stopped_early": bool(row["stopped_early"]),
            "logical_failure_rate": float(row["logical_failure_rate"]),
            "convergence_rate": float(row["convergence_rate"]),
            "mean_iterations": float(row["mean_iterations"]),
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"none", "null", ""}:
        return None
    return float(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
