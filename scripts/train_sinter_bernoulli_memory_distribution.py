#!/usr/bin/env python3
"""Train Bernoulli relay memory weights on a fixed Stim circuit sample."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import relay_bp
import stim  # type: ignore

from relay_bp.stim.sinter.check_matrices import CheckMatrices
from relay_bp.stim.sinter.runner import (
    _filter_detectors_by_basis,
    _normalize_basis_filter,
    find_circuits,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuits-dir", type=Path, required=True)
    parser.add_argument("--circuit-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--filename-contains", type=str, default="")
    parser.add_argument("--basis-filter", type=str, default="none")
    parser.add_argument("--max-circuits", type=int, default=1)
    parser.add_argument("--train-shots", type=int, default=2_000)
    parser.add_argument("--validation-shots", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decomposed-hyperedges", action="store_true")
    parser.add_argument("--allow-undecomposed-hyperedges", action="store_true")
    parser.add_argument("--no-prune-decided-errors", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.0)

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma0", type=float, default=0.1)
    parser.add_argument("--c-damp", type=float, default=None)
    parser.add_argument("--random-decimation-candidates", type=int, default=1)
    parser.add_argument("--pre-iter", type=int, default=80)
    parser.add_argument("--num-sets", type=int, default=300)
    parser.add_argument("--set-max-iter", type=int, default=60)
    parser.add_argument("--no-relay-posteriors", action="store_true")
    parser.add_argument("--stop-nconv", type=int, default=5)
    parser.add_argument("--stopping-criterion", choices=["pre_iter", "nconv", "all"], default="nconv")
    parser.add_argument("--decimation-pre-iter", type=int, default=0)
    parser.add_argument("--initial-decimation-percentage", type=float, default=0.0)
    parser.add_argument("--r-low", type=int, default=0)
    parser.add_argument("--t-low", type=int, default=3)
    parser.add_argument("--r-high", type=int, default=0)
    parser.add_argument("--t-high", type=int, default=5)

    parser.add_argument("--negative-min", type=float, default=-0.3)
    parser.add_argument("--negative-max", type=float, default=0.0)
    parser.add_argument("--positive-min", type=float, default=0.0)
    parser.add_argument("--positive-max", type=float, default=0.66)
    parser.add_argument("--probability-min", type=float, default=0.001)
    parser.add_argument("--probability-max", type=float, default=0.999)
    parser.add_argument("--candidate-count", type=int, default=64)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--distribution-repeats", type=int, default=1)
    parser.add_argument("--local-refinement-steps", type=int, default=3)
    parser.add_argument("--initial-std", type=float, default=1.0)
    parser.add_argument("--std-floor", type=float, default=0.05)
    parser.add_argument("--smoothing", type=float, default=0.7)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_params_path = args.output_dir / "best_params.json"
    checkpoint_path = args.output_dir / "checkpoint_latest.json"
    if best_params_path.exists() and not args.no_resume:
        print(f"Existing training result found: {best_params_path}")
        return
    resume_state = None
    if checkpoint_path.exists() and not args.no_resume:
        resume_state = _load_json(checkpoint_path)
        completed = int(resume_state.get("completed_generations", 0))
        print(f"Resuming from completed generation {completed}.")

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
    train_detectors, train_observables = _sample_adjusted_shots(
        circuit,
        check_matrices,
        shots=args.train_shots,
        seed=args.seed + 100_000,
    )
    validation_detectors, validation_observables = _sample_adjusted_shots(
        circuit,
        check_matrices,
        shots=args.validation_shots,
        seed=args.seed + 200_000,
    )

    config_payload = _config_payload(args, circuit_path, basis_filter)
    _write_json(args.output_dir / "training_config.json", config_payload)

    def progress_callback(progress: dict[str, Any]) -> None:
        progress = _json_ready(progress)
        _write_generation_csv(args.output_dir / "generation_metrics.csv", progress["generation_records"])
        _write_json(
            checkpoint_path,
            {
                "completed_generations": progress["completed_generations"],
                "final_raw_mean": progress["final_raw_mean"],
                "final_raw_std": progress["final_raw_std"],
                "best_candidate": progress["best_candidate"],
                "generation_records": progress["generation_records"],
                "finalized": False,
            },
        )
        print(
            json.dumps(
                {
                    "completed_generations": progress["completed_generations"],
                    "best_candidate": progress["best_candidate"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    result = relay_bp.train_relayed_bpgd_bernoulli_memory(
        check_matrices.check_matrix,
        check_matrices.observables_matrix,
        check_matrices.error_priors,
        train_detectors,
        train_observables,
        validation_detectors,
        validation_observables,
        alpha=None if args.alpha == 0.0 else args.alpha,
        gamma0=args.gamma0,
        c_damp=args.c_damp,
        random_decimation_candidates=args.random_decimation_candidates,
        pre_iter=args.pre_iter,
        num_sets=args.num_sets,
        set_max_iter=args.set_max_iter,
        relay_posteriors=not args.no_relay_posteriors,
        stop_nconv=args.stop_nconv,
        stopping_criterion=args.stopping_criterion,
        decimation_pre_iter=args.decimation_pre_iter,
        initial_decimation_percentage=args.initial_decimation_percentage,
        r_low=args.r_low,
        t_low=args.t_low,
        r_high=args.r_high,
        t_high=args.t_high,
        negative_min=args.negative_min,
        negative_max=args.negative_max,
        positive_min=args.positive_min,
        positive_max=args.positive_max,
        probability_min=args.probability_min,
        probability_max=args.probability_max,
        candidate_count=args.candidate_count,
        elite_count=args.elite_count,
        generations=args.generations,
        distribution_repeats=args.distribution_repeats,
        local_refinement_steps=args.local_refinement_steps,
        initial_std=args.initial_std,
        std_floor=args.std_floor,
        smoothing=args.smoothing,
        seed=args.seed,
        resume_state=resume_state,
        progress_callback=progress_callback,
    )
    result = _json_ready(result)
    _write_generation_csv(args.output_dir / "generation_metrics.csv", result["generation_records"])
    _write_json(best_params_path, result["best_candidate"])
    _write_json(args.output_dir / "validation_metrics.json", result["validation_metrics"])
    _write_json(
        checkpoint_path,
        {
            "completed_generations": args.generations,
            "final_raw_mean": result["final_raw_mean"],
            "final_raw_std": result["final_raw_std"],
            "best_candidate": result["best_candidate"],
            "validation_metrics": result["validation_metrics"],
            "generation_records": result["generation_records"],
            "finalized": True,
        },
    )
    _write_json(args.output_dir / "training_summary.json", result)
    print(json.dumps(result["best_candidate"], indent=2, sort_keys=True))
    print(json.dumps(result["validation_metrics"], indent=2, sort_keys=True))


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
            "Bernoulli training v1 expects exactly one circuit per run. "
            f"Matched {len(paths)} circuits; use --filename-contains or --max-circuits 1."
        )
    return paths[0]


def _decomposed_hyperedges_arg(args: argparse.Namespace) -> bool | None:
    if args.decomposed_hyperedges and args.allow_undecomposed_hyperedges:
        raise SystemExit("Use at most one of --decomposed-hyperedges and --allow-undecomposed-hyperedges.")
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
        detectors = (detectors + np.asarray(check_matrices.syndrome_bias, dtype=np.uint8)) % 2
    if check_matrices.observables_bias is not None:
        observables = (observables + np.asarray(check_matrices.observables_bias, dtype=np.uint8)) % 2
    return detectors, observables


def _config_payload(args: argparse.Namespace, circuit_path: Path, basis_filter: str) -> dict[str, Any]:
    payload = vars(args).copy()
    payload["circuits_dir"] = str(args.circuits_dir)
    payload["output_dir"] = str(args.output_dir)
    payload["selected_circuit"] = str(circuit_path)
    payload["basis_filter_applied"] = basis_filter
    return _json_ready(payload)


def _write_generation_csv(path: Path, records: list[dict[str, Any]]) -> None:
    rows = []
    for record in records:
        best = record["best_candidate"]
        best_params = best["params"]
        best_metrics = best["metrics"]
        mean_params = record["mean_params"]
        rows.append(
            {
                "generation": record["generation"],
                "best_negative": best_params["negative"],
                "best_positive": best_params["positive"],
                "best_p_positive": best_params["p_positive"],
                "best_logical_failure_rate": best_metrics["logical_failure_rate"],
                "best_mean_iterations": best_metrics["mean_iterations"],
                "best_convergence_rate": best_metrics["convergence_rate"],
                "mean_negative": mean_params["negative"],
                "mean_positive": mean_params["positive"],
                "mean_p_positive": mean_params["p_positive"],
                "raw_std_negative": record["raw_std"][0],
                "raw_std_positive": record["raw_std"][1],
                "raw_std_probability": record["raw_std"][2],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["generation"])
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
