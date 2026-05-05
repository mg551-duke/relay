#!/usr/bin/env python3
"""Train deterministic static Relay memory assignments on a fixed Stim circuit."""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuits-dir", type=Path, required=True)
    parser.add_argument("--circuit-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--filename-contains", type=str, default="")
    parser.add_argument("--basis-filter", type=str, default="none")
    parser.add_argument("--max-circuits", type=int, default=1)
    parser.add_argument("--assignment-mode", choices=["discrete", "continuous", "both"], default="both")
    parser.add_argument("--discrete-optimizer", choices=["cem", "nes"], default="cem")
    parser.add_argument("--continuous-optimizer", choices=["cem", "nes"], default="cem")
    parser.add_argument("--warm-start-base", type=Path, default=None)
    parser.add_argument("--warm-start-mode", choices=["none", "same-seed", "best-seed"], default="same-seed")
    parser.add_argument("--train-shots", type=int, default=150_000)
    parser.add_argument("--validation-shots", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decomposed-hyperedges", action="store_true")
    parser.add_argument("--allow-undecomposed-hyperedges", action="store_true")
    parser.add_argument("--no-prune-decided-errors", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.0)

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma0", type=float, default=0.125)
    parser.add_argument("--c-damp", type=float, default=None)
    parser.add_argument("--random-decimation-candidates", type=int, default=1)
    parser.add_argument("--pre-iter", type=int, default=80)
    parser.add_argument("--num-sets", type=int, default=600)
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

    parser.add_argument("--negative", type=float, default=-0.0687506131581227)
    parser.add_argument("--positive", type=float, default=0.3557972204473916)
    parser.add_argument("--p-positive", type=float, default=0.5451025293701163)
    parser.add_argument("--memory-min", type=float, default=-0.3)
    parser.add_argument("--memory-max", type=float, default=0.66)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--elite-count", type=int, default=6)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--distribution-repeats", type=int, default=1)
    parser.add_argument("--initialization-candidates", type=int, default=32)
    parser.add_argument("--probability-floor", type=float, default=0.001)
    parser.add_argument("--initial-std", type=float, default=0.05)
    parser.add_argument("--std-floor", type=float, default=0.001)
    parser.add_argument("--smoothing", type=float, default=0.7)
    parser.add_argument("--discrete-nes-learning-rate", type=float, default=0.15)
    parser.add_argument("--continuous-nes-learning-rate", type=float, default=0.2)
    parser.add_argument("--continuous-nes-sigma-learning-rate", type=float, default=0.05)
    parser.add_argument("--continuous-nes-impact-decay", type=float, default=0.9)
    parser.add_argument("--continuous-sigma-min", type=float, default=0.0001)
    parser.add_argument("--continuous-sigma-max", type=float, default=0.25)
    parser.add_argument("--train-max-logical-failures", type=int, default=50)
    parser.add_argument("--validation-max-logical-failures", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
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

    _write_json(args.output_dir / "training_config.json", _config_payload(args, circuit_path, basis_filter))

    summary: dict[str, Any] = {}
    discrete_result: dict[str, Any] | None = None
    if args.assignment_mode in {"discrete", "both"}:
        discrete_result = _run_discrete(
            args,
            check_matrices,
            train_detectors,
            train_observables,
            validation_detectors,
            validation_observables,
        )
        summary["discrete"] = discrete_result

    if args.assignment_mode in {"continuous", "both"}:
        initial_gammas = None
        if args.assignment_mode == "both":
            initial_gammas_path = args.output_dir / "discrete_best_static_gammas.npy"
            if initial_gammas_path.exists():
                initial_gammas = _load_static_gamma_vector(initial_gammas_path)
            elif discrete_result is not None:
                initial_gammas = np.asarray(discrete_result["best_candidate"]["gammas"], dtype=np.float64)
        continuous_result = _run_continuous(
            args,
            check_matrices,
            train_detectors,
            train_observables,
            validation_detectors,
            validation_observables,
            initial_gammas=initial_gammas,
        )
        summary["continuous"] = continuous_result

    _write_json(args.output_dir / "training_summary.json", summary)
    print(json.dumps(_summary_preview(summary), indent=2, sort_keys=True))


def _run_discrete(
    args: argparse.Namespace,
    check_matrices: CheckMatrices,
    train_detectors: np.ndarray,
    train_observables: np.ndarray,
    validation_detectors: np.ndarray,
    validation_observables: np.ndarray,
) -> dict[str, Any]:
    best_params_path = args.output_dir / "discrete_best_params.json"
    if best_params_path.exists() and not args.no_resume:
        print(f"Existing discrete training result found: {best_params_path}")
        return {"best_candidate": _load_json(best_params_path)}

    resume_state = _load_checkpoint_mode(args.output_dir / "checkpoint_latest.npz", "discrete", args.no_resume)
    warm_start = _load_discrete_warm_start(args)

    def progress_callback(progress: dict[str, Any]) -> None:
        progress = _json_ready(progress)
        _update_checkpoint(args.output_dir / "checkpoint_latest.npz", "discrete", progress, finalized=False)
        _write_generation_csv(args.output_dir / "generation_metrics.csv", _load_checkpoint(args.output_dir / "checkpoint_latest.npz"))
        print(_progress_line("discrete", progress), flush=True)

    result = relay_bp.train_relayed_bpgd_static_discrete_memory(
        check_matrices.check_matrix,
        check_matrices.observables_matrix,
        check_matrices.error_priors,
        train_detectors,
        train_observables,
        validation_detectors,
        validation_observables,
        **_decoder_kwargs(args),
        negative=args.negative,
        positive=args.positive,
        p_positive=args.p_positive,
        candidate_count=args.candidate_count,
        elite_count=args.elite_count,
        generations=args.generations,
        distribution_repeats=args.distribution_repeats,
        probability_floor=args.probability_floor,
        smoothing=args.smoothing,
        optimizer=args.discrete_optimizer,
        nes_learning_rate=args.discrete_nes_learning_rate,
        initial_probability_vector=warm_start["initial_probability_vector"],
        initial_mask=warm_start["initial_mask"],
        train_max_logical_failures=_positive_or_none(args.train_max_logical_failures),
        validation_max_logical_failures=_positive_or_none(args.validation_max_logical_failures),
        seed=args.seed,
        resume_state=resume_state,
        progress_callback=progress_callback,
    )
    result = _json_ready(result)
    mask = np.asarray(result["best_candidate"]["mask"], dtype=np.uint8)
    gammas = np.asarray(result["best_candidate"]["gammas"], dtype=np.float64)
    probability_vector = np.asarray(result["final_probability_vector"], dtype=np.float64)
    np.save(args.output_dir / "discrete_best_mask.npy", mask)
    np.save(args.output_dir / "discrete_best_static_gammas.npy", gammas.reshape(1, -1))
    np.save(args.output_dir / "discrete_probability_vector.npy", probability_vector)

    best_payload = _best_payload(
        result["best_candidate"],
        mode="discrete",
        args=args,
        artifact_path=args.output_dir / "discrete_best_static_gammas.npy",
    )
    _write_json(best_params_path, best_payload)
    _write_validation_metrics(args.output_dir, "discrete", result["validation_metrics"])
    _update_checkpoint(args.output_dir / "checkpoint_latest.npz", "discrete", result, finalized=True)
    _write_generation_csv(args.output_dir / "generation_metrics.csv", _load_checkpoint(args.output_dir / "checkpoint_latest.npz"))
    return result


def _run_continuous(
    args: argparse.Namespace,
    check_matrices: CheckMatrices,
    train_detectors: np.ndarray,
    train_observables: np.ndarray,
    validation_detectors: np.ndarray,
    validation_observables: np.ndarray,
    *,
    initial_gammas: np.ndarray | None,
) -> dict[str, Any]:
    best_params_path = args.output_dir / "continuous_best_params.json"
    if best_params_path.exists() and not args.no_resume:
        print(f"Existing continuous training result found: {best_params_path}")
        return {"best_candidate": _load_json(best_params_path)}

    resume_state = _load_checkpoint_mode(args.output_dir / "checkpoint_latest.npz", "continuous", args.no_resume)
    warm_start_gammas = _load_continuous_warm_start(args)
    if warm_start_gammas is not None:
        initial_gammas = warm_start_gammas

    def progress_callback(progress: dict[str, Any]) -> None:
        progress = _json_ready(progress)
        _update_checkpoint(args.output_dir / "checkpoint_latest.npz", "continuous", progress, finalized=False)
        _write_generation_csv(args.output_dir / "generation_metrics.csv", _load_checkpoint(args.output_dir / "checkpoint_latest.npz"))
        print(_progress_line("continuous", progress), flush=True)

    result = relay_bp.train_relayed_bpgd_static_continuous_memory(
        check_matrices.check_matrix,
        check_matrices.observables_matrix,
        check_matrices.error_priors,
        train_detectors,
        train_observables,
        validation_detectors,
        validation_observables,
        **_decoder_kwargs(args),
        negative=args.negative,
        positive=args.positive,
        p_positive=args.p_positive,
        memory_min=args.memory_min,
        memory_max=args.memory_max,
        candidate_count=args.candidate_count,
        elite_count=args.elite_count,
        generations=args.generations,
        distribution_repeats=args.distribution_repeats,
        initialization_candidates=args.initialization_candidates,
        initial_std=args.initial_std,
        std_floor=args.std_floor,
        smoothing=args.smoothing,
        optimizer=args.continuous_optimizer,
        nes_learning_rate=args.continuous_nes_learning_rate,
        nes_sigma_learning_rate=args.continuous_nes_sigma_learning_rate,
        nes_impact_decay=args.continuous_nes_impact_decay,
        sigma_min=args.continuous_sigma_min,
        sigma_max=args.continuous_sigma_max,
        train_max_logical_failures=_positive_or_none(args.train_max_logical_failures),
        validation_max_logical_failures=_positive_or_none(args.validation_max_logical_failures),
        seed=args.seed + 1_000_000,
        initial_gammas=initial_gammas,
        resume_state=resume_state,
        progress_callback=progress_callback,
    )
    result = _json_ready(result)
    gammas = np.asarray(result["best_candidate"]["gammas"], dtype=np.float64)
    mean_gammas = np.asarray(result["final_mean_gammas"], dtype=np.float64)
    std_gammas = np.asarray(result["final_std_gammas"], dtype=np.float64)
    impact_gammas = np.asarray(result.get("final_impact_gammas", []), dtype=np.float64)
    np.save(args.output_dir / "continuous_best_static_gammas.npy", gammas.reshape(1, -1))
    np.save(args.output_dir / "continuous_mean_gammas.npy", mean_gammas)
    np.save(args.output_dir / "continuous_sigma_gammas.npy", std_gammas)
    if impact_gammas.size:
        np.save(args.output_dir / "continuous_impact_vector.npy", impact_gammas)

    best_payload = _best_payload(
        result["best_candidate"],
        mode="continuous",
        args=args,
        artifact_path=args.output_dir / "continuous_best_static_gammas.npy",
    )
    _write_json(best_params_path, best_payload)
    _write_validation_metrics(args.output_dir, "continuous", result["validation_metrics"])
    _update_checkpoint(args.output_dir / "checkpoint_latest.npz", "continuous", result, finalized=True)
    _write_generation_csv(args.output_dir / "generation_metrics.csv", _load_checkpoint(args.output_dir / "checkpoint_latest.npz"))
    return result


def _decoder_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "alpha": None if args.alpha == 0.0 else args.alpha,
        "gamma0": args.gamma0,
        "c_damp": args.c_damp,
        "random_decimation_candidates": args.random_decimation_candidates,
        "pre_iter": args.pre_iter,
        "num_sets": args.num_sets,
        "set_max_iter": args.set_max_iter,
        "relay_posteriors": not args.no_relay_posteriors,
        "stop_nconv": args.stop_nconv,
        "stopping_criterion": args.stopping_criterion,
        "decimation_pre_iter": args.decimation_pre_iter,
        "initial_decimation_percentage": args.initial_decimation_percentage,
        "r_low": args.r_low,
        "t_low": args.t_low,
        "r_high": args.r_high,
        "t_high": args.t_high,
    }


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
            "Static assignment training expects exactly one circuit per run. "
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


def _best_payload(candidate: dict[str, Any], *, mode: str, args: argparse.Namespace, artifact_path: Path) -> dict[str, Any]:
    payload = dict(candidate)
    payload["assignment_mode"] = mode
    payload["optimizer"] = args.discrete_optimizer if mode == "discrete" else args.continuous_optimizer
    payload["warm_start_base"] = str(args.warm_start_base) if args.warm_start_base is not None else None
    payload["warm_start_mode"] = args.warm_start_mode
    payload["seed_params"] = {
        "negative": float(args.negative),
        "positive": float(args.positive),
        "p_positive": float(args.p_positive),
    }
    payload["static_gammas_path"] = str(artifact_path)
    return _json_ready(payload)


def _write_generation_csv(path: Path, checkpoint: dict[str, Any]) -> None:
    rows = []
    for mode in ["discrete", "continuous"]:
        payload = checkpoint.get(mode)
        if not isinstance(payload, dict):
            continue
        for record in payload.get("generation_records", []):
            best = record["best_candidate"]
            metrics = best["metrics"]
            row = {
                "mode": mode,
                "generation": record["generation"],
                "best_trials": metrics["trials"],
                "best_logical_failures": metrics["logical_failures"],
                "best_stopped_early": metrics["stopped_early"],
                "best_logical_failure_rate": metrics["logical_failure_rate"],
                "best_mean_iterations": metrics["mean_iterations"],
                "best_convergence_rate": metrics["convergence_rate"],
                "state_mean": "",
                "state_std": "",
                "positive_fraction": "",
                "impact_mean": "",
                "impact_std": "",
            }
            if mode == "discrete":
                probabilities = np.asarray(record["probability_vector"], dtype=np.float64)
                mask = np.asarray(best["mask"], dtype=np.uint8)
                row["state_mean"] = float(probabilities.mean())
                row["state_std"] = float(probabilities.std())
                row["positive_fraction"] = float(mask.mean())
            else:
                mean_gammas = np.asarray(record["mean_gammas"], dtype=np.float64)
                std_gammas = np.asarray(record["std_gammas"], dtype=np.float64)
                row["state_mean"] = float(mean_gammas.mean())
                row["state_std"] = float(std_gammas.mean())
                if "impact_gammas" in record:
                    impact_gammas = np.asarray(record["impact_gammas"], dtype=np.float64)
                    row["impact_mean"] = float(impact_gammas.mean())
                    row["impact_std"] = float(impact_gammas.std())
            rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["mode", "generation"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_validation_metrics(output_dir: Path, mode: str, metrics: dict[str, Any]) -> None:
    path = output_dir / "validation_metrics.json"
    payload = _load_json(path) if path.exists() else {}
    payload[mode] = metrics
    _write_json(path, payload)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        if "payload_json" not in data:
            return {}
        return json.loads(str(data["payload_json"].item()))


def _load_checkpoint_mode(path: Path, mode: str, no_resume: bool) -> dict[str, Any] | None:
    if no_resume:
        return None
    payload = _load_checkpoint(path)
    state = payload.get(mode)
    if isinstance(state, dict) and not state.get("finalized", False):
        completed = int(state.get("completed_generations", 0))
        print(f"Resuming {mode} training from completed generation {completed}.")
        return state
    return None


def _update_checkpoint(path: Path, mode: str, payload: dict[str, Any], *, finalized: bool) -> None:
    checkpoint = _load_checkpoint(path)
    updated = dict(payload)
    updated["finalized"] = bool(finalized)
    checkpoint[mode] = _json_ready(updated)
    np.savez(path, payload_json=np.array(json.dumps(checkpoint, sort_keys=True)))


def _load_static_gamma_vector(path: Path) -> np.ndarray:
    gammas = np.asarray(np.load(path), dtype=np.float64)
    if gammas.ndim == 2:
        if gammas.shape[0] != 1:
            raise ValueError(f"Expected one static gamma row in {path}; got shape {gammas.shape}.")
        gammas = gammas[0]
    if gammas.ndim != 1:
        raise ValueError(f"Expected 1D or single-row static gammas in {path}; got shape {gammas.shape}.")
    return gammas


def _load_discrete_warm_start(args: argparse.Namespace) -> dict[str, np.ndarray | None]:
    state: dict[str, np.ndarray | None] = {
        "initial_probability_vector": None,
        "initial_mask": None,
    }
    directory = _resolve_warm_start_dir(args, "discrete")
    if directory is None:
        return state

    probability_path = directory / "discrete_probability_vector.npy"
    mask_path = directory / "discrete_best_mask.npy"
    gammas_path = directory / "discrete_best_static_gammas.npy"

    if probability_path.exists():
        probabilities = np.asarray(np.load(probability_path), dtype=np.float64)
        if probabilities.ndim != 1:
            raise ValueError(f"Expected 1D discrete probability vector in {probability_path}; got {probabilities.shape}.")
        state["initial_probability_vector"] = probabilities
        print(f"Warm-starting discrete probabilities from {probability_path}")

    if mask_path.exists():
        mask = np.asarray(np.load(mask_path), dtype=np.uint8)
        if mask.ndim != 1:
            raise ValueError(f"Expected 1D discrete mask in {mask_path}; got {mask.shape}.")
        state["initial_mask"] = mask
        print(f"Warm-starting discrete best mask from {mask_path}")
    elif gammas_path.exists():
        gammas = _load_static_gamma_vector(gammas_path)
        midpoint = 0.5 * (float(args.negative) + float(args.positive))
        state["initial_mask"] = (gammas >= midpoint).astype(np.uint8)
        print(f"Warm-starting discrete mask converted from {gammas_path}")

    return state


def _load_continuous_warm_start(args: argparse.Namespace) -> np.ndarray | None:
    directory = _resolve_warm_start_dir(args, "continuous")
    for path in _continuous_warm_start_candidates(args, directory):
        if path.exists():
            print(f"Warm-starting continuous gammas from {path}")
            return _load_static_gamma_vector(path)
    return None


def _continuous_warm_start_candidates(args: argparse.Namespace, directory: Path | None) -> list[Path]:
    paths: list[Path] = []
    if directory is not None:
        paths.append(directory / "continuous_best_static_gammas.npy")
        paths.append(directory / "discrete_best_static_gammas.npy")

    fallback_dir = _matching_discrete_warm_start_dir(args)
    if fallback_dir is not None and fallback_dir != directory:
        paths.append(fallback_dir / "discrete_best_static_gammas.npy")
    return paths


def _resolve_warm_start_dir(args: argparse.Namespace, mode: str) -> Path | None:
    if args.warm_start_mode == "none" or args.warm_start_base is None:
        return None
    base = args.warm_start_base
    if args.warm_start_mode == "same-seed":
        if _warm_start_dir_has_artifact(base, mode):
            return base
        seed_dir = base / f"seed_{args.seed}"
        if seed_dir.exists():
            return seed_dir
        return None
    return _best_seed_warm_start_dir(base, mode)


def _matching_discrete_warm_start_dir(args: argparse.Namespace) -> Path | None:
    if args.warm_start_mode == "none" or args.warm_start_base is None:
        return None
    base = Path(str(args.warm_start_base).replace("static_continuous_memory", "static_discrete_memory"))
    shadow = argparse.Namespace(**vars(args))
    shadow.warm_start_base = base
    return _resolve_warm_start_dir(shadow, "discrete")


def _warm_start_dir_has_artifact(directory: Path, mode: str) -> bool:
    if mode == "discrete":
        names = [
            "discrete_probability_vector.npy",
            "discrete_best_mask.npy",
            "discrete_best_static_gammas.npy",
        ]
    else:
        names = [
            "continuous_best_static_gammas.npy",
            "discrete_best_static_gammas.npy",
        ]
    return any((directory / name).exists() for name in names)


def _best_seed_warm_start_dir(base: Path, mode: str) -> Path | None:
    directories = [base] if _warm_start_dir_has_artifact(base, mode) else sorted(base.glob("seed_*"))
    best_dir: Path | None = None
    best_metrics: dict[str, Any] | None = None
    params_name = f"{mode}_best_params.json"
    for directory in directories:
        params_path = directory / params_name
        if not params_path.exists() and mode == "continuous":
            params_path = directory / "discrete_best_params.json"
        if not params_path.exists():
            continue
        try:
            payload = _load_json(params_path)
            metrics = payload.get("metrics", {})
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metrics, dict):
            continue
        if best_metrics is None or _metrics_key(metrics) < _metrics_key(best_metrics):
            best_metrics = metrics
            best_dir = directory
    return best_dir


def _metrics_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics.get("logical_failure_rate", float("inf"))),
        float(metrics.get("mean_iterations", float("inf"))),
        -float(metrics.get("convergence_rate", 0.0)),
    )


def _progress_line(mode: str, progress: dict[str, Any]) -> str:
    best = progress["best_candidate"]
    generation_best = progress.get("generation_record", {}).get("best_candidate")
    return json.dumps(
        {
            "mode": mode,
            "completed_generations": progress["completed_generations"],
            "best_metrics": best["metrics"],
            "generation_best_metrics": generation_best.get("metrics") if generation_best else None,
        },
        sort_keys=True,
    )


def _summary_preview(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        mode: {
            "best_candidate": payload.get("best_candidate"),
            "validation_metrics": payload.get("validation_metrics"),
        }
        for mode, payload in summary.items()
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _positive_or_none(value: int) -> int | None:
    return int(value) if value > 0 else None


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
