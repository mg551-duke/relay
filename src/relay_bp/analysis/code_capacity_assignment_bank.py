from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from .code_capacity_common import (
    CodeCapacityProblem,
    decode_with_edge_damping,
    decode_with_memory_and_edge_damping,
    decode_with_memory_strengths,
    evaluate_decode_result,
    make_min_sum_tracer,
)
from .code_capacity_memory import bootstrap_confidence_interval
from .code_capacity_training import _append_csv_row, _write_json

StaticAssignmentFamily = Literal["memory", "damping", "joint"]

FAMILY_CHOICES = {"memory", "damping", "joint"}
SAMPLING_FAMILIES = {"uniform_interval", "truncated_gaussian"}
SELECTION_MODES = {"practical", "oracle"}


@dataclass(frozen=True)
class ScalarDistributionSpec:
    sampling_family: str
    low: float
    high: float
    mean: float | None = None
    sigma: float | None = None

    def __post_init__(self) -> None:
        if str(self.sampling_family) not in SAMPLING_FAMILIES:
            raise ValueError(f"Unsupported sampling_family: {self.sampling_family!r}")
        if float(self.high) < float(self.low):
            raise ValueError("high must be >= low.")
        if str(self.sampling_family) == "truncated_gaussian":
            if self.mean is None:
                raise ValueError("truncated_gaussian requires mean.")
            if self.sigma is None:
                raise ValueError("truncated_gaussian requires sigma.")
            if float(self.sigma) < 0.0:
                raise ValueError("sigma must be non-negative.")


@dataclass(frozen=True)
class StaticAssignmentBankConfig:
    code_path: str
    checkpoint_dir: str
    family: str
    bank_size: int = 1
    max_iter: int = 40
    alpha: float | None = 1.0
    train_random_error_rate: float = 0.1
    validation_random_error_rate: float = 0.1
    test_random_error_rate: float = 0.1
    random_train_samples: int = 4096
    random_validation_samples: int = 2048
    random_test_samples: int = 4096
    random_train_batch_size: int = 256
    candidate_count: int = 64
    elite_count: int = 8
    rounds: int = 80
    validation_interval: int = 5
    practical_k: int = 8
    logical_failure_penalty: float = 20.0
    logical_weight_penalty: float = 4.0
    convergence_penalty: float = 1.0
    iteration_penalty: float = 0.1
    selection_half_weight: float = 1.0
    diversity_penalty: float = 0.01
    diversity_threshold_fraction: float = 0.05
    memory_min: float = -0.3
    memory_max: float = 0.3
    damping_min: float = 0.0
    damping_max: float = 1.0
    memory_init_low: float | None = None
    memory_init_high: float | None = None
    damping_init_low: float | None = None
    damping_init_high: float | None = None
    initial_memory_sigma: float | None = None
    initial_damping_sigma: float | None = None
    sigma_floor: float = 1e-3
    memory_sigma_ceiling: float | None = None
    damping_sigma_ceiling: float | None = None
    initialization_candidates: int = 8
    base_seed: int = 0

    def __post_init__(self) -> None:
        if str(self.family) not in FAMILY_CHOICES:
            raise ValueError(f"Unsupported family: {self.family!r}")
        if int(self.bank_size) <= 0:
            raise ValueError("bank_size must be positive.")
        if int(self.candidate_count) <= 0:
            raise ValueError("candidate_count must be positive.")
        if int(self.elite_count) <= 0:
            raise ValueError("elite_count must be positive.")
        if int(self.elite_count) > int(self.candidate_count):
            raise ValueError("elite_count must be <= candidate_count.")
        if int(self.rounds) <= 0:
            raise ValueError("rounds must be positive.")
        if int(self.validation_interval) <= 0:
            raise ValueError("validation_interval must be positive.")
        if int(self.practical_k) <= 0:
            raise ValueError("practical_k must be positive.")
        if int(self.initialization_candidates) <= 0:
            raise ValueError("initialization_candidates must be positive.")
        if float(self.memory_max) < float(self.memory_min):
            raise ValueError("memory_max must be >= memory_min.")
        if float(self.damping_max) < float(self.damping_min):
            raise ValueError("damping_max must be >= damping_min.")


def evaluate_assignment_bank(
    *,
    problem: CodeCapacityProblem,
    cases: Sequence[dict[str, Any]],
    family: str,
    memory_bank: np.ndarray | None,
    damping_bank: np.ndarray | None,
    max_iter: int,
    alpha: float | None,
    selection_mode: str,
    logical_failure_penalty: float,
    logical_weight_penalty: float,
    convergence_penalty: float,
    iteration_penalty: float,
    return_case_rows: bool = False,
) -> dict[str, Any]:
    family = _validate_family(family)
    selection_mode = _validate_selection_mode(selection_mode)
    validated_memory_bank, validated_damping_bank = _validate_bank_shapes(
        problem=problem,
        family=family,
        memory_bank=memory_bank,
        damping_bank=damping_bank,
    )
    bank_size = _bank_size(validated_memory_bank, validated_damping_bank)
    tracer = _make_tracer(problem=problem, family=family, max_iter=max_iter, alpha=alpha)
    logical_success_sum = 0.0
    exact_recovery_sum = 0.0
    convergence_sum = 0.0
    iterations_sum = 0.0
    logical_weight_sum = 0.0
    loss_sum = 0.0
    case_rows: list[dict[str, Any]] = []

    for case in cases:
        member_rows: list[dict[str, Any]] = []
        for bank_idx in range(bank_size):
            memory_vector = None if validated_memory_bank is None else validated_memory_bank[bank_idx]
            damping_vector = None if validated_damping_bank is None else validated_damping_bank[bank_idx]
            member_rows.append(
                _decode_assignment_member(
                    problem=problem,
                    tracer=tracer,
                    case=case,
                    family=family,
                    memory_vector=memory_vector,
                    damping_vector=damping_vector,
                    bank_idx=bank_idx,
                    max_iter=max_iter,
                    logical_failure_penalty=logical_failure_penalty,
                    logical_weight_penalty=logical_weight_penalty,
                    convergence_penalty=convergence_penalty,
                    iteration_penalty=iteration_penalty,
                )
            )
        selected_row = _select_assignment_row(member_rows, selection_mode=selection_mode)
        logical_success_sum += float(selected_row["logical_success"])
        exact_recovery_sum += float(selected_row["exact_recovery"])
        convergence_sum += float(selected_row["converged"])
        iterations_sum += float(selected_row["iterations"])
        logical_weight_sum += float(selected_row["logical_weight"])
        loss_sum += float(selected_row["loss"])
        if return_case_rows:
            case_rows.append(
                {
                    "case_id": int(case.get("case_id", case.get("sample_idx", -1))),
                    "row_idx": int(case.get("row_idx", -1)),
                    "row_weight": int(case.get("row_weight", -1)),
                    "support_bits": list(case.get("support_bits", ())),
                    "flipped_bits": list(case.get("flipped_bits", ())),
                    **selected_row,
                }
            )

    num_cases = max(1, len(cases))
    metrics = {
        "num_cases": int(len(cases)),
        "bank_size": int(bank_size),
        "selection_mode": str(selection_mode),
        "loss_mean": float(loss_sum / num_cases),
        "logical_success_rate": float(logical_success_sum / num_cases),
        "exact_recovery_rate": float(exact_recovery_sum / num_cases),
        "convergence_rate": float(convergence_sum / num_cases),
        "mean_iterations": float(iterations_sum / num_cases),
        "mean_logical_weight": float(logical_weight_sum / num_cases),
    }
    if return_case_rows:
        metrics["case_rows"] = case_rows
    return metrics


def evaluate_random_assignment_sampler(
    *,
    problem: CodeCapacityProblem,
    cases: Sequence[dict[str, Any]],
    family: str,
    draw_count: int,
    selection_mode: str,
    max_iter: int,
    alpha: float | None,
    logical_failure_penalty: float,
    logical_weight_penalty: float,
    convergence_penalty: float,
    iteration_penalty: float,
    base_seed: int,
    memory_distribution: ScalarDistributionSpec | None = None,
    damping_distribution: ScalarDistributionSpec | None = None,
    return_case_rows: bool = False,
) -> dict[str, Any]:
    family = _validate_family(family)
    selection_mode = _validate_selection_mode(selection_mode)
    if int(draw_count) <= 0:
        raise ValueError("draw_count must be positive.")
    _validate_distribution_requirements(
        family=family,
        memory_distribution=memory_distribution,
        damping_distribution=damping_distribution,
    )
    tracer = _make_tracer(problem=problem, family=family, max_iter=max_iter, alpha=alpha)
    logical_success_sum = 0.0
    exact_recovery_sum = 0.0
    convergence_sum = 0.0
    iterations_sum = 0.0
    logical_weight_sum = 0.0
    loss_sum = 0.0
    case_rows: list[dict[str, Any]] = []

    for case_idx, case in enumerate(cases):
        draw_rows: list[dict[str, Any]] = []
        case_id = int(case.get("case_id", case.get("sample_idx", case_idx)))
        for draw_idx in range(int(draw_count)):
            seed = int(base_seed) + 1_000_003 * int(case_id) + 10_007 * int(draw_idx)
            rng = np.random.default_rng(seed)
            memory_vector = (
                None
                if family == "damping"
                else _sample_distribution_vector(
                    size=int(problem.n_bits),
                    distribution=memory_distribution,
                    rng=rng,
                )
            )
            damping_vector = (
                None
                if family == "memory"
                else _sample_distribution_vector(
                    size=int(problem.hz.nnz),
                    distribution=damping_distribution,
                    rng=rng,
                )
            )
            draw_rows.append(
                _decode_assignment_member(
                    problem=problem,
                    tracer=tracer,
                    case=case,
                    family=family,
                    memory_vector=memory_vector,
                    damping_vector=damping_vector,
                    bank_idx=draw_idx,
                    max_iter=max_iter,
                    logical_failure_penalty=logical_failure_penalty,
                    logical_weight_penalty=logical_weight_penalty,
                    convergence_penalty=convergence_penalty,
                    iteration_penalty=iteration_penalty,
                )
            )
        selected_row = _select_assignment_row(draw_rows, selection_mode=selection_mode)
        logical_success_sum += float(selected_row["logical_success"])
        exact_recovery_sum += float(selected_row["exact_recovery"])
        convergence_sum += float(selected_row["converged"])
        iterations_sum += float(selected_row["iterations"])
        logical_weight_sum += float(selected_row["logical_weight"])
        loss_sum += float(selected_row["loss"])
        if return_case_rows:
            case_rows.append(
                {
                    "case_id": int(case_id),
                    "row_idx": int(case.get("row_idx", -1)),
                    "row_weight": int(case.get("row_weight", -1)),
                    "support_bits": list(case.get("support_bits", ())),
                    "flipped_bits": list(case.get("flipped_bits", ())),
                    **selected_row,
                }
            )

    num_cases = max(1, len(cases))
    metrics = {
        "num_cases": int(len(cases)),
        "draw_count": int(draw_count),
        "selection_mode": str(selection_mode),
        "loss_mean": float(loss_sum / num_cases),
        "logical_success_rate": float(logical_success_sum / num_cases),
        "exact_recovery_rate": float(exact_recovery_sum / num_cases),
        "convergence_rate": float(convergence_sum / num_cases),
        "mean_iterations": float(iterations_sum / num_cases),
        "mean_logical_weight": float(logical_weight_sum / num_cases),
    }
    if return_case_rows:
        metrics["case_rows"] = case_rows
    return metrics


def compare_bank_to_random_baselines(
    *,
    problem: CodeCapacityProblem,
    cases: Sequence[dict[str, Any]],
    family: str,
    memory_bank: np.ndarray | None,
    damping_bank: np.ndarray | None,
    memory_distribution: ScalarDistributionSpec | None = None,
    damping_distribution: ScalarDistributionSpec | None = None,
    max_iter: int,
    alpha: float | None,
    k_draw: int,
    logical_failure_penalty: float,
    logical_weight_penalty: float,
    convergence_penalty: float,
    iteration_penalty: float,
    bootstrap_samples: int,
    repeat_count: int,
    base_seed: int,
) -> dict[str, Any]:
    practical_bank = evaluate_assignment_bank(
        problem=problem,
        cases=cases,
        family=family,
        memory_bank=memory_bank,
        damping_bank=damping_bank,
        max_iter=max_iter,
        alpha=alpha,
        selection_mode="practical",
        logical_failure_penalty=logical_failure_penalty,
        logical_weight_penalty=logical_weight_penalty,
        convergence_penalty=convergence_penalty,
        iteration_penalty=iteration_penalty,
    )
    oracle_bank = evaluate_assignment_bank(
        problem=problem,
        cases=cases,
        family=family,
        memory_bank=memory_bank,
        damping_bank=damping_bank,
        max_iter=max_iter,
        alpha=alpha,
        selection_mode="oracle",
        logical_failure_penalty=logical_failure_penalty,
        logical_weight_penalty=logical_weight_penalty,
        convergence_penalty=convergence_penalty,
        iteration_penalty=iteration_penalty,
    )
    practical_candidate_rows = [practical_bank for _ in range(max(1, int(repeat_count)))]
    oracle_candidate_rows = [oracle_bank for _ in range(max(1, int(repeat_count)))]
    practical_baseline_rows = []
    oracle_baseline_rows = []
    for repeat_idx in range(max(1, int(repeat_count))):
        practical_baseline_rows.append(
            evaluate_random_assignment_sampler(
                problem=problem,
                cases=cases,
                family=family,
                draw_count=int(k_draw),
                selection_mode="practical",
                max_iter=max_iter,
                alpha=alpha,
                logical_failure_penalty=logical_failure_penalty,
                logical_weight_penalty=logical_weight_penalty,
                convergence_penalty=convergence_penalty,
                iteration_penalty=iteration_penalty,
                base_seed=int(base_seed) + 100_003 * repeat_idx,
                memory_distribution=memory_distribution,
                damping_distribution=damping_distribution,
            )
        )
        oracle_baseline_rows.append(
            evaluate_random_assignment_sampler(
                problem=problem,
                cases=cases,
                family=family,
                draw_count=int(k_draw),
                selection_mode="oracle",
                max_iter=max_iter,
                alpha=alpha,
                logical_failure_penalty=logical_failure_penalty,
                logical_weight_penalty=logical_weight_penalty,
                convergence_penalty=convergence_penalty,
                iteration_penalty=iteration_penalty,
                base_seed=int(base_seed) + 500_003 + 100_003 * repeat_idx,
                memory_distribution=memory_distribution,
                damping_distribution=damping_distribution,
            )
        )

    practical_difference = _bootstrap_difference_report(
        candidate_rows=practical_candidate_rows,
        baseline_rows=practical_baseline_rows,
        bootstrap_samples=bootstrap_samples,
        seed=int(base_seed) + 700_001,
    )
    oracle_difference = _bootstrap_difference_report(
        candidate_rows=oracle_candidate_rows,
        baseline_rows=oracle_baseline_rows,
        bootstrap_samples=bootstrap_samples,
        seed=int(base_seed) + 900_001,
    )
    return {
        "bank_practical": practical_bank,
        "bank_oracle": oracle_bank,
        "random_practical": _average_metric_rows(practical_baseline_rows),
        "random_oracle": _average_metric_rows(oracle_baseline_rows),
        "practical_difference": practical_difference,
        "oracle_difference": oracle_difference,
    }


def scan_memory_distribution_support(
    *,
    problem: CodeCapacityProblem,
    random_cases: Sequence[dict[str, Any]],
    half_cases: Sequence[dict[str, Any]],
    centers: Sequence[float],
    widths: Sequence[float],
    gaussian_refinement_points: int,
    draw_count: int,
    max_iter: int,
    alpha: float | None,
    logical_failure_penalty: float,
    logical_weight_penalty: float,
    convergence_penalty: float,
    iteration_penalty: float,
    selection_half_weight: float,
    base_seed: int,
) -> dict[str, Any]:
    return _scan_distribution_support(
        problem=problem,
        family="memory",
        random_cases=random_cases,
        half_cases=half_cases,
        centers=centers,
        widths=widths,
        gaussian_refinement_points=gaussian_refinement_points,
        draw_count=draw_count,
        max_iter=max_iter,
        alpha=alpha,
        logical_failure_penalty=logical_failure_penalty,
        logical_weight_penalty=logical_weight_penalty,
        convergence_penalty=convergence_penalty,
        iteration_penalty=iteration_penalty,
        selection_half_weight=selection_half_weight,
        base_seed=base_seed,
    )


def scan_damping_distribution_support(
    *,
    problem: CodeCapacityProblem,
    random_cases: Sequence[dict[str, Any]],
    half_cases: Sequence[dict[str, Any]],
    centers: Sequence[float],
    widths: Sequence[float],
    gaussian_refinement_points: int,
    draw_count: int,
    max_iter: int,
    alpha: float | None,
    logical_failure_penalty: float,
    logical_weight_penalty: float,
    convergence_penalty: float,
    iteration_penalty: float,
    selection_half_weight: float,
    base_seed: int,
) -> dict[str, Any]:
    return _scan_distribution_support(
        problem=problem,
        family="damping",
        random_cases=random_cases,
        half_cases=half_cases,
        centers=centers,
        widths=widths,
        gaussian_refinement_points=gaussian_refinement_points,
        draw_count=draw_count,
        max_iter=max_iter,
        alpha=alpha,
        logical_failure_penalty=logical_failure_penalty,
        logical_weight_penalty=logical_weight_penalty,
        convergence_penalty=convergence_penalty,
        iteration_penalty=iteration_penalty,
        selection_half_weight=selection_half_weight,
        base_seed=base_seed,
    )


def train_assignment_bank_cem(
    problem: CodeCapacityProblem,
    *,
    config: StaticAssignmentBankConfig,
    half_cases: Sequence[dict[str, Any]],
    random_train_cases: Sequence[dict[str, Any]],
    random_validation_cases: Sequence[dict[str, Any]],
    memory_support: dict[str, Any] | None = None,
    damping_support: dict[str, Any] | None = None,
    initial_memory_bank: np.ndarray | None = None,
    initial_damping_bank: np.ndarray | None = None,
) -> dict[str, Any]:
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_json(checkpoint_dir / "config.json", _config_payload(problem, config))
    initial_state = _initialize_distribution_state(
        problem=problem,
        config=config,
        memory_support=memory_support,
        damping_support=damping_support,
        half_cases=half_cases,
        random_cases=random_train_cases,
        initial_memory_bank=initial_memory_bank,
        initial_damping_bank=initial_damping_bank,
    )
    memory_mean_bank = initial_state["memory_mean_bank"]
    damping_mean_bank = initial_state["damping_mean_bank"]
    memory_sigma = float(initial_state["memory_sigma"])
    damping_sigma = float(initial_state["damping_sigma"])

    best_validation_row: dict[str, Any] | None = None
    best_memory_bank: np.ndarray | None = None
    best_damping_bank: np.ndarray | None = None
    rng = np.random.default_rng(int(config.base_seed))

    for round_idx in range(int(config.rounds)):
        batch_random_cases = _sample_random_batch(
            random_train_cases,
            batch_size=min(int(config.random_train_batch_size), len(random_train_cases)),
            rng=rng,
        )
        candidate_rows = []
        candidate_memory_banks: list[np.ndarray | None] = []
        candidate_damping_banks: list[np.ndarray | None] = []
        candidate_losses: list[float] = []
        for candidate_idx in range(int(config.candidate_count)):
            seed = int(config.base_seed) + 100_003 * int(round_idx) + int(candidate_idx)
            candidate_memory_bank, candidate_damping_bank = _sample_candidate_bank(
                problem=problem,
                config=config,
                memory_mean_bank=memory_mean_bank,
                damping_mean_bank=damping_mean_bank,
                memory_sigma=memory_sigma,
                damping_sigma=damping_sigma,
                seed=seed,
            )
            half_metrics = evaluate_assignment_bank(
                problem=problem,
                cases=half_cases,
                family=config.family,
                memory_bank=candidate_memory_bank,
                damping_bank=candidate_damping_bank,
                max_iter=config.max_iter,
                alpha=config.alpha,
                selection_mode="practical",
                logical_failure_penalty=config.logical_failure_penalty,
                logical_weight_penalty=config.logical_weight_penalty,
                convergence_penalty=config.convergence_penalty,
                iteration_penalty=config.iteration_penalty,
            )
            random_metrics = evaluate_assignment_bank(
                problem=problem,
                cases=batch_random_cases,
                family=config.family,
                memory_bank=candidate_memory_bank,
                damping_bank=candidate_damping_bank,
                max_iter=config.max_iter,
                alpha=config.alpha,
                selection_mode="practical",
                logical_failure_penalty=config.logical_failure_penalty,
                logical_weight_penalty=config.logical_weight_penalty,
                convergence_penalty=config.convergence_penalty,
                iteration_penalty=config.iteration_penalty,
            )
            diversity_penalty = _bank_diversity_penalty(
                family=config.family,
                memory_bank=candidate_memory_bank,
                damping_bank=candidate_damping_bank,
                config=config,
            )
            selection_loss = float(
                random_metrics["loss_mean"]
                + float(config.selection_half_weight) * half_metrics["loss_mean"]
                + float(config.diversity_penalty) * diversity_penalty
            )
            row = {
                "round": int(round_idx + 1),
                "candidate_idx": int(candidate_idx),
                "selection_loss": float(selection_loss),
                "diversity_penalty": float(diversity_penalty),
                "random_batch_logical_success_rate": float(random_metrics["logical_success_rate"]),
                "random_batch_convergence_rate": float(random_metrics["convergence_rate"]),
                "random_batch_mean_iterations": float(random_metrics["mean_iterations"]),
                "half_logical_success_rate": float(half_metrics["logical_success_rate"]),
                "half_convergence_rate": float(half_metrics["convergence_rate"]),
                "half_mean_iterations": float(half_metrics["mean_iterations"]),
            }
            candidate_rows.append(row)
            candidate_memory_banks.append(candidate_memory_bank)
            candidate_damping_banks.append(candidate_damping_bank)
            candidate_losses.append(selection_loss)

        elite_indices = np.argsort(np.asarray(candidate_losses, dtype=np.float64))[
            : max(1, min(int(config.elite_count), len(candidate_losses)))
        ]
        if memory_mean_bank is not None:
            elite_memory = np.asarray(
                [candidate_memory_banks[int(index)] for index in elite_indices if candidate_memory_banks[int(index)] is not None],
                dtype=np.float64,
            )
            if elite_memory.size:
                elite_losses = [candidate_losses[int(index)] for index in elite_indices]
                memory_mean_bank = _weighted_elite_average(elite_memory, elite_losses)
                memory_sigma = _clip_sigma(
                    value=float(np.sqrt(np.mean((elite_memory - memory_mean_bank[None, :, :]) ** 2))),
                    low=float(config.sigma_floor),
                    high=float(config.memory_sigma_ceiling) if config.memory_sigma_ceiling is not None else None,
                )
        if damping_mean_bank is not None:
            elite_damping = np.asarray(
                [candidate_damping_banks[int(index)] for index in elite_indices if candidate_damping_banks[int(index)] is not None],
                dtype=np.float64,
            )
            if elite_damping.size:
                elite_losses = [candidate_losses[int(index)] for index in elite_indices]
                damping_mean_bank = _weighted_elite_average(elite_damping, elite_losses)
                damping_sigma = _clip_sigma(
                    value=float(np.sqrt(np.mean((elite_damping - damping_mean_bank[None, :, :]) ** 2))),
                    low=float(config.sigma_floor),
                    high=float(config.damping_sigma_ceiling) if config.damping_sigma_ceiling is not None else None,
                )

        representative_memory_bank = None if memory_mean_bank is None else _clip_array(
            memory_mean_bank,
            low=float(config.memory_min),
            high=float(config.memory_max),
        )
        representative_damping_bank = None if damping_mean_bank is None else _clip_array(
            damping_mean_bank,
            low=float(config.damping_min),
            high=float(config.damping_max),
        )
        _append_csv_row(
            checkpoint_dir / "train_metrics.csv",
            {
                "round": int(round_idx + 1),
                "candidate_loss_min": float(np.min(candidate_losses)),
                "candidate_loss_mean": float(np.mean(candidate_losses)),
                "candidate_loss_max": float(np.max(candidate_losses)),
                "memory_sigma": float(memory_sigma),
                "damping_sigma": float(damping_sigma),
                "elite_count": int(len(elite_indices)),
            },
        )

        if ((round_idx + 1) % int(config.validation_interval) == 0) or (round_idx + 1 == int(config.rounds)):
            validation_row = _evaluate_bank_validation(
                problem=problem,
                config=config,
                half_cases=half_cases,
                random_validation_cases=random_validation_cases,
                memory_bank=representative_memory_bank,
                damping_bank=representative_damping_bank,
                round_idx=round_idx,
            )
            _append_csv_row(checkpoint_dir / "validation_metrics.csv", validation_row)
            _save_bank_checkpoint(
                checkpoint_dir=checkpoint_dir,
                memory_bank=representative_memory_bank,
                damping_bank=representative_damping_bank,
                memory_sigma=memory_sigma,
                damping_sigma=damping_sigma,
                completed_round=int(round_idx + 1),
            )
            if _is_better_bank_validation(validation_row, best_validation_row):
                best_validation_row = dict(validation_row)
                best_memory_bank = None if representative_memory_bank is None else np.asarray(
                    representative_memory_bank,
                    dtype=np.float64,
                )
                best_damping_bank = None if representative_damping_bank is None else np.asarray(
                    representative_damping_bank,
                    dtype=np.float64,
                )
                if best_memory_bank is not None:
                    np.save(checkpoint_dir / "best_memory_bank.npy", best_memory_bank)
                if best_damping_bank is not None:
                    np.save(checkpoint_dir / "best_damping_bank.npy", best_damping_bank)
                _write_json(checkpoint_dir / "best_validation.json", best_validation_row)

    if best_memory_bank is None and representative_memory_bank is not None:
        best_memory_bank = np.asarray(representative_memory_bank, dtype=np.float64)
    if best_damping_bank is None and representative_damping_bank is not None:
        best_damping_bank = np.asarray(representative_damping_bank, dtype=np.float64)
    summary = {
        "code_path": str(problem.path),
        "checkpoint_dir": str(checkpoint_dir),
        "family": str(config.family),
        "bank_size": int(config.bank_size),
        "completed_rounds": int(config.rounds),
        "best_memory_bank_path": None if best_memory_bank is None else str(checkpoint_dir / "best_memory_bank.npy"),
        "best_damping_bank_path": None if best_damping_bank is None else str(checkpoint_dir / "best_damping_bank.npy"),
        "train_metrics_csv": str(checkpoint_dir / "train_metrics.csv"),
        "validation_metrics_csv": str(checkpoint_dir / "validation_metrics.csv"),
        "best_validation": best_validation_row,
    }
    _write_json(checkpoint_dir / "training_summary.json", summary)
    return {
        **summary,
        "best_memory_bank": best_memory_bank,
        "best_damping_bank": best_damping_bank,
    }


def _decode_assignment_member(
    *,
    problem: CodeCapacityProblem,
    tracer,
    case: dict[str, Any],
    family: str,
    memory_vector: np.ndarray | None,
    damping_vector: np.ndarray | None,
    bank_idx: int,
    max_iter: int,
    logical_failure_penalty: float,
    logical_weight_penalty: float,
    convergence_penalty: float,
    iteration_penalty: float,
) -> dict[str, Any]:
    if family == "memory":
        result = decode_with_memory_strengths(tracer, case["syndrome"], np.asarray(memory_vector, dtype=np.float64))
    elif family == "damping":
        result = decode_with_edge_damping(
            tracer,
            case["syndrome"],
            np.asarray(damping_vector, dtype=np.float64),
            n_bits=problem.n_bits,
        )
    else:
        result = decode_with_memory_and_edge_damping(
            tracer,
            case["syndrome"],
            n_bits=problem.n_bits,
            memory_strengths=np.asarray(memory_vector, dtype=np.float64),
            edge_damping_messages=np.asarray(damping_vector, dtype=np.float64),
        )
    evaluation = evaluate_decode_result(result=result, error=case["error"], lz=problem.lz)
    residual_syndrome_weight = _residual_syndrome_weight(
        problem=problem,
        syndrome=np.asarray(case["syndrome"], dtype=np.uint8),
        decoding=np.asarray(result.decoding, dtype=np.uint8),
    )
    return {
        "bank_idx": int(bank_idx),
        "converged": bool(evaluation.converged),
        "logical_success": bool(evaluation.logical_success),
        "exact_recovery": bool(evaluation.exact_recovery),
        "logical_weight": int(evaluation.logical_weight),
        "iterations": int(evaluation.iterations),
        "decoding_weight": int(np.asarray(result.decoding, dtype=np.uint8).sum()),
        "residual_syndrome_weight": int(residual_syndrome_weight),
        "loss": float(
            _assignment_case_loss(
                logical_success=evaluation.logical_success,
                logical_weight=evaluation.logical_weight,
                converged=evaluation.converged,
                iterations=evaluation.iterations,
                max_iter=max_iter,
                logical_failure_penalty=logical_failure_penalty,
                logical_weight_penalty=logical_weight_penalty,
                convergence_penalty=convergence_penalty,
                iteration_penalty=iteration_penalty,
            )
        ),
    }


def _assignment_case_loss(
    *,
    logical_success: bool,
    logical_weight: int,
    converged: bool,
    iterations: int,
    max_iter: int,
    logical_failure_penalty: float,
    logical_weight_penalty: float,
    convergence_penalty: float,
    iteration_penalty: float,
) -> float:
    return float(
        float(logical_failure_penalty) * float(not logical_success)
        + float(logical_weight_penalty) * float(logical_weight)
        + float(convergence_penalty) * float(not converged)
        + float(iteration_penalty) * (float(iterations) / max(1, int(max_iter)))
    )


def _select_assignment_row(rows: Sequence[dict[str, Any]], *, selection_mode: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty.")
    if selection_mode == "practical":
        return dict(
            min(
                rows,
                key=lambda row: (
                    int(row["residual_syndrome_weight"]),
                    not bool(row["converged"]),
                    int(row["iterations"]),
                    int(row["decoding_weight"]),
                    int(row["bank_idx"]),
                ),
            )
        )
    if selection_mode == "oracle":
        return dict(
            min(
                rows,
                key=lambda row: (
                    not bool(row["logical_success"]),
                    int(row["logical_weight"]),
                    int(row["iterations"]),
                    int(row["bank_idx"]),
                ),
            )
        )
    raise ValueError(f"Unsupported selection_mode: {selection_mode!r}")


def _residual_syndrome_weight(
    *,
    problem: CodeCapacityProblem,
    syndrome: np.ndarray,
    decoding: np.ndarray,
) -> int:
    decoded_syndrome = np.asarray((problem.hz @ decoding) % 2, dtype=np.uint8).reshape(-1)
    residual = np.bitwise_xor(decoded_syndrome, np.asarray(syndrome, dtype=np.uint8))
    return int(residual.sum())


def _validate_family(family: str) -> str:
    family = str(family)
    if family not in FAMILY_CHOICES:
        raise ValueError(f"Unsupported family: {family!r}")
    return family


def _validate_selection_mode(selection_mode: str) -> str:
    selection_mode = str(selection_mode)
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"Unsupported selection_mode: {selection_mode!r}")
    return selection_mode


def _validate_distribution_requirements(
    *,
    family: str,
    memory_distribution: ScalarDistributionSpec | None,
    damping_distribution: ScalarDistributionSpec | None,
) -> None:
    if family in {"memory", "joint"} and memory_distribution is None:
        raise ValueError(f"{family} requires memory_distribution.")
    if family in {"damping", "joint"} and damping_distribution is None:
        raise ValueError(f"{family} requires damping_distribution.")


def _validate_bank_shapes(
    *,
    problem: CodeCapacityProblem,
    family: str,
    memory_bank: np.ndarray | None,
    damping_bank: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    validated_memory_bank = None if memory_bank is None else np.asarray(memory_bank, dtype=np.float64)
    validated_damping_bank = None if damping_bank is None else np.asarray(damping_bank, dtype=np.float64)
    if family in {"memory", "joint"}:
        if validated_memory_bank is None:
            raise ValueError(f"{family} requires memory_bank.")
        if validated_memory_bank.ndim != 2 or validated_memory_bank.shape[1] != problem.n_bits:
            raise ValueError(
                f"memory_bank shape {validated_memory_bank.shape} does not match (*, {problem.n_bits})."
            )
    if family in {"damping", "joint"}:
        if validated_damping_bank is None:
            raise ValueError(f"{family} requires damping_bank.")
        if validated_damping_bank.ndim != 2 or validated_damping_bank.shape[1] != problem.hz.nnz:
            raise ValueError(
                f"damping_bank shape {validated_damping_bank.shape} does not match (*, {problem.hz.nnz})."
            )
    if family == "memory":
        validated_damping_bank = None
    if family == "damping":
        validated_memory_bank = None
    if validated_memory_bank is not None and validated_damping_bank is not None:
        if validated_memory_bank.shape[0] != validated_damping_bank.shape[0]:
            raise ValueError("memory_bank and damping_bank must share the same bank dimension.")
    return validated_memory_bank, validated_damping_bank


def _bank_size(memory_bank: np.ndarray | None, damping_bank: np.ndarray | None) -> int:
    if memory_bank is not None:
        return int(memory_bank.shape[0])
    if damping_bank is not None:
        return int(damping_bank.shape[0])
    raise ValueError("At least one bank must be present.")


def _make_tracer(
    *,
    problem: CodeCapacityProblem,
    family: str,
    max_iter: int,
    alpha: float | None,
):
    return make_min_sum_tracer(
        problem,
        max_iter=max_iter,
        alpha=alpha,
        gamma0=0.0 if family in {"memory", "joint"} else None,
    )


def _sample_distribution_vector(
    *,
    size: int,
    distribution: ScalarDistributionSpec | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if distribution is None:
        raise ValueError("distribution must not be None.")
    if distribution.sampling_family == "uniform_interval":
        if np.isclose(float(distribution.low), float(distribution.high)):
            return np.full(int(size), float(distribution.low), dtype=np.float64)
        return rng.uniform(float(distribution.low), float(distribution.high), size=int(size)).astype(np.float64)
    assert distribution.mean is not None
    sigma = max(0.0, float(distribution.sigma or 0.0))
    if sigma == 0.0:
        return np.full(
            int(size),
            float(np.clip(float(distribution.mean), float(distribution.low), float(distribution.high))),
            dtype=np.float64,
        )
    sampled = rng.normal(loc=float(distribution.mean), scale=sigma, size=int(size))
    return np.clip(sampled, float(distribution.low), float(distribution.high)).astype(np.float64)


def _scan_distribution_support(
    *,
    problem: CodeCapacityProblem,
    family: str,
    random_cases: Sequence[dict[str, Any]],
    half_cases: Sequence[dict[str, Any]],
    centers: Sequence[float],
    widths: Sequence[float],
    gaussian_refinement_points: int,
    draw_count: int,
    max_iter: int,
    alpha: float | None,
    logical_failure_penalty: float,
    logical_weight_penalty: float,
    convergence_penalty: float,
    iteration_penalty: float,
    selection_half_weight: float,
    base_seed: int,
) -> dict[str, Any]:
    bounds = _family_bounds(family)
    uniform_rows = []
    gaussian_support_rows = []
    for width_idx, width in enumerate(widths):
        for center_idx, center in enumerate(centers):
            low, high = _bounded_interval_from_center_width(
                center=float(center),
                width=float(width),
                low_bound=float(bounds[0]),
                high_bound=float(bounds[1]),
            )
            uniform_spec = ScalarDistributionSpec(
                sampling_family="uniform_interval",
                low=low,
                high=high,
            )
            gaussian_spec = ScalarDistributionSpec(
                sampling_family="truncated_gaussian",
                low=low,
                high=high,
                mean=float(np.clip(float(center), low, high)),
                sigma=max((high - low) / 4.0, 1e-5),
            )
            uniform_rows.append(
                _support_row_from_distribution(
                    problem=problem,
                    family=family,
                    random_cases=random_cases,
                    half_cases=half_cases,
                    draw_count=draw_count,
                    max_iter=max_iter,
                    alpha=alpha,
                    logical_failure_penalty=logical_failure_penalty,
                    logical_weight_penalty=logical_weight_penalty,
                    convergence_penalty=convergence_penalty,
                    iteration_penalty=iteration_penalty,
                    selection_half_weight=selection_half_weight,
                    base_seed=int(base_seed) + 1_000_003 * width_idx + 10_007 * center_idx,
                    memory_distribution=uniform_spec if family == "memory" else None,
                    damping_distribution=uniform_spec if family == "damping" else None,
                    extra={"center": float(center), "width": float(width), "variance": 0.0},
                )
            )
            gaussian_support_rows.append(
                _support_row_from_distribution(
                    problem=problem,
                    family=family,
                    random_cases=random_cases,
                    half_cases=half_cases,
                    draw_count=draw_count,
                    max_iter=max_iter,
                    alpha=alpha,
                    logical_failure_penalty=logical_failure_penalty,
                    logical_weight_penalty=logical_weight_penalty,
                    convergence_penalty=convergence_penalty,
                    iteration_penalty=iteration_penalty,
                    selection_half_weight=selection_half_weight,
                    base_seed=int(base_seed) + 5_000_003 + 1_000_003 * width_idx + 10_007 * center_idx,
                    memory_distribution=gaussian_spec if family == "memory" else None,
                    damping_distribution=gaussian_spec if family == "damping" else None,
                    extra={
                        "center": float(center),
                        "width": float(width),
                        "mean": float(np.clip(float(center), low, high)),
                        "sigma": float(gaussian_spec.sigma or 0.0),
                        "variance": float((gaussian_spec.sigma or 0.0) ** 2),
                    },
                )
            )
    best_uniform = _best_support_row(uniform_rows)
    best_gaussian_support = _best_support_row(gaussian_support_rows)
    refinement_rows = []
    refine_low = float(best_gaussian_support["low"])
    refine_high = float(best_gaussian_support["high"])
    support_width = max(0.0, refine_high - refine_low)
    sigma_values = np.linspace(max(support_width / 16.0, 1e-5), max(support_width / 2.0, 1e-5), int(gaussian_refinement_points))
    mean_values = np.linspace(refine_low, refine_high, int(gaussian_refinement_points))
    for sigma_idx, sigma in enumerate(sigma_values):
        for mean_idx, mean in enumerate(mean_values):
            gaussian_spec = ScalarDistributionSpec(
                sampling_family="truncated_gaussian",
                low=refine_low,
                high=refine_high,
                mean=float(mean),
                sigma=float(sigma),
            )
            refinement_rows.append(
                _support_row_from_distribution(
                    problem=problem,
                    family=family,
                    random_cases=random_cases,
                    half_cases=half_cases,
                    draw_count=draw_count,
                    max_iter=max_iter,
                    alpha=alpha,
                    logical_failure_penalty=logical_failure_penalty,
                    logical_weight_penalty=logical_weight_penalty,
                    convergence_penalty=convergence_penalty,
                    iteration_penalty=iteration_penalty,
                    selection_half_weight=selection_half_weight,
                    base_seed=int(base_seed) + 9_000_003 + 100_003 * sigma_idx + 1_003 * mean_idx,
                    memory_distribution=gaussian_spec if family == "memory" else None,
                    damping_distribution=gaussian_spec if family == "damping" else None,
                    extra={
                        "mean": float(mean),
                        "sigma": float(sigma),
                        "variance": float(sigma**2),
                    },
                )
            )
    best_gaussian = _best_support_row(refinement_rows)
    best_overall = best_uniform if _support_row_better(best_uniform, best_gaussian) else best_gaussian
    return {
        "family": family,
        "draw_count": int(draw_count),
        "uniform_rows": uniform_rows,
        "gaussian_support_rows": gaussian_support_rows,
        "gaussian_refinement_rows": refinement_rows,
        "best_uniform": best_uniform,
        "best_gaussian_support": best_gaussian_support,
        "best_gaussian": best_gaussian,
        "best_overall": best_overall,
    }


def _support_row_from_distribution(
    *,
    problem: CodeCapacityProblem,
    family: str,
    random_cases: Sequence[dict[str, Any]],
    half_cases: Sequence[dict[str, Any]],
    draw_count: int,
    max_iter: int,
    alpha: float | None,
    logical_failure_penalty: float,
    logical_weight_penalty: float,
    convergence_penalty: float,
    iteration_penalty: float,
    selection_half_weight: float,
    base_seed: int,
    memory_distribution: ScalarDistributionSpec | None,
    damping_distribution: ScalarDistributionSpec | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    random_metrics = evaluate_random_assignment_sampler(
        problem=problem,
        cases=random_cases,
        family=family,
        draw_count=draw_count,
        selection_mode="practical",
        max_iter=max_iter,
        alpha=alpha,
        logical_failure_penalty=logical_failure_penalty,
        logical_weight_penalty=logical_weight_penalty,
        convergence_penalty=convergence_penalty,
        iteration_penalty=iteration_penalty,
        base_seed=base_seed,
        memory_distribution=memory_distribution,
        damping_distribution=damping_distribution,
    )
    half_metrics = evaluate_random_assignment_sampler(
        problem=problem,
        cases=half_cases,
        family=family,
        draw_count=draw_count,
        selection_mode="practical",
        max_iter=max_iter,
        alpha=alpha,
        logical_failure_penalty=logical_failure_penalty,
        logical_weight_penalty=logical_weight_penalty,
        convergence_penalty=convergence_penalty,
        iteration_penalty=iteration_penalty,
        base_seed=int(base_seed) + 500_001,
        memory_distribution=memory_distribution,
        damping_distribution=damping_distribution,
    )
    selection_loss = float(
        random_metrics["loss_mean"] + float(selection_half_weight) * half_metrics["loss_mean"]
    )
    row = {
        "selection_loss": selection_loss,
        "low": float(memory_distribution.low if memory_distribution is not None else damping_distribution.low),
        "high": float(memory_distribution.high if memory_distribution is not None else damping_distribution.high),
        "random_logical_success_rate": float(random_metrics["logical_success_rate"]),
        "random_convergence_rate": float(random_metrics["convergence_rate"]),
        "random_exact_recovery_rate": float(random_metrics["exact_recovery_rate"]),
        "random_mean_iterations": float(random_metrics["mean_iterations"]),
        "half_logical_success_rate": float(half_metrics["logical_success_rate"]),
        "half_convergence_rate": float(half_metrics["convergence_rate"]),
        "half_exact_recovery_rate": float(half_metrics["exact_recovery_rate"]),
        "half_mean_iterations": float(half_metrics["mean_iterations"]),
    }
    row.update(extra)
    return row


def _best_support_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty.")
    best = dict(rows[0])
    for row in rows[1:]:
        if _support_row_better(row, best):
            best = dict(row)
    return best


def _support_row_better(candidate: dict[str, Any], current_best: dict[str, Any]) -> bool:
    tol = 1e-12
    if float(candidate["selection_loss"]) < float(current_best["selection_loss"]) - tol:
        return True
    if float(candidate["selection_loss"]) > float(current_best["selection_loss"]) + tol:
        return False
    candidate_key = (
        float(candidate["random_logical_success_rate"]),
        float(candidate["half_logical_success_rate"]),
        float(candidate["random_convergence_rate"]),
        float(candidate["half_convergence_rate"]),
        -float(candidate["random_mean_iterations"]),
        -float(candidate["half_mean_iterations"]),
    )
    current_key = (
        float(current_best["random_logical_success_rate"]),
        float(current_best["half_logical_success_rate"]),
        float(current_best["random_convergence_rate"]),
        float(current_best["half_convergence_rate"]),
        -float(current_best["random_mean_iterations"]),
        -float(current_best["half_mean_iterations"]),
    )
    return candidate_key > current_key


def _family_bounds(family: str) -> tuple[float, float]:
    if family == "memory":
        return (-0.3, 0.3)
    if family == "damping":
        return (0.0, 1.0)
    raise ValueError(f"Unsupported family: {family!r}")


def _bounded_interval_from_center_width(
    *,
    center: float,
    width: float,
    low_bound: float,
    high_bound: float,
) -> tuple[float, float]:
    half_width = max(0.0, float(width) / 2.0)
    low = max(float(low_bound), float(center) - half_width)
    high = min(float(high_bound), float(center) + half_width)
    if high < low:
        low, high = high, low
    return (low, high)


def _initialize_distribution_state(
    *,
    problem: CodeCapacityProblem,
    config: StaticAssignmentBankConfig,
    memory_support: dict[str, Any] | None,
    damping_support: dict[str, Any] | None,
    half_cases: Sequence[dict[str, Any]],
    random_cases: Sequence[dict[str, Any]],
    initial_memory_bank: np.ndarray | None,
    initial_damping_bank: np.ndarray | None,
) -> dict[str, Any]:
    memory_mean_bank = None
    damping_mean_bank = None
    memory_sigma = 0.0
    damping_sigma = 0.0
    initialization_summary: dict[str, Any] = {
        "family": str(config.family),
        "memory_support": memory_support,
        "damping_support": damping_support,
        "bank_size": int(config.bank_size),
    }
    if config.family in {"memory", "joint"}:
        memory_low = float(
            config.memory_init_low if config.memory_init_low is not None else _support_low(memory_support, default=config.memory_min)
        )
        memory_high = float(
            config.memory_init_high if config.memory_init_high is not None else _support_high(memory_support, default=config.memory_max)
        )
        if initial_memory_bank is not None:
            memory_mean_bank = _clip_array(
                np.asarray(initial_memory_bank, dtype=np.float64),
                low=float(config.memory_min),
                high=float(config.memory_max),
            )
            if memory_mean_bank.shape != (int(config.bank_size), int(problem.n_bits)):
                raise ValueError(
                    "initial_memory_bank shape "
                    f"{memory_mean_bank.shape} does not match {(int(config.bank_size), int(problem.n_bits))}."
                )
            initialization_summary["memory_initialization"] = {
                "candidate_count": 1,
                "low": float(memory_low),
                "high": float(memory_high),
                "provided_initial_bank": True,
            }
        else:
            init_candidates = []
            for candidate_idx in range(max(1, int(config.initialization_candidates))):
                seed_rng = np.random.default_rng(int(config.base_seed) + 10_007 * candidate_idx)
                init_candidates.append(
                    _sample_initial_candidate_bank(
                        size=(int(config.bank_size), int(problem.n_bits)),
                        low=memory_low,
                        high=memory_high,
                        support=memory_support,
                        rng=seed_rng,
                    )
                )
            memory_mean_bank = _choose_best_initial_bank(
                problem=problem,
                config=config,
                family="memory",
                candidate_banks=init_candidates,
                half_cases=half_cases,
                random_cases=random_cases,
            )
            initialization_summary["memory_initialization"] = {
                "candidate_count": int(len(init_candidates)),
                "low": float(memory_low),
                "high": float(memory_high),
                "provided_initial_bank": False,
            }
        default_sigma = max((memory_high - memory_low) / 4.0, float(config.sigma_floor))
        memory_sigma = _clip_sigma(
            value=float(
                _support_sigma(
                    memory_support,
                    default=default_sigma,
                    override=config.initial_memory_sigma,
                )
            ),
            low=float(config.sigma_floor),
            high=float(config.memory_sigma_ceiling) if config.memory_sigma_ceiling is not None else None,
        )
        initialization_summary["memory_initialization"]["sigma"] = float(memory_sigma)
    if config.family in {"damping", "joint"}:
        damping_low = float(
            config.damping_init_low if config.damping_init_low is not None else _support_low(damping_support, default=config.damping_min)
        )
        damping_high = float(
            config.damping_init_high if config.damping_init_high is not None else _support_high(damping_support, default=config.damping_max)
        )
        if initial_damping_bank is not None:
            damping_mean_bank = _clip_array(
                np.asarray(initial_damping_bank, dtype=np.float64),
                low=float(config.damping_min),
                high=float(config.damping_max),
            )
            if damping_mean_bank.shape != (int(config.bank_size), int(problem.hz.nnz)):
                raise ValueError(
                    "initial_damping_bank shape "
                    f"{damping_mean_bank.shape} does not match {(int(config.bank_size), int(problem.hz.nnz))}."
                )
            initialization_summary["damping_initialization"] = {
                "candidate_count": 1,
                "low": float(damping_low),
                "high": float(damping_high),
                "provided_initial_bank": True,
            }
        else:
            init_candidates = []
            for candidate_idx in range(max(1, int(config.initialization_candidates))):
                seed_rng = np.random.default_rng(int(config.base_seed) + 50_007 * candidate_idx)
                init_candidates.append(
                    _sample_initial_candidate_bank(
                        size=(int(config.bank_size), int(problem.hz.nnz)),
                        low=damping_low,
                        high=damping_high,
                        support=damping_support,
                        rng=seed_rng,
                    )
                )
            damping_mean_bank = _choose_best_initial_bank(
                problem=problem,
                config=config,
                family="damping",
                candidate_banks=init_candidates,
                half_cases=half_cases,
                random_cases=random_cases,
            )
            initialization_summary["damping_initialization"] = {
                "candidate_count": int(len(init_candidates)),
                "low": float(damping_low),
                "high": float(damping_high),
                "provided_initial_bank": False,
            }
        default_sigma = max((damping_high - damping_low) / 4.0, float(config.sigma_floor))
        damping_sigma = _clip_sigma(
            value=float(
                _support_sigma(
                    damping_support,
                    default=default_sigma,
                    override=config.initial_damping_sigma,
                )
            ),
            low=float(config.sigma_floor),
            high=float(config.damping_sigma_ceiling) if config.damping_sigma_ceiling is not None else None,
        )
        initialization_summary["damping_initialization"]["sigma"] = float(damping_sigma)
    initialization_summary["memory_sigma"] = float(memory_sigma)
    initialization_summary["damping_sigma"] = float(damping_sigma)
    _write_json(Path(config.checkpoint_dir) / "initialization_summary.json", initialization_summary)
    return {
        "memory_mean_bank": None if memory_mean_bank is None else np.asarray(memory_mean_bank, dtype=np.float64),
        "damping_mean_bank": None if damping_mean_bank is None else np.asarray(damping_mean_bank, dtype=np.float64),
        "memory_sigma": float(memory_sigma),
        "damping_sigma": float(damping_sigma),
    }


def _choose_best_initial_bank(
    *,
    problem: CodeCapacityProblem,
    config: StaticAssignmentBankConfig,
    family: str,
    candidate_banks: Sequence[np.ndarray],
    half_cases: Sequence[dict[str, Any]],
    random_cases: Sequence[dict[str, Any]],
) -> np.ndarray:
    if not candidate_banks:
        raise ValueError("candidate_banks must not be empty.")
    half_subset = list(half_cases[: min(len(half_cases), 32)])
    random_subset = list(random_cases[: min(len(random_cases), 128)])
    if not half_subset and not random_subset:
        return np.asarray(candidate_banks[0], dtype=np.float64)
    scored_candidates: list[tuple[float, np.ndarray]] = []
    for candidate_bank in candidate_banks:
        memory_bank = np.asarray(candidate_bank, dtype=np.float64) if family == "memory" else None
        damping_bank = np.asarray(candidate_bank, dtype=np.float64) if family == "damping" else None
        loss_terms = []
        if random_subset:
            random_metrics = evaluate_assignment_bank(
                problem=problem,
                cases=random_subset,
                family=family,
                memory_bank=memory_bank,
                damping_bank=damping_bank,
                max_iter=config.max_iter,
                alpha=config.alpha,
                selection_mode="practical",
                logical_failure_penalty=config.logical_failure_penalty,
                logical_weight_penalty=config.logical_weight_penalty,
                convergence_penalty=config.convergence_penalty,
                iteration_penalty=config.iteration_penalty,
            )
            loss_terms.append(float(random_metrics["loss_mean"]))
        if half_subset:
            half_metrics = evaluate_assignment_bank(
                problem=problem,
                cases=half_subset,
                family=family,
                memory_bank=memory_bank,
                damping_bank=damping_bank,
                max_iter=config.max_iter,
                alpha=config.alpha,
                selection_mode="practical",
                logical_failure_penalty=config.logical_failure_penalty,
                logical_weight_penalty=config.logical_weight_penalty,
                convergence_penalty=config.convergence_penalty,
                iteration_penalty=config.iteration_penalty,
            )
            loss_terms.append(float(config.selection_half_weight) * float(half_metrics["loss_mean"]))
        scored_candidates.append((float(np.sum(loss_terms)), np.asarray(candidate_bank, dtype=np.float64)))
    scored_candidates.sort(key=lambda item: item[0])
    topk = min(max(1, int(config.elite_count)), len(scored_candidates))
    top_banks = np.asarray([bank for _, bank in scored_candidates[:topk]], dtype=np.float64)
    top_losses = [loss for loss, _ in scored_candidates[:topk]]
    return _weighted_elite_average(top_banks, top_losses)


def _sample_initial_candidate_bank(
    *,
    size: tuple[int, int],
    low: float,
    high: float,
    support: dict[str, Any] | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if support is not None and "mean" in support:
        sigma = float(_support_sigma(support, default=max((high - low) / 4.0, 1e-6)))
        sampled = rng.normal(loc=float(support["mean"]), scale=sigma, size=size)
        return _clip_array(sampled, low=low, high=high)
    return rng.uniform(low, high, size=size).astype(np.float64)


def _sample_candidate_bank(
    *,
    problem: CodeCapacityProblem,
    config: StaticAssignmentBankConfig,
    memory_mean_bank: np.ndarray | None,
    damping_mean_bank: np.ndarray | None,
    memory_sigma: float,
    damping_sigma: float,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    rng = np.random.default_rng(int(seed))
    sampled_memory_bank = None
    sampled_damping_bank = None
    if memory_mean_bank is not None:
        sampled_memory_bank = _sample_bank_from_mean(
            mean_bank=memory_mean_bank,
            sigma=float(memory_sigma),
            low=float(config.memory_min),
            high=float(config.memory_max),
            rng=rng,
        )
    if damping_mean_bank is not None:
        sampled_damping_bank = _sample_bank_from_mean(
            mean_bank=damping_mean_bank,
            sigma=float(damping_sigma),
            low=float(config.damping_min),
            high=float(config.damping_max),
            rng=rng,
        )
    return sampled_memory_bank, sampled_damping_bank


def _sample_bank_from_mean(
    *,
    mean_bank: np.ndarray,
    sigma: float,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if sigma <= 0.0:
        return _clip_array(mean_bank, low=low, high=high)
    sampled = rng.normal(loc=np.asarray(mean_bank, dtype=np.float64), scale=float(sigma))
    return _clip_array(sampled, low=low, high=high)


def _sample_random_batch(
    cases: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if not cases:
        return []
    if batch_size <= 0:
        return []
    indices = rng.integers(0, len(cases), size=int(batch_size))
    return [cases[int(index)] for index in indices]


def _bank_diversity_penalty(
    *,
    family: str,
    memory_bank: np.ndarray | None,
    damping_bank: np.ndarray | None,
    config: StaticAssignmentBankConfig,
) -> float:
    penalties = []
    if family in {"memory", "joint"} and memory_bank is not None and memory_bank.shape[0] > 1:
        penalties.append(
            _duplicate_member_penalty(
                bank=memory_bank,
                span=max(float(config.memory_max) - float(config.memory_min), 1e-6),
                threshold_fraction=float(config.diversity_threshold_fraction),
            )
        )
    if family in {"damping", "joint"} and damping_bank is not None and damping_bank.shape[0] > 1:
        penalties.append(
            _duplicate_member_penalty(
                bank=damping_bank,
                span=max(float(config.damping_max) - float(config.damping_min), 1e-6),
                threshold_fraction=float(config.diversity_threshold_fraction),
            )
        )
    if not penalties:
        return 0.0
    return float(np.mean(np.asarray(penalties, dtype=np.float64)))


def _duplicate_member_penalty(
    *,
    bank: np.ndarray,
    span: float,
    threshold_fraction: float,
) -> float:
    threshold = max(float(threshold_fraction) * float(span), 1e-9)
    penalties = []
    for first_idx in range(bank.shape[0]):
        for second_idx in range(first_idx + 1, bank.shape[0]):
            diff = float(np.mean(np.abs(bank[first_idx] - bank[second_idx])))
            penalties.append(max(0.0, threshold - diff) / threshold)
    if not penalties:
        return 0.0
    return float(np.mean(np.asarray(penalties, dtype=np.float64)))


def _weighted_elite_average(elite_banks: np.ndarray, elite_losses: Sequence[float]) -> np.ndarray:
    losses = np.asarray(elite_losses, dtype=np.float64)
    loss_scale = max(float(np.std(losses)), 1e-6)
    logits = -((losses - float(losses.min())) / loss_scale)
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    weights = weights / float(np.sum(weights))
    return np.sum(elite_banks * weights[:, None, None], axis=0, dtype=np.float64)


def _clip_sigma(*, low: float, high: float | None, value: float | None = None) -> float:
    sigma = float(low if value is None else value)
    sigma = max(float(low), sigma)
    if high is not None:
        sigma = min(float(high), sigma)
    return float(sigma)


def _clip_array(values: np.ndarray, *, low: float, high: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), float(low), float(high)).astype(np.float64)


def _support_low(support: dict[str, Any] | None, *, default: float) -> float:
    if support is None:
        return float(default)
    return float(support.get("low", support.get("interval_low", default)))


def _support_high(support: dict[str, Any] | None, *, default: float) -> float:
    if support is None:
        return float(default)
    return float(support.get("high", support.get("interval_high", default)))


def _support_sigma(
    support: dict[str, Any] | None,
    *,
    default: float,
    override: float | None = None,
) -> float:
    if override is not None:
        return float(override)
    if support is None:
        return float(default)
    if "sigma" in support:
        return float(support["sigma"])
    return float(default)


def _evaluate_bank_validation(
    *,
    problem: CodeCapacityProblem,
    config: StaticAssignmentBankConfig,
    half_cases: Sequence[dict[str, Any]],
    random_validation_cases: Sequence[dict[str, Any]],
    memory_bank: np.ndarray | None,
    damping_bank: np.ndarray | None,
    round_idx: int,
) -> dict[str, Any]:
    random_practical = evaluate_assignment_bank(
        problem=problem,
        cases=random_validation_cases,
        family=config.family,
        memory_bank=memory_bank,
        damping_bank=damping_bank,
        max_iter=config.max_iter,
        alpha=config.alpha,
        selection_mode="practical",
        logical_failure_penalty=config.logical_failure_penalty,
        logical_weight_penalty=config.logical_weight_penalty,
        convergence_penalty=config.convergence_penalty,
        iteration_penalty=config.iteration_penalty,
    )
    half_practical = evaluate_assignment_bank(
        problem=problem,
        cases=half_cases,
        family=config.family,
        memory_bank=memory_bank,
        damping_bank=damping_bank,
        max_iter=config.max_iter,
        alpha=config.alpha,
        selection_mode="practical",
        logical_failure_penalty=config.logical_failure_penalty,
        logical_weight_penalty=config.logical_weight_penalty,
        convergence_penalty=config.convergence_penalty,
        iteration_penalty=config.iteration_penalty,
    )
    random_oracle = evaluate_assignment_bank(
        problem=problem,
        cases=random_validation_cases,
        family=config.family,
        memory_bank=memory_bank,
        damping_bank=damping_bank,
        max_iter=config.max_iter,
        alpha=config.alpha,
        selection_mode="oracle",
        logical_failure_penalty=config.logical_failure_penalty,
        logical_weight_penalty=config.logical_weight_penalty,
        convergence_penalty=config.convergence_penalty,
        iteration_penalty=config.iteration_penalty,
    )
    half_oracle = evaluate_assignment_bank(
        problem=problem,
        cases=half_cases,
        family=config.family,
        memory_bank=memory_bank,
        damping_bank=damping_bank,
        max_iter=config.max_iter,
        alpha=config.alpha,
        selection_mode="oracle",
        logical_failure_penalty=config.logical_failure_penalty,
        logical_weight_penalty=config.logical_weight_penalty,
        convergence_penalty=config.convergence_penalty,
        iteration_penalty=config.iteration_penalty,
    )
    return {
        "round": int(round_idx + 1),
        "random_practical_logical_success_rate": float(random_practical["logical_success_rate"]),
        "half_practical_logical_success_rate": float(half_practical["logical_success_rate"]),
        "random_practical_convergence_rate": float(random_practical["convergence_rate"]),
        "half_practical_convergence_rate": float(half_practical["convergence_rate"]),
        "random_practical_mean_iterations": float(random_practical["mean_iterations"]),
        "half_practical_mean_iterations": float(half_practical["mean_iterations"]),
        "random_oracle_logical_success_rate": float(random_oracle["logical_success_rate"]),
        "half_oracle_logical_success_rate": float(half_oracle["logical_success_rate"]),
    }


def _is_better_bank_validation(candidate: dict[str, Any], current_best: dict[str, Any] | None) -> bool:
    if current_best is None:
        return True
    return _bank_validation_key(candidate) > _bank_validation_key(current_best)


def _bank_validation_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float]:
    return (
        float(row["random_practical_logical_success_rate"]),
        float(row["half_practical_logical_success_rate"]),
        float(row["random_practical_convergence_rate"]),
        float(row["half_practical_convergence_rate"]),
        -float(row["random_practical_mean_iterations"]),
        -float(row["half_practical_mean_iterations"]),
        float(row["random_oracle_logical_success_rate"]),
        float(row["half_oracle_logical_success_rate"]),
    )


def _save_bank_checkpoint(
    *,
    checkpoint_dir: Path,
    memory_bank: np.ndarray | None,
    damping_bank: np.ndarray | None,
    memory_sigma: float,
    damping_sigma: float,
    completed_round: int,
) -> None:
    np.savez_compressed(
        checkpoint_dir / "checkpoint_latest.npz",
        memory_bank=np.asarray([] if memory_bank is None else memory_bank, dtype=np.float64),
        damping_bank=np.asarray([] if damping_bank is None else damping_bank, dtype=np.float64),
        memory_sigma=np.asarray([float(memory_sigma)], dtype=np.float64),
        damping_sigma=np.asarray([float(damping_sigma)], dtype=np.float64),
        completed_round=np.asarray([int(completed_round)], dtype=np.int64),
    )


def _bootstrap_difference_report(
    *,
    candidate_rows: Sequence[dict[str, Any]],
    baseline_rows: Sequence[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    logical_diffs = [
        float(candidate["logical_success_rate"]) - float(baseline["logical_success_rate"])
        for candidate, baseline in zip(candidate_rows, baseline_rows)
    ]
    convergence_diffs = [
        float(candidate["convergence_rate"]) - float(baseline["convergence_rate"])
        for candidate, baseline in zip(candidate_rows, baseline_rows)
    ]
    iteration_ratios = [
        float(candidate["mean_iterations"]) / max(float(baseline["mean_iterations"]), 1e-12)
        for candidate, baseline in zip(candidate_rows, baseline_rows)
    ]
    return {
        "logical_success_difference": bootstrap_confidence_interval(
            logical_diffs,
            bootstrap_samples=int(bootstrap_samples),
            seed=int(seed),
        ),
        "convergence_difference": bootstrap_confidence_interval(
            convergence_diffs,
            bootstrap_samples=int(bootstrap_samples),
            seed=int(seed) + 1,
        ),
        "iteration_ratio": bootstrap_confidence_interval(
            iteration_ratios,
            bootstrap_samples=int(bootstrap_samples),
            seed=int(seed) + 2,
        ),
    }


def _average_metric_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty.")
    aggregate = {
        "logical_success_rate": 0.0,
        "convergence_rate": 0.0,
        "exact_recovery_rate": 0.0,
        "mean_iterations": 0.0,
        "loss_mean": 0.0,
    }
    for row in rows:
        for key in aggregate:
            aggregate[key] += float(row[key])
    count = float(len(rows))
    return {key: float(value / count) for key, value in aggregate.items()}


def _config_payload(problem: CodeCapacityProblem, config: StaticAssignmentBankConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["n_bits"] = int(problem.n_bits)
    payload["n_edges"] = int(problem.hz.nnz)
    payload["n_logicals"] = int(problem.n_logicals)
    payload["metadata"] = problem.metadata
    return payload
