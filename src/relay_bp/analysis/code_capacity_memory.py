from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from tqdm.auto import tqdm

from .code_capacity_common import (
    CodeCapacityProblem,
    decode_with_memory_strengths,
    evaluate_decode_result,
    make_min_sum_tracer,
)
from .code_capacity_training import evaluate_memory_weight_vector

SAMPLING_FAMILIES = {"uniform_interval", "truncated_gaussian"}
APPLICATION_SCOPES = {"all_bits", "stabilizer_support"}
SELECTION_MODES = {"single_static", "single_draw", "k_draw_oracle", "k_draw_practical"}


@dataclass(frozen=True)
class MemorySamplingSpec:
    sampling_family: str
    application_scope: str = "all_bits"
    selection_mode: str = "single_draw"
    draw_count: int = 1
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    mean_vector: np.ndarray | None = None
    sigma: float | None = None
    static_memory_strengths: np.ndarray | None = None

    def __post_init__(self) -> None:
        sampling_family = str(self.sampling_family)
        application_scope = str(self.application_scope)
        selection_mode = str(self.selection_mode)
        if sampling_family not in SAMPLING_FAMILIES:
            raise ValueError(f"Unsupported sampling_family: {sampling_family!r}")
        if application_scope not in APPLICATION_SCOPES:
            raise ValueError(f"Unsupported application_scope: {application_scope!r}")
        if selection_mode not in SELECTION_MODES:
            raise ValueError(f"Unsupported selection_mode: {selection_mode!r}")
        if int(self.draw_count) <= 0:
            raise ValueError("draw_count must be positive.")
        if selection_mode == "single_draw" and int(self.draw_count) != 1:
            raise ValueError("single_draw requires draw_count == 1.")
        if selection_mode == "single_static" and application_scope != "all_bits":
            raise ValueError("single_static only supports application_scope='all_bits'.")
        if self.static_memory_strengths is not None and selection_mode != "single_static":
            raise ValueError("static_memory_strengths is only valid for selection_mode='single_static'.")
        if self.static_memory_strengths is not None and selection_mode == "single_static":
            return
        if sampling_family == "uniform_interval":
            if self.low is None or self.high is None:
                raise ValueError("uniform_interval requires low and high.")
            if float(self.high) < float(self.low):
                raise ValueError("uniform_interval requires high >= low.")
        if sampling_family == "truncated_gaussian":
            if self.low is None or self.high is None:
                raise ValueError("truncated_gaussian requires low and high.")
            if float(self.high) < float(self.low):
                raise ValueError("truncated_gaussian requires high >= low.")
            if self.mean is None and self.mean_vector is None:
                raise ValueError("truncated_gaussian requires either mean or mean_vector.")
            if self.sigma is None:
                raise ValueError("truncated_gaussian requires sigma.")
            if float(self.sigma) < 0.0:
                raise ValueError("truncated_gaussian requires sigma >= 0.")


@dataclass(frozen=True)
class EquivalenceThresholds:
    logical_margin: float = -0.01
    convergence_margin: float = -0.02
    iteration_ratio_ceiling: float = 1.10
    confidence: float = 0.95
    bootstrap_samples: int = 2000
    seed: int = 0


def interval_from_center_width(center: float, width: float) -> tuple[float, float]:
    half_width = max(0.0, float(width) / 2.0)
    low = float(center) - half_width
    high = float(center) + half_width
    if high < low:
        low, high = high, low
    return (low, high)


def get_memory_target_indices(
    n_bits: int,
    case: dict[str, Any],
    *,
    application_scope: str,
) -> np.ndarray:
    application_scope = str(application_scope)
    if application_scope == "all_bits":
        return np.arange(int(n_bits), dtype=int)
    if application_scope == "stabilizer_support":
        support_bits = case.get("support_bits")
        if support_bits is None:
            raise ValueError("stabilizer_support requires case['support_bits'].")
        return np.asarray(support_bits, dtype=int)
    raise ValueError(f"Unsupported application_scope: {application_scope!r}")


def sample_memory_strengths(
    *,
    problem: CodeCapacityProblem,
    case: dict[str, Any],
    sampling_spec: MemorySamplingSpec,
    seed: int,
) -> np.ndarray:
    if sampling_spec.selection_mode == "single_static" and sampling_spec.static_memory_strengths is not None:
        weights = np.asarray(sampling_spec.static_memory_strengths, dtype=np.float64)
        if weights.shape != (problem.n_bits,):
            raise ValueError(
                f"static_memory_strengths has shape {weights.shape}, expected {(problem.n_bits,)}."
            )
        return weights

    rng = np.random.default_rng(int(seed))
    if sampling_spec.selection_mode == "single_static":
        return _sample_full_vector(problem.n_bits, sampling_spec=sampling_spec, rng=rng)

    weights = np.zeros(problem.n_bits, dtype=np.float64)
    target_indices = get_memory_target_indices(
        problem.n_bits,
        case,
        application_scope=sampling_spec.application_scope,
    )
    if target_indices.size == 0:
        return weights
    weights[target_indices] = _sample_values(
        size=int(target_indices.size),
        sampling_spec=sampling_spec,
        rng=rng,
        target_indices=target_indices,
    )
    return weights


def evaluate_sampling_strategy(
    *,
    problem: CodeCapacityProblem,
    cases: Sequence[dict[str, Any]],
    sampling_spec: MemorySamplingSpec,
    max_iter: int,
    alpha: float | None,
    logical_weight_penalty: float,
    convergence_penalty: float,
    iteration_penalty: float,
    base_seed: int,
    return_case_rows: bool = False,
) -> dict[str, Any]:
    tracer = make_min_sum_tracer(problem, max_iter=max_iter, alpha=alpha, gamma0=0.0)
    static_weights = None
    if sampling_spec.selection_mode == "single_static":
        static_weights = sample_memory_strengths(
            problem=problem,
            case={"support_bits": tuple(range(problem.n_bits))},
            sampling_spec=sampling_spec,
            seed=int(base_seed),
        )
    logical_success_sum = 0.0
    exact_recovery_sum = 0.0
    convergence_sum = 0.0
    iterations_sum = 0.0
    logical_weight_sum = 0.0
    loss_sum = 0.0
    case_rows: list[dict[str, Any]] = []

    for case in cases:
        draw_rows: list[dict[str, Any]] = []
        case_id = int(case.get("case_id", case.get("sample_idx", 0)))
        for draw_idx in range(_selection_draw_count(sampling_spec)):
            if static_weights is None:
                weights = sample_memory_strengths(
                    problem=problem,
                    case=case,
                    sampling_spec=sampling_spec,
                    seed=_case_draw_seed(base_seed=base_seed, case_id=case_id, draw_idx=draw_idx),
                )
            else:
                weights = static_weights
            result = decode_with_memory_strengths(tracer, case["syndrome"], weights)
            evaluation = evaluate_decode_result(result=result, error=case["error"], lz=problem.lz)
            draw_rows.append(
                {
                    "draw_idx": int(draw_idx),
                    "converged": bool(evaluation.converged),
                    "logical_success": bool(evaluation.logical_success),
                    "exact_recovery": bool(evaluation.exact_recovery),
                    "logical_weight": int(evaluation.logical_weight),
                    "iterations": int(evaluation.iterations),
                    "decoding_weight": int(np.asarray(result.decoding, dtype=np.uint8).sum()),
                    "loss": float(
                        _case_loss(
                            logical_weight=evaluation.logical_weight,
                            converged=evaluation.converged,
                            iterations=evaluation.iterations,
                            max_iter=max_iter,
                            logical_weight_penalty=logical_weight_penalty,
                            convergence_penalty=convergence_penalty,
                            iteration_penalty=iteration_penalty,
                        )
                    ),
                }
            )
        selected_row = _select_draw_row(draw_rows, selection_mode=sampling_spec.selection_mode)
        logical_success_sum += float(selected_row["logical_success"])
        exact_recovery_sum += float(selected_row["exact_recovery"])
        convergence_sum += float(selected_row["converged"])
        iterations_sum += float(selected_row["iterations"])
        logical_weight_sum += float(selected_row["logical_weight"])
        loss_sum += float(selected_row["loss"])
        if return_case_rows:
            case_rows.append(
                {
                    "case_id": case_id,
                    "row_idx": int(case.get("row_idx", -1)),
                    "row_weight": int(case.get("row_weight", -1)),
                    "support_bits": list(case.get("support_bits", ())),
                    "flipped_bits": list(case.get("flipped_bits", ())),
                    "selection_mode": sampling_spec.selection_mode,
                    "draw_count": int(_selection_draw_count(sampling_spec)),
                    **selected_row,
                }
            )

    num_cases = max(1, len(cases))
    metrics = {
        "num_cases": int(len(cases)),
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


def evaluate_matched_support_heatmaps(
    *,
    problem: CodeCapacityProblem,
    cases: Sequence[dict[str, Any]],
    max_iter: int,
    alpha: float | None,
    centers: Sequence[float],
    widths: Sequence[float],
    base_seed: int,
    random_draws_per_point: int,
    application_scope: str,
    selection_mode: str,
    logical_weight_penalty: float,
    convergence_penalty: float,
    iteration_penalty: float,
    show_progress: bool = False,
) -> dict[str, Any]:
    shape = (len(widths), len(centers))
    artifact = {
        "centers": np.asarray(centers, dtype=np.float64),
        "widths": np.asarray(widths, dtype=np.float64),
        "application_scope": str(application_scope),
        "selection_mode": str(selection_mode),
        "random_draws_per_point": int(random_draws_per_point),
        "families": {
            family: _empty_metric_grid(shape)
            for family in ("uniform_interval", "truncated_gaussian")
        },
        "interval_lows": np.full(shape, np.nan, dtype=np.float64),
        "interval_highs": np.full(shape, np.nan, dtype=np.float64),
        "gaussian_sigmas": np.full(shape, np.nan, dtype=np.float64),
    }
    points = [
        (width_idx, float(width), center_idx, float(center))
        for width_idx, width in enumerate(widths)
        for center_idx, center in enumerate(centers)
    ]
    points_iter = tqdm(points, total=len(points), desc="matched-support heatmap", disable=not show_progress)
    for width_idx, width, center_idx, center in points_iter:
        low, high = interval_from_center_width(center, width)
        artifact["interval_lows"][width_idx, center_idx] = low
        artifact["interval_highs"][width_idx, center_idx] = high
        gaussian_sigma = max(0.0, float(width) / 4.0)
        artifact["gaussian_sigmas"][width_idx, center_idx] = gaussian_sigma
        family_specs = {
            "uniform_interval": MemorySamplingSpec(
                sampling_family="uniform_interval",
                application_scope=application_scope,
                selection_mode=selection_mode,
                draw_count=_strategy_draw_count(selection_mode=selection_mode, random_draws_per_point=random_draws_per_point),
                low=low,
                high=high,
            ),
            "truncated_gaussian": MemorySamplingSpec(
                sampling_family="truncated_gaussian",
                application_scope=application_scope,
                selection_mode=selection_mode,
                draw_count=_strategy_draw_count(selection_mode=selection_mode, random_draws_per_point=random_draws_per_point),
                low=low,
                high=high,
                mean=float(center),
                sigma=gaussian_sigma,
            ),
        }
        for family_name, sampling_spec in family_specs.items():
            metrics = _evaluate_sampling_repeats(
                problem=problem,
                cases=cases,
                sampling_spec=sampling_spec,
                max_iter=max_iter,
                alpha=alpha,
                logical_weight_penalty=logical_weight_penalty,
                convergence_penalty=convergence_penalty,
                iteration_penalty=iteration_penalty,
                base_seed=int(base_seed) + 100_003 * width_idx + 1_003 * center_idx,
                repeat_count=_repeat_count(selection_mode=selection_mode, random_draws_per_point=random_draws_per_point),
            )
            _store_metric_grid_cell(artifact["families"][family_name], width_idx, center_idx, metrics)
    return artifact


def evaluate_gaussian_refinement_heatmap(
    *,
    problem: CodeCapacityProblem,
    cases: Sequence[dict[str, Any]],
    max_iter: int,
    alpha: float | None,
    means: Sequence[float],
    sigmas: Sequence[float],
    low: float,
    high: float,
    base_seed: int,
    random_draws_per_point: int,
    application_scope: str,
    selection_mode: str,
    logical_weight_penalty: float,
    convergence_penalty: float,
    iteration_penalty: float,
    show_progress: bool = False,
) -> dict[str, Any]:
    shape = (len(sigmas), len(means))
    artifact = {
        "means": np.asarray(means, dtype=np.float64),
        "sigmas": np.asarray(sigmas, dtype=np.float64),
        "application_scope": str(application_scope),
        "selection_mode": str(selection_mode),
        "random_draws_per_point": int(random_draws_per_point),
        "low": float(low),
        "high": float(high),
        **_empty_metric_grid(shape),
    }
    points = [
        (sigma_idx, float(sigma), mean_idx, float(mean))
        for sigma_idx, sigma in enumerate(sigmas)
        for mean_idx, mean in enumerate(means)
    ]
    points_iter = tqdm(points, total=len(points), desc="gaussian refinement heatmap", disable=not show_progress)
    for sigma_idx, sigma, mean_idx, mean in points_iter:
        sampling_spec = MemorySamplingSpec(
            sampling_family="truncated_gaussian",
            application_scope=application_scope,
            selection_mode=selection_mode,
            draw_count=_strategy_draw_count(selection_mode=selection_mode, random_draws_per_point=random_draws_per_point),
            low=float(low),
            high=float(high),
            mean=float(mean),
            sigma=float(sigma),
        )
        metrics = _evaluate_sampling_repeats(
            problem=problem,
            cases=cases,
            sampling_spec=sampling_spec,
            max_iter=max_iter,
            alpha=alpha,
            logical_weight_penalty=logical_weight_penalty,
            convergence_penalty=convergence_penalty,
            iteration_penalty=iteration_penalty,
            base_seed=int(base_seed) + 100_003 * sigma_idx + 1_003 * mean_idx,
            repeat_count=_repeat_count(selection_mode=selection_mode, random_draws_per_point=random_draws_per_point),
        )
        _store_metric_grid_cell(artifact, sigma_idx, mean_idx, metrics)
    return artifact


def matched_support_heatmap_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_name, family_artifact in artifact["families"].items():
        for width_idx, width in enumerate(artifact["widths"]):
            for center_idx, center in enumerate(artifact["centers"]):
                rows.append(
                    {
                        "family": family_name,
                        "center": float(center),
                        "width": float(width),
                        "interval_low": float(artifact["interval_lows"][width_idx, center_idx]),
                        "interval_high": float(artifact["interval_highs"][width_idx, center_idx]),
                        "sigma": float(artifact["gaussian_sigmas"][width_idx, center_idx]),
                        "logical_success_rate": float(family_artifact["logical_success_rate"][width_idx, center_idx]),
                        "convergence_rate": float(family_artifact["convergence_rate"][width_idx, center_idx]),
                        "exact_recovery_rate": float(family_artifact["exact_recovery_rate"][width_idx, center_idx]),
                        "mean_iterations": float(family_artifact["mean_iterations"][width_idx, center_idx]),
                        "mean_logical_weight": float(family_artifact["mean_logical_weight"][width_idx, center_idx]),
                        "loss_mean": float(family_artifact["loss_mean"][width_idx, center_idx]),
                        "trial_count": int(family_artifact["trial_count"][width_idx, center_idx]),
                    }
                )
    return rows


def gaussian_refinement_heatmap_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sigma_idx, sigma in enumerate(artifact["sigmas"]):
        for mean_idx, mean in enumerate(artifact["means"]):
            rows.append(
                {
                    "mean": float(mean),
                    "sigma": float(sigma),
                    "low": float(artifact["low"]),
                    "high": float(artifact["high"]),
                    "logical_success_rate": float(artifact["logical_success_rate"][sigma_idx, mean_idx]),
                    "convergence_rate": float(artifact["convergence_rate"][sigma_idx, mean_idx]),
                    "exact_recovery_rate": float(artifact["exact_recovery_rate"][sigma_idx, mean_idx]),
                    "mean_iterations": float(artifact["mean_iterations"][sigma_idx, mean_idx]),
                    "mean_logical_weight": float(artifact["mean_logical_weight"][sigma_idx, mean_idx]),
                    "loss_mean": float(artifact["loss_mean"][sigma_idx, mean_idx]),
                    "trial_count": int(artifact["trial_count"][sigma_idx, mean_idx]),
                }
            )
    return rows


def best_heatmap_rows(rows: Sequence[dict[str, Any]], *, topk: int = 5) -> list[dict[str, Any]]:
    return list(
        sorted(
            rows,
            key=lambda row: (
                float(row["loss_mean"]),
                -float(row["logical_success_rate"]),
                -float(row["convergence_rate"]),
                float(row["mean_iterations"]),
            ),
        )[: max(1, int(topk))]
    )


def bootstrap_confidence_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("values must not be empty.")
    if values.size == 1:
        value = float(values[0])
        return {"mean": value, "lower": value, "upper": value}
    rng = np.random.default_rng(int(seed))
    sampled_means = np.empty(int(bootstrap_samples), dtype=np.float64)
    for sample_idx in range(int(bootstrap_samples)):
        indices = rng.integers(0, values.size, size=values.size)
        sampled_means[sample_idx] = float(np.mean(values[indices]))
    alpha = max(0.0, min(1.0, 1.0 - float(confidence)))
    return {
        "mean": float(np.mean(values)),
        "lower": float(np.quantile(sampled_means, alpha / 2.0)),
        "upper": float(np.quantile(sampled_means, 1.0 - alpha / 2.0)),
    }


def evaluate_equivalence(
    *,
    candidate_rows: Sequence[dict[str, Any]],
    baseline_rows: Sequence[dict[str, Any]],
    thresholds: EquivalenceThresholds = EquivalenceThresholds(),
) -> dict[str, Any]:
    if len(candidate_rows) != len(baseline_rows):
        raise ValueError("candidate_rows and baseline_rows must have the same length.")
    paired_candidate = list(candidate_rows)
    paired_baseline = list(baseline_rows)
    logical_diff = np.asarray(
        [float(c["logical_success_rate"]) - float(b["logical_success_rate"]) for c, b in zip(paired_candidate, paired_baseline)],
        dtype=np.float64,
    )
    convergence_diff = np.asarray(
        [float(c["convergence_rate"]) - float(b["convergence_rate"]) for c, b in zip(paired_candidate, paired_baseline)],
        dtype=np.float64,
    )
    iteration_ratio = np.asarray(
        [
            float(c["mean_iterations"]) / max(float(b["mean_iterations"]), 1e-12)
            for c, b in zip(paired_candidate, paired_baseline)
        ],
        dtype=np.float64,
    )
    logical_ci = bootstrap_confidence_interval(
        logical_diff,
        confidence=thresholds.confidence,
        bootstrap_samples=thresholds.bootstrap_samples,
        seed=thresholds.seed,
    )
    convergence_ci = bootstrap_confidence_interval(
        convergence_diff,
        confidence=thresholds.confidence,
        bootstrap_samples=thresholds.bootstrap_samples,
        seed=thresholds.seed + 1,
    )
    iteration_ci = bootstrap_confidence_interval(
        iteration_ratio,
        confidence=thresholds.confidence,
        bootstrap_samples=thresholds.bootstrap_samples,
        seed=thresholds.seed + 2,
    )
    verdict = (
        logical_ci["lower"] >= float(thresholds.logical_margin)
        and convergence_ci["lower"] >= float(thresholds.convergence_margin)
        and iteration_ci["upper"] <= float(thresholds.iteration_ratio_ceiling)
    )
    return {
        "verdict": bool(verdict),
        "thresholds": {
            "logical_margin": float(thresholds.logical_margin),
            "convergence_margin": float(thresholds.convergence_margin),
            "iteration_ratio_ceiling": float(thresholds.iteration_ratio_ceiling),
            "confidence": float(thresholds.confidence),
            "bootstrap_samples": int(thresholds.bootstrap_samples),
        },
        "logical_success_difference": logical_ci,
        "convergence_difference": convergence_ci,
        "iteration_ratio": iteration_ci,
    }


def _sample_full_vector(
    n_bits: int,
    *,
    sampling_spec: MemorySamplingSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    if sampling_spec.static_memory_strengths is not None:
        weights = np.asarray(sampling_spec.static_memory_strengths, dtype=np.float64)
        if weights.shape != (int(n_bits),):
            raise ValueError(f"static_memory_strengths has shape {weights.shape}, expected {(int(n_bits),)}.")
        return weights
    return _sample_values(
        size=int(n_bits),
        sampling_spec=sampling_spec,
        rng=rng,
        target_indices=np.arange(int(n_bits), dtype=int),
    )


def _sample_values(
    *,
    size: int,
    sampling_spec: MemorySamplingSpec,
    rng: np.random.Generator,
    target_indices: np.ndarray,
) -> np.ndarray:
    if size <= 0:
        return np.zeros(0, dtype=np.float64)
    if sampling_spec.sampling_family == "uniform_interval":
        low = float(sampling_spec.low)
        high = float(sampling_spec.high)
        if np.isclose(low, high):
            return np.full(int(size), low, dtype=np.float64)
        return rng.uniform(low, high, size=int(size)).astype(np.float64)
    if sampling_spec.sampling_family == "truncated_gaussian":
        means = _resolve_gaussian_means(sampling_spec, target_indices=target_indices, size=size)
        return _sample_truncated_gaussian(
            means=means,
            sigma=float(sampling_spec.sigma),
            low=float(sampling_spec.low),
            high=float(sampling_spec.high),
            rng=rng,
        )
    raise ValueError(f"Unsupported sampling_family: {sampling_spec.sampling_family!r}")


def _resolve_gaussian_means(
    sampling_spec: MemorySamplingSpec,
    *,
    target_indices: np.ndarray,
    size: int,
) -> np.ndarray:
    if sampling_spec.mean_vector is not None:
        mean_vector = np.asarray(sampling_spec.mean_vector, dtype=np.float64).reshape(-1)
        if target_indices.size and int(np.max(target_indices)) >= int(mean_vector.size):
            raise ValueError(
                "mean_vector is shorter than the requested target indices."
            )
        if (not target_indices.size) and mean_vector.size < int(size):
            raise ValueError("mean_vector is shorter than the requested size.")
        if target_indices.size:
            return mean_vector[target_indices]
        return mean_vector[: int(size)]
    return np.full(int(size), float(sampling_spec.mean), dtype=np.float64)


def _sample_truncated_gaussian(
    *,
    means: np.ndarray,
    sigma: float,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    means = np.asarray(means, dtype=np.float64).reshape(-1)
    if means.size == 0:
        return means
    if np.isclose(low, high):
        return np.full(means.shape, float(low), dtype=np.float64)
    if float(sigma) <= 1e-12:
        return np.clip(means, float(low), float(high)).astype(np.float64)
    samples = np.empty_like(means, dtype=np.float64)
    remaining = np.arange(means.size, dtype=int)
    max_rounds = 32
    for _ in range(max_rounds):
        if remaining.size == 0:
            break
        draws = rng.normal(loc=means[remaining], scale=float(sigma), size=remaining.size)
        accepted = (draws >= float(low)) & (draws <= float(high))
        if np.any(accepted):
            samples[remaining[accepted]] = draws[accepted]
            remaining = remaining[~accepted]
    if remaining.size:
        residual_draws = rng.normal(loc=means[remaining], scale=float(sigma), size=remaining.size)
        samples[remaining] = np.clip(residual_draws, float(low), float(high))
    return samples.astype(np.float64)


def _selection_draw_count(sampling_spec: MemorySamplingSpec) -> int:
    if sampling_spec.selection_mode == "single_static":
        return 1
    return int(sampling_spec.draw_count)


def _strategy_draw_count(*, selection_mode: str, random_draws_per_point: int) -> int:
    if selection_mode in {"single_static", "single_draw"}:
        return 1
    return max(1, int(random_draws_per_point))


def _repeat_count(*, selection_mode: str, random_draws_per_point: int) -> int:
    if selection_mode in {"single_static", "single_draw"}:
        return max(1, int(random_draws_per_point))
    return 1


def _evaluate_sampling_repeats(
    *,
    problem: CodeCapacityProblem,
    cases: Sequence[dict[str, Any]],
    sampling_spec: MemorySamplingSpec,
    max_iter: int,
    alpha: float | None,
    logical_weight_penalty: float,
    convergence_penalty: float,
    iteration_penalty: float,
    base_seed: int,
    repeat_count: int,
) -> dict[str, Any]:
    repeat_count = max(1, int(repeat_count))
    aggregate = {
        "loss_mean": 0.0,
        "logical_success_rate": 0.0,
        "exact_recovery_rate": 0.0,
        "convergence_rate": 0.0,
        "mean_iterations": 0.0,
        "mean_logical_weight": 0.0,
    }
    first_metrics: dict[str, Any] | None = None
    for repeat_idx in range(repeat_count):
        metrics = evaluate_sampling_strategy(
            problem=problem,
            cases=cases,
            sampling_spec=sampling_spec,
            max_iter=max_iter,
            alpha=alpha,
            logical_weight_penalty=logical_weight_penalty,
            convergence_penalty=convergence_penalty,
            iteration_penalty=iteration_penalty,
            base_seed=int(base_seed) + 10_007 * repeat_idx,
        )
        if first_metrics is None:
            first_metrics = metrics
        for key in aggregate:
            aggregate[key] += float(metrics[key])
    assert first_metrics is not None
    return {
        "num_cases": int(first_metrics["num_cases"]),
        **{key: float(value / repeat_count) for key, value in aggregate.items()},
    }


def _select_draw_row(draw_rows: Sequence[dict[str, Any]], *, selection_mode: str) -> dict[str, Any]:
    if not draw_rows:
        raise ValueError("draw_rows must not be empty.")
    if selection_mode in {"single_static", "single_draw"}:
        return dict(draw_rows[0])
    if selection_mode == "k_draw_practical":
        return dict(
            min(
                draw_rows,
                key=lambda row: (
                    not bool(row["logical_success"]),
                    int(row["iterations"]),
                    int(row["decoding_weight"]),
                    int(row["draw_idx"]),
                ),
            )
        )
    if selection_mode == "k_draw_oracle":
        return dict(
            min(
                draw_rows,
                key=lambda row: (
                    not bool(row["logical_success"]),
                    int(row["iterations"]),
                    int(row["decoding_weight"]),
                    int(row["draw_idx"]),
                ),
            )
        )
    raise ValueError(f"Unsupported selection_mode: {selection_mode!r}")


def _case_draw_seed(*, base_seed: int, case_id: int, draw_idx: int) -> int:
    return int(base_seed) + 1_000_003 * int(case_id) + 10_007 * int(draw_idx)


def _empty_metric_grid(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        "loss_mean": np.full(shape, np.nan, dtype=np.float64),
        "logical_success_rate": np.full(shape, np.nan, dtype=np.float64),
        "exact_recovery_rate": np.full(shape, np.nan, dtype=np.float64),
        "convergence_rate": np.full(shape, np.nan, dtype=np.float64),
        "mean_iterations": np.full(shape, np.nan, dtype=np.float64),
        "mean_logical_weight": np.full(shape, np.nan, dtype=np.float64),
        "trial_count": np.zeros(shape, dtype=np.int64),
    }


def _store_metric_grid_cell(
    artifact: dict[str, np.ndarray],
    row_idx: int,
    col_idx: int,
    metrics: dict[str, Any],
) -> None:
    for key in ("loss_mean", "logical_success_rate", "exact_recovery_rate", "convergence_rate", "mean_iterations", "mean_logical_weight"):
        artifact[key][row_idx, col_idx] = float(metrics[key])
    artifact["trial_count"][row_idx, col_idx] = int(metrics["num_cases"])


def _case_loss(
    *,
    logical_weight: int,
    converged: bool,
    iterations: int,
    max_iter: int,
    logical_weight_penalty: float,
    convergence_penalty: float,
    iteration_penalty: float,
) -> float:
    return float(
        float(logical_weight_penalty) * float(logical_weight)
        + float(convergence_penalty) * float(not converged)
        + float(iteration_penalty) * (float(iterations) / max(1, int(max_iter)))
    )


def evaluate_static_vector_against_sampler(
    *,
    problem: CodeCapacityProblem,
    cases: Sequence[dict[str, Any]],
    static_memory_strengths: np.ndarray,
    baseline_sampling_spec: MemorySamplingSpec,
    max_iter: int,
    alpha: float | None,
    logical_weight_penalty: float,
    convergence_penalty: float,
    iteration_penalty: float,
    base_seed: int,
) -> dict[str, Any]:
    static_metrics = evaluate_memory_weight_vector(
        problem=problem,
        cases=cases,
        memory_strengths=np.asarray(static_memory_strengths, dtype=np.float64),
        max_iter=max_iter,
        alpha=alpha,
        logical_weight_penalty=logical_weight_penalty,
        convergence_penalty=convergence_penalty,
        iteration_penalty=iteration_penalty,
    )
    baseline_metrics = evaluate_sampling_strategy(
        problem=problem,
        cases=cases,
        sampling_spec=baseline_sampling_spec,
        max_iter=max_iter,
        alpha=alpha,
        logical_weight_penalty=logical_weight_penalty,
        convergence_penalty=convergence_penalty,
        iteration_penalty=iteration_penalty,
        base_seed=base_seed,
    )
    return {
        "static": static_metrics,
        "baseline": baseline_metrics,
    }
