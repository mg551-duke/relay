# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from relay_bp.analysis import enumerate_half_stabilizer_cases
from relay_bp.analysis.code_capacity_common import CodeCapacityProblem
from relay_bp.analysis.code_capacity_memory import (
    EquivalenceThresholds,
    MemorySamplingSpec,
    _select_draw_row,
    best_heatmap_rows,
    evaluate_equivalence,
    evaluate_gaussian_refinement_heatmap,
    evaluate_matched_support_heatmaps,
    gaussian_refinement_heatmap_rows,
    matched_support_heatmap_rows,
    sample_memory_strengths,
)


def _toy_problem() -> CodeCapacityProblem:
    hx = np.array(
        [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    hz = csc_matrix(
        np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
            ],
            dtype=np.uint8,
        )
    )
    lz = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    return CodeCapacityProblem(
        path=Path("toy_memory_code.npz"),
        hx=hx,
        hz=hz,
        lz=lz,
        error_priors=np.full(4, 0.1, dtype=np.float64),
        metadata={"name": "toy-memory"},
    )


def test_uniform_sampler_is_deterministic_and_respects_bounds():
    problem = _toy_problem()
    case = {"support_bits": (0, 1, 2, 3)}
    sampling_spec = MemorySamplingSpec(
        sampling_family="uniform_interval",
        application_scope="all_bits",
        selection_mode="single_draw",
        low=-0.2,
        high=0.4,
    )

    first = sample_memory_strengths(problem=problem, case=case, sampling_spec=sampling_spec, seed=17)
    second = sample_memory_strengths(problem=problem, case=case, sampling_spec=sampling_spec, seed=17)

    assert first == pytest.approx(second)
    assert np.all(first >= -0.2)
    assert np.all(first <= 0.4)


def test_sampler_respects_stabilizer_support_scope():
    problem = _toy_problem()
    case = {"support_bits": (1, 3)}
    sampling_spec = MemorySamplingSpec(
        sampling_family="uniform_interval",
        application_scope="stabilizer_support",
        selection_mode="single_draw",
        low=0.1,
        high=0.2,
    )

    weights = sample_memory_strengths(problem=problem, case=case, sampling_spec=sampling_spec, seed=3)

    assert weights[0] == pytest.approx(0.0)
    assert weights[2] == pytest.approx(0.0)
    assert 0.1 <= weights[1] <= 0.2
    assert 0.1 <= weights[3] <= 0.2


def test_single_static_sampler_accepts_direct_weights_without_interval_parameters():
    problem = _toy_problem()
    weights = np.array([-0.1, 0.2, 0.0, 0.15], dtype=np.float64)
    sampling_spec = MemorySamplingSpec(
        sampling_family="uniform_interval",
        selection_mode="single_static",
        static_memory_strengths=weights,
    )

    sampled = sample_memory_strengths(
        problem=problem,
        case={"support_bits": tuple(range(problem.n_bits))},
        sampling_spec=sampling_spec,
        seed=5,
    )

    assert sampled == pytest.approx(weights)


def test_truncated_gaussian_clips_for_degenerate_interval_and_zero_sigma():
    problem = _toy_problem()
    case = {"support_bits": tuple(range(problem.n_bits))}
    point_spec = MemorySamplingSpec(
        sampling_family="truncated_gaussian",
        application_scope="all_bits",
        selection_mode="single_draw",
        low=0.13,
        high=0.13,
        mean=0.5,
        sigma=0.1,
    )
    point_weights = sample_memory_strengths(problem=problem, case=case, sampling_spec=point_spec, seed=11)
    assert point_weights == pytest.approx(np.full(problem.n_bits, 0.13))

    zero_sigma_spec = MemorySamplingSpec(
        sampling_family="truncated_gaussian",
        application_scope="all_bits",
        selection_mode="single_draw",
        low=-0.2,
        high=0.2,
        mean=0.5,
        sigma=0.0,
    )
    zero_sigma_weights = sample_memory_strengths(
        problem=problem,
        case=case,
        sampling_spec=zero_sigma_spec,
        seed=19,
    )
    assert zero_sigma_weights == pytest.approx(np.full(problem.n_bits, 0.2))


def test_k_draw_practical_prefers_success_then_speed_then_weight():
    chosen = _select_draw_row(
        [
            {"draw_idx": 0, "logical_success": False, "converged": True, "exact_recovery": False, "iterations": 1, "decoding_weight": 1},
            {"draw_idx": 1, "logical_success": True, "converged": False, "exact_recovery": False, "iterations": 9, "decoding_weight": 9},
            {"draw_idx": 2, "logical_success": True, "converged": True, "exact_recovery": False, "iterations": 10, "decoding_weight": 1},
        ],
        selection_mode="k_draw_practical",
    )

    assert int(chosen["draw_idx"]) == 1


def test_k_draw_oracle_counts_any_success_as_solved():
    chosen = _select_draw_row(
        [
            {"draw_idx": 0, "logical_success": False, "converged": True, "exact_recovery": True, "iterations": 1, "decoding_weight": 0},
            {"draw_idx": 1, "logical_success": True, "converged": False, "exact_recovery": False, "iterations": 12, "decoding_weight": 4},
            {"draw_idx": 2, "logical_success": True, "converged": True, "exact_recovery": True, "iterations": 13, "decoding_weight": 1},
        ],
        selection_mode="k_draw_oracle",
    )

    assert int(chosen["draw_idx"]) == 1


def test_heatmap_rows_and_best_points_are_reproducible():
    problem = _toy_problem()
    cases = enumerate_half_stabilizer_cases(problem, max_cases=2, shuffle_seed=3)
    artifact = evaluate_matched_support_heatmaps(
        problem=problem,
        cases=cases,
        max_iter=6,
        alpha=1.0,
        centers=[0.2, 0.4],
        widths=[0.4],
        base_seed=9,
        random_draws_per_point=1,
        application_scope="stabilizer_support",
        selection_mode="single_draw",
        logical_weight_penalty=4.0,
        convergence_penalty=1.0,
        iteration_penalty=0.05,
    )

    assert artifact["families"]["uniform_interval"]["trial_count"][0, 0] == len(cases)
    assert artifact["interval_lows"][0, 1] == pytest.approx(artifact["interval_lows"][0, 1])
    assert artifact["interval_lows"].shape == artifact["gaussian_sigmas"].shape

    rows_first = matched_support_heatmap_rows(artifact)
    rows_second = matched_support_heatmap_rows(artifact)
    assert rows_first == rows_second
    assert best_heatmap_rows(rows_first, topk=2) == best_heatmap_rows(rows_second, topk=2)


def test_gaussian_refinement_heatmap_smoke():
    problem = _toy_problem()
    cases = enumerate_half_stabilizer_cases(problem, max_cases=2, shuffle_seed=5)
    artifact = evaluate_gaussian_refinement_heatmap(
        problem=problem,
        cases=cases,
        max_iter=6,
        alpha=1.0,
        means=[0.1, 0.2],
        sigmas=[0.01, 0.05],
        low=-0.1,
        high=0.3,
        base_seed=21,
        random_draws_per_point=1,
        application_scope="stabilizer_support",
        selection_mode="single_draw",
        logical_weight_penalty=4.0,
        convergence_penalty=1.0,
        iteration_penalty=0.05,
    )

    rows = gaussian_refinement_heatmap_rows(artifact)
    assert len(rows) == 4
    assert rows[0]["low"] == pytest.approx(-0.1)
    assert rows[0]["high"] == pytest.approx(0.3)


def test_equivalence_report_exposes_bootstrap_intervals_and_verdict():
    candidate_rows = [
        {"logical_success_rate": 0.80, "convergence_rate": 0.70, "mean_iterations": 9.0},
        {"logical_success_rate": 0.82, "convergence_rate": 0.72, "mean_iterations": 8.5},
    ]
    baseline_rows = [
        {"logical_success_rate": 0.80, "convergence_rate": 0.70, "mean_iterations": 9.0},
        {"logical_success_rate": 0.81, "convergence_rate": 0.71, "mean_iterations": 8.6},
    ]

    report = evaluate_equivalence(
        candidate_rows=candidate_rows,
        baseline_rows=baseline_rows,
        thresholds=EquivalenceThresholds(bootstrap_samples=256, seed=4),
    )

    assert "lower" in report["logical_success_difference"]
    assert "upper" in report["convergence_difference"]
    assert "mean" in report["iteration_ratio"]
    assert isinstance(report["verdict"], bool)
