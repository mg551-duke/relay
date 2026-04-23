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
from functools import lru_cache

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from relay_bp.analysis import (
    audit_half_null_complement_symmetry,
    audit_half_null_posteriors,
    compute_half_null_pair_posteriors,
    default_export_code_paths,
    enumerate_half_null_cases,
    load_code_capacity_problem,
    pair_half_null_cases,
    summarize_half_null_pair_posteriors,
    trace_half_null_pair,
)
from relay_bp.analysis.code_capacity_common import CodeCapacityProblem


def _synthetic_problem(error_priors: np.ndarray) -> CodeCapacityProblem:
    return CodeCapacityProblem(
        path=Path("synthetic_half_null.npz"),
        hx=np.zeros((1, 4), dtype=np.uint8),
        hz=csc_matrix(np.zeros((1, 4), dtype=np.uint8)),
        lz=np.zeros((1, 4), dtype=np.uint8),
        error_priors=np.asarray(error_priors, dtype=np.float64),
        metadata={"Hx_semantics": "heuristic_low_weight_independent_nulls_in_kernel_of_[Hz;Lz]"},
    )


@lru_cache(maxsize=None)
def _projected_problem_and_pairs(code_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    problem = load_code_capacity_problem(default_export_code_paths(repo_root)[code_name])
    cases = enumerate_half_null_cases(problem)
    pairs = pair_half_null_cases(cases)
    return problem, pairs


def test_synthetic_uniform_equal_size_halves_have_zero_posterior_gap():
    problem = _synthetic_problem(np.full(4, 0.1, dtype=np.float64))
    error_a = np.array([1, 1, 0, 0], dtype=np.uint8)
    error_b = np.array([0, 0, 1, 1], dtype=np.uint8)

    pair_rows = compute_half_null_pair_posteriors(
        problem=problem,
        pairs=[
            {
                "pair_id": 0,
                "row_idx": 0,
                "row_weight": 4,
                "support_bits": (0, 1, 2, 3),
                "case_a": {"flipped_bits": (0, 1), "error": error_a, "syndrome": np.zeros(1, dtype=np.uint8)},
                "case_b": {"flipped_bits": (2, 3), "error": error_b, "syndrome": np.zeros(1, dtype=np.uint8)},
            }
        ],
    )
    row = pair_rows[0]

    assert row["posterior_gap_log_odds"] == 0.0
    assert row["pair_class"] == "exact_equal"
    assert row["half_a_log_prob"] == pytest.approx(row["half_b_log_prob"])


def test_synthetic_nonuniform_halves_have_expected_nonzero_posterior_gap():
    problem = _synthetic_problem(np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float64))
    error_a = np.array([1, 1, 0, 0], dtype=np.uint8)
    error_b = np.array([0, 0, 1, 1], dtype=np.uint8)

    pair_rows = compute_half_null_pair_posteriors(
        problem=problem,
        pairs=[
            {
                "pair_id": 0,
                "row_idx": 0,
                "row_weight": 4,
                "support_bits": (0, 1, 2, 3),
                "case_a": {"flipped_bits": (0, 1), "error": error_a, "syndrome": np.zeros(1, dtype=np.uint8)},
                "case_b": {"flipped_bits": (2, 3), "error": error_b, "syndrome": np.zeros(1, dtype=np.uint8)},
            }
        ],
    )
    row = pair_rows[0]
    expected_gap = abs(
        2.0 * np.log(0.1 / 0.9)
        - 2.0 * np.log(0.2 / 0.8)
    )

    assert row["posterior_gap_log_odds"] == pytest.approx(expected_gap)
    assert row["pair_class"] == "strong_bias"
    assert row["half_a_log_prob"] != row["half_b_log_prob"]


def test_circuit_half_null_complements_match_decoder_input_and_output():
    problem, pairs = _projected_problem_and_pairs("surface13")

    audit = audit_half_null_complement_symmetry(
        problem=problem,
        pairs=pairs,
        max_iter=20,
        alpha=1.0,
        max_pairs=4,
    )

    assert audit["summary"]["num_pairs"] == 4
    assert audit["summary"]["syndrome_mismatches"] == 0
    assert audit["summary"]["logical_signature_mismatches"] == 0
    assert audit["summary"]["decoder_output_mismatches"] == 0
    assert audit["summary"]["logical_success_mismatches"] == 0
    assert audit["summary"]["residual_relation_failures"] == 0


def test_circuit_half_null_trace_matches_between_complements():
    problem, pairs = _projected_problem_and_pairs("surface13")

    trace = trace_half_null_pair(problem=problem, pair=pairs[0], max_iter=20, alpha=1.0)

    assert trace["same_iteration_grid"]
    assert trace["same_posterior_trace"]
    assert trace["same_decoding"]
    assert trace["same_logical_success"]


def test_projected_exports_have_expected_exact_equal_pair_counts():
    expected_exact_counts = {
        "gross": 56,
        "two_gross": 18,
        "surface13": 122,
    }

    for code_name, exact_count in expected_exact_counts.items():
        problem, pairs = _projected_problem_and_pairs(code_name)
        pair_rows = compute_half_null_pair_posteriors(problem, pairs)
        posterior_summary = summarize_half_null_pair_posteriors(pair_rows)
        symmetry_audit = audit_half_null_complement_symmetry(
            problem=problem,
            pairs=pairs,
            max_iter=20,
            alpha=1.0,
            max_pairs=8,
        )

        assert posterior_summary["exact_equal"]["num_pairs"] == exact_count
        assert posterior_summary["exact_equal"]["num_pairs"] > 0
        assert symmetry_audit["summary"]["syndrome_mismatches"] == 0
        assert symmetry_audit["summary"]["logical_signature_mismatches"] == 0


def test_posterior_audit_reports_class_tables_and_decoded_match_counts():
    problem, pairs = _projected_problem_and_pairs("surface13")

    audit = audit_half_null_posteriors(
        problem=problem,
        pairs=pairs,
        max_iter=20,
        alpha=1.0,
        max_pairs_per_class=2,
    )

    assert audit["pair_class_counts"]["exact_equal"]["num_pairs"] == 122
    assert set(audit["decoded_matches_counts"]) == {"half_a", "half_b", "neither", "both"}
    assert set(audit["pair_class_summary"]) == {
        "exact_equal",
        "near_equal",
        "mild_bias",
        "biased",
        "strong_bias",
    }
