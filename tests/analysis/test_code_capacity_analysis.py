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

from relay_bp.analysis import (
    build_edge_damping_messages,
    build_edge_message_weights,
    default_export_code_paths,
    enumerate_half_stabilizer_cases,
    evaluate_edge_weight_heatmap,
    load_code_capacity_problem,
    make_min_sum_tracer,
)
from relay_bp.analysis.code_capacity_common import CodeCapacityProblem
from relay_bp.analysis.code_capacity_damping import (
    evaluate_decoder_family,
    evaluate_interval_heatmap,
    run_trace_suite,
)


def _toy_problem() -> CodeCapacityProblem:
    hx = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=np.uint8,
    )
    hz = csc_matrix(
        np.array(
            [
                [1, 1, 0],
                [0, 1, 1],
            ],
            dtype=np.uint8,
        )
    )
    lz = np.array([[1, 0, 1]], dtype=np.uint8)
    return CodeCapacityProblem(
        path=Path("toy_code.npz"),
        hx=hx,
        hz=hz,
        lz=lz,
        error_priors=np.full(3, 0.1, dtype=np.float64),
        metadata={"name": "toy"},
    )


def test_default_export_paths_exist():
    repo_root = Path(__file__).resolve().parents[2]
    paths = default_export_code_paths(repo_root)
    assert set(paths) == {"gross", "two_gross", "surface13"}
    for path in paths.values():
        assert path.exists()


def test_build_edge_damping_messages_support_only_defaults_outside_edges_to_one():
    check_matrix = csc_matrix(
        np.array(
            [
                [1, 1, 0],
                [0, 1, 1],
            ],
            dtype=np.uint8,
        )
    )

    messages = build_edge_damping_messages(
        check_matrix,
        interval=(0.5, 0.5),
        coeff_seed=7,
        support_bits=(1,),
    )

    assert np.allclose(messages, np.array([1.0, 0.5, 0.5, 1.0], dtype=np.float64))


def test_build_edge_message_weights_support_only_defaults_outside_edges_to_one():
    check_matrix = csc_matrix(
        np.array(
            [
                [1, 1, 0],
                [0, 1, 1],
            ],
            dtype=np.uint8,
        )
    )

    weights = build_edge_message_weights(
        check_matrix,
        interval=(-0.25, -0.25),
        coeff_seed=7,
        support_bits=(1,),
    )

    assert np.allclose(weights, np.array([1.0, -0.25, -0.25, 1.0], dtype=np.float64))


def test_interval_heatmap_decoder_family_and_trace_smoke():
    problem = _toy_problem()
    cases = enumerate_half_stabilizer_cases(problem, max_cases=2, shuffle_seed=3)

    heatmap = evaluate_interval_heatmap(
        problem=problem,
        cases=cases,
        max_iter=5,
        alpha=1.0,
        centers=[0.75],
        widths=[0.25],
        base_seed=11,
        random_draws_per_point=1,
        support_only=True,
    )
    assert heatmap["logical_success_rate"].shape == (1, 1)
    assert int(heatmap["trial_count"][0, 0]) == len(cases)

    family = evaluate_decoder_family(
        problem=problem,
        cases=cases,
        max_iter=5,
        alpha=1.0,
        same_damping_value=0.8,
        random_interval=(0.5, 1.0),
        base_seed=5,
        random_seed_offsets=[0],
        support_only=True,
    )
    assert set(family) == {"bp", "damp_same", "damp_random_interval"}
    assert len(family["bp"]) == len(cases)
    assert len(family["damp_same"]) == len(cases)
    assert len(family["damp_random_interval"]) == len(cases)

    trace_artifact = run_trace_suite(
        problem=problem,
        core_seed=9,
        max_iter=5,
        alpha=1.0,
        same_damping_value=0.8,
        random_interval=(0.5, 1.0),
        random_seed_offsets=[0],
        support_only=True,
    )
    assert len(trace_artifact["runs"]) == 3
    assert all("posterior_trace" in run for run in trace_artifact["runs"])


def test_edge_weight_heatmap_smoke():
    problem = _toy_problem()
    cases = enumerate_half_stabilizer_cases(problem, max_cases=2, shuffle_seed=3)
    tracer = make_min_sum_tracer(problem, max_iter=5, alpha=1.0, gamma0=None)
    if not hasattr(tracer, "set_explicit_edge_message_weights"):
        pytest.skip("editable relay_bp extension is stale; explicit edge-message-weight bindings are unavailable")

    artifact = evaluate_edge_weight_heatmap(
        problem=problem,
        cases=cases,
        max_iter=5,
        alpha=1.0,
        centers=[0.0],
        widths=[0.5],
        base_seed=11,
        random_draws_per_point=1,
        support_only=True,
    )
    assert artifact["logical_success_rate"].shape == (1, 1)
    assert int(artifact["trial_count"][0, 0]) == len(cases)
    assert artifact["memory_interval"] is None
    assert artifact["damping_interval"] is None

    combined_artifact = evaluate_edge_weight_heatmap(
        problem=problem,
        cases=cases,
        max_iter=5,
        alpha=1.0,
        centers=[0.0],
        widths=[0.5],
        base_seed=11,
        random_draws_per_point=1,
        support_only=True,
        memory_interval=(-0.1, 0.1),
        damping_interval=(0.85, 0.95),
    )
    assert combined_artifact["logical_success_rate"].shape == (1, 1)
    assert combined_artifact["memory_interval"] == (-0.1, 0.1)
    assert combined_artifact["damping_interval"] == (0.85, 0.95)


def test_load_code_capacity_problem_reads_exported_npz():
    repo_root = Path(__file__).resolve().parents[2]
    problem = load_code_capacity_problem(default_export_code_paths(repo_root)["surface13"])
    assert problem.hx.shape[0] > 0
    assert problem.hz.shape[1] == problem.n_bits
    assert problem.lz.shape[1] == problem.n_bits
    assert problem.error_priors.shape == (problem.n_bits,)
