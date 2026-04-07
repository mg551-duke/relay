# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from scipy.sparse import csc_matrix

import relay_bp


def test_bpgd_decode_detailed_no_fallback(repetition_code_config):
    decoder = relay_bp.BPGDDecoderF64(
        repetition_code_config["check_matrix"],
        error_priors=repetition_code_config["error_priors"],
        r_low=5,
        t_low=3,
        r_high=1000,
        t_high=5,
        fallback_kind="none",
    )

    result = decoder.decode_detailed(np.array([1, 1], dtype=np.uint8))

    assert result.success
    assert np.all(result.decoding == np.array([0, 1, 0], dtype=np.uint8))
    assert result.extra_kind == "bpgd"
    assert result.fallback_used is False
    assert result.converged_stage_index == 0
    assert list(result.stage_names) == ["BPGD_low"]
    assert list(result.stage_converged) == [True]
    assert list(result.stage_iterations) == [result.iterations]


def test_bpgd_decode_detailed_with_relay_fallback(repetition_code_config):
    decoder = relay_bp.BPGDDecoderF64(
        repetition_code_config["check_matrix"],
        error_priors=repetition_code_config["error_priors"],
        r_low=0,
        t_low=1,
        r_high=0,
        t_high=1,
        fallback_kind="relay-bp",
        fallback_pre_iter=40,
        fallback_num_sets=10,
        fallback_set_max_iter=20,
    )

    result = decoder.decode_detailed(np.array([1, 1], dtype=np.uint8))

    assert result.success
    assert np.all(result.decoding == np.array([0, 1, 0], dtype=np.uint8))
    assert result.extra_kind == "bpgd"
    assert result.fallback_used is True
    assert result.converged_stage_index == 2
    assert list(result.stage_names) == ["BPGD_low", "BPGD_high", "relay-bp"]
    assert list(result.stage_converged) == [False, False, True]
    assert list(result.stage_iterations)[:2] == [0, 0]
    assert list(result.stage_iterations)[2] > 0
    assert sum(result.stage_iterations) == result.iterations


def test_bpgd_decode_detailed_with_pre_iter_and_initial_high_decimation(
    repetition_code_config,
):
    decoder = relay_bp.BPGDDecoderF64(
        repetition_code_config["check_matrix"],
        error_priors=repetition_code_config["error_priors"],
        pre_iter=1,
        initial_decimation_percentage=50.0,
        r_low=0,
        t_low=1,
        r_high=0,
        t_high=1,
        fallback_kind="relay-bp",
        fallback_pre_iter=40,
        fallback_num_sets=10,
        fallback_set_max_iter=20,
    )

    result = decoder.decode_detailed(np.array([1, 1], dtype=np.uint8))

    assert result.success
    assert result.extra_kind == "bpgd"
    assert result.fallback_used is True
    assert list(result.stage_names) == ["BPGD_low", "BPGD_high", "relay-bp"]
    assert list(result.stage_iterations)[:2] == [1, 1]
    assert list(result.stage_converged) == [False, False, True]
    assert sum(result.stage_iterations) == result.iterations


def test_bpgd_high_phase_stops_when_no_variables_left_to_decimate():
    check_matrix = csc_matrix(
        np.array(
            [
                [1, 0],
                [1, 0],
            ],
            dtype=np.uint8,
        )
    )
    decoder = relay_bp.BPGDDecoderF64(
        check_matrix,
        error_priors=np.zeros(2, dtype=np.float64),
        r_low=2,
        t_low=1,
        r_high=100,
        t_high=1,
        fallback_kind="none",
    )

    result = decoder.decode_detailed(np.array([0, 1], dtype=np.uint8))

    assert not result.success
    assert list(result.stage_names) == ["BPGD_low", "BPGD_high"]
    assert list(result.stage_converged) == [False, False]
    assert list(result.stage_iterations) == [2, 1]
    assert result.iterations == 3
