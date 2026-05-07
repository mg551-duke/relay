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
import pytest

import relay_bp


def test_decode_detailed(repetition_code_config):
    repetition_code_config.pop("max_iter", None)
    decoder = relay_bp.RelayDecoderF32(
        **repetition_code_config,
        pre_iter=120,
        num_sets=40,
        set_max_iter=60,
        gamma_dist_interval=(-0.24, 0.66),
        explicit_gammas=None,
        stop_nconv=3,
    )

    detectors = np.array([1, 1], dtype=np.uint8)

    result = decoder.decode_detailed(detectors)
    assert result.success
    assert np.all(result.decoding == np.array([0, 1, 0]))
    assert result.iterations <= 120
    assert result.max_iter == 120


def test_decode_detailed_batch(repetition_code_config):
    repetition_code_config.pop("max_iter", None)
    decoder = relay_bp.RelayDecoderF32(
        **repetition_code_config,
        pre_iter=120,
        num_sets=40,
        set_max_iter=60,
        gamma_dist_interval=(-0.24, 0.66),
        explicit_gammas=None,
        stop_nconv=3,
    )

    detectors = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.uint8)

    results = decoder.decode_detailed_batch(detectors)

    result0 = results[0]
    assert result0.success
    assert np.all(result0.decoding == np.array([1, 0, 0]))

    result1 = results[1]
    assert result1.success
    assert np.all(result1.decoding == np.array([0, 1, 0]))

    result2 = results[2]
    assert result2.success
    assert np.all(result2.decoding == np.array([0, 0, 1]))


def test_decode_detailed_with_disordered_edge_damping_interval(repetition_code_config):
    repetition_code_config.pop("max_iter", None)
    decoder = relay_bp.RelayDecoderF64(
        **repetition_code_config,
        pre_iter=20,
        num_sets=2,
        set_max_iter=10,
        gamma0=0.15,
        c_damp_dist_interval=(0.5, 1.0),
        seed=7,
    )

    result = decoder.decode_detailed(np.array([1, 1], dtype=np.uint8))

    assert result.success
    assert result.damping_mode == "edge_message"
    assert result.damping_min is not None
    assert result.damping_max is not None
    assert 0.5 <= result.damping_min <= result.damping_max <= 1.0


def test_decode_detailed_with_scalar_damping(repetition_code_config):
    repetition_code_config.pop("max_iter", None)
    decoder = relay_bp.RelayDecoderF64(
        **repetition_code_config,
        pre_iter=20,
        num_sets=2,
        set_max_iter=10,
        gamma0=0.15,
        c_damp=0.9,
        seed=7,
    )

    result = decoder.decode_detailed(np.array([1, 1], dtype=np.uint8))

    assert result.success
    assert result.damping_mode == "scalar"
    assert result.damping_min == pytest.approx(0.9)
    assert result.damping_max == pytest.approx(0.9)


def test_decode_detailed_with_explicit_edge_damping_messages(repetition_code_config):
    repetition_code_config.pop("max_iter", None)
    nnz = repetition_code_config["check_matrix"].nnz
    explicit = np.full((3, nnz), 0.75, dtype=np.float64)
    decoder = relay_bp.RelayDecoderF64(
        **repetition_code_config,
        pre_iter=20,
        num_sets=2,
        set_max_iter=10,
        gamma0=0.15,
        explicit_c_damp_messages=explicit,
    )

    result = decoder.decode_detailed(np.array([1, 1], dtype=np.uint8))

    assert result.success
    assert result.damping_mode == "edge_message"
    assert result.damping_min == pytest.approx(0.75)
    assert result.damping_max == pytest.approx(0.75)


def test_message_mix_bernoulli_is_applied_like_fresh_scaling(repetition_code_config):
    repetition_code_config.pop("max_iter", None)
    detectors = np.array([1, 1], dtype=np.uint8)

    mixed = relay_bp.RelayDecoderF64(
        **repetition_code_config,
        pre_iter=1,
        num_sets=0,
        set_max_iter=1,
        gamma0=None,
        message_mix_bernoulli=(-1.0, 0.5, 1.0, -1.0, 0.0, 1.0),
        stopping_criterion="pre_iter",
        seed=11,
    ).decode_detailed(detectors)
    reference_config = dict(repetition_code_config)
    reference_config["alpha"] = 0.5
    reference = relay_bp.RelayDecoderF64(
        **reference_config,
        pre_iter=1,
        num_sets=0,
        set_max_iter=1,
        gamma0=None,
        stopping_criterion="pre_iter",
        seed=11,
    ).decode_detailed(detectors)

    assert mixed.damping_mode == "message_mix"
    assert mixed.damping_min == pytest.approx(0.0)
    assert mixed.damping_max == pytest.approx(0.5)
    np.testing.assert_allclose(mixed.posterior_ratios, reference.posterior_ratios)


def test_message_mix_rejects_existing_damping_sources(repetition_code_config):
    repetition_code_config.pop("max_iter", None)
    message_mix = (-1.0, 0.5, 1.0, -1.0, 0.0, 1.0)

    with pytest.raises(ValueError, match="message_mix_bernoulli"):
        relay_bp.RelayDecoderF64(
            **repetition_code_config,
            c_damp=0.9,
            message_mix_bernoulli=message_mix,
        )

    explicit = np.full((2, repetition_code_config["check_matrix"].nnz), 0.75)
    with pytest.raises(ValueError, match="message_mix_bernoulli"):
        relay_bp.RelayDecoderF64(
            **repetition_code_config,
            explicit_c_damp_messages=explicit,
            message_mix_bernoulli=message_mix,
        )
