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

import relay_bp


def test_min_sum_trace_matches_decode_detailed(repetition_code_config):
    detectors = np.array([1, 1], dtype=np.uint8)

    decoder = relay_bp.MinSumBPDecoderF64(
        repetition_code_config["check_matrix"],
        error_priors=repetition_code_config["error_priors"],
        max_iter=10,
        gamma0=0.15,
    )
    expected = decoder.decode_detailed(detectors)

    tracer = relay_bp.MinSumBPDecoderTraceF64(
        repetition_code_config["check_matrix"],
        error_priors=repetition_code_config["error_priors"],
        max_iter=10,
        gamma0=0.15,
    )
    tracer.reset()

    trace_results = [tracer.snapshot(detectors)]
    result = trace_results[-1]
    while tracer.current_iteration < 10 and not result.success:
        result = tracer.run_iteration(detectors)
        trace_results.append(result)

    assert result.success == expected.success
    assert result.iterations == expected.iterations
    assert np.array_equal(result.decoding, expected.decoding)
    assert np.array_equal(result.decoded_detectors, expected.decoded_detectors)
    assert np.allclose(result.posterior_ratios, expected.posterior_ratios)
    assert trace_results[0].iterations == 0
    assert trace_results[-1].iterations == expected.iterations
