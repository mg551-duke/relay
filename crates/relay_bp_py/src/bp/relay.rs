// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
use std::sync::Arc;

use pyo3::prelude::*;

use crate::decoder::{get_sprs_bit_matrix_from_python, DecodeResult, DynDecoder};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use relay_bp::bp::min_sum::MinSumDecoderConfig;
use relay_bp::bp::relay::{
    GammaSampler, MessageMixBernoulliSampler, RelayDecoder, RelayDecoderConfig, StoppingCriterion,
};
use relay_bp::decoder::Bit;

macro_rules! create_bp_interface {
    ($name: ident, $type: ident) => {
        #[pyclass(extends=DynDecoder, subclass, module = "bp")]
        #[allow(dead_code)]
        pub struct $name {}

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (check_matrix, error_priors, alpha=None, alpha_iteration_scaling_factor=1.0, gamma0=0.1, c_damp=None, data_scale_value=None, max_data_value=None, explicit_edge_message_weights=None, pre_iter=80, num_sets=300,
                set_max_iter=60, gamma_dist_interval=(-0.24, 0.66), gamma_bernoulli=None, explicit_gammas=None, explicit_c_damp_messages=None, c_damp_dist_interval=None, message_mix_bernoulli=None, relay_posteriors=true, stop_nconv=1,
                stopping_criterion="nconv".to_string(), logging=false, seed=0))]
            #[allow(clippy::missing_transmute_annotations, clippy::too_many_arguments)]
            pub fn new(
                py: Python<'_>,
                check_matrix: &Bound<'_, PyAny>,
                error_priors: &Bound<'_, PyArray1<f64>>,
                alpha: Option<f64>,
                alpha_iteration_scaling_factor: f64,
                gamma0: Option<f64>,
                c_damp: Option<f64>,
                data_scale_value: Option<f64>,
                max_data_value: Option<f64>,
                explicit_edge_message_weights: Option<&Bound<'_, PyArray1<f64>>>,
                pre_iter: usize,
                num_sets: usize,
                set_max_iter: usize,
                gamma_dist_interval: (f64, f64),
                gamma_bernoulli: Option<(f64, f64, f64)>,
                explicit_gammas: Option<&Bound<'_, PyArray2<f64>>>,
                explicit_c_damp_messages: Option<&Bound<'_, PyArray2<f64>>>,
                c_damp_dist_interval: Option<(f64, f64)>,
                message_mix_bernoulli: Option<(f64, f64, f64, f64, f64, f64)>,
                relay_posteriors: bool,
                stop_nconv: usize,
                stopping_criterion: String,
                logging: bool,
                seed: u64,
            ) -> PyResult<(Self, DynDecoder)> {
                let min_sum_decoder = Self {};
                validate_damping_args(
                    c_damp,
                    explicit_c_damp_messages.is_some(),
                    c_damp_dist_interval,
                    message_mix_bernoulli.is_some(),
                )?;

                let min_sum_config = MinSumDecoderConfig {
                    error_priors: unsafe { error_priors.as_array() }.to_owned(),
                    max_iter: pre_iter, // pre_iter is equal to max_iter for a single bp run.
                    alpha,
                    alpha_iteration_scaling_factor,
                    c_damp,
                    explicit_c_damp_messages: None,
                    explicit_message_mix_fresh_coefficients: None,
                    explicit_message_mix_previous_coefficients: None,
                    explicit_edge_message_weights: explicit_edge_message_weights
                        .map(|weights| unsafe { weights.as_array() }.to_owned()),
                    gamma0,
                    data_scale_value,
                    max_data_value,
                    int_bits: None,
                    frac_bits: None
                };

                let stopping_criterion = match stopping_criterion.as_str() {
                    "pre_iter" => StoppingCriterion::PreIter,
                    "nconv" => StoppingCriterion::NConv {
                        stop_after: stop_nconv,
                    },
                    "all" => StoppingCriterion::All,
                    _ => StoppingCriterion::default(),
                };

                let relay_config = RelayDecoderConfig {
                    pre_iter,
                    num_sets,
                    set_max_iter,
                    gamma_dist_interval,
                    gamma_sampler: gamma_sampler_from_args(gamma_dist_interval, gamma_bernoulli)?,
                    explicit_gammas: explicit_gammas
                        .map(|explicit_gammas| unsafe { explicit_gammas.as_array() }.to_owned()),
                    explicit_c_damp_messages: explicit_c_damp_messages
                        .map(|explicit_c_damp_messages| unsafe { explicit_c_damp_messages.as_array() }.to_owned()),
                    c_damp_dist_interval,
                    message_mix_bernoulli: message_mix_sampler_from_args(message_mix_bernoulli)?,
                    relay_posteriors,
                    stopping_criterion,
                    logging,
                    seed,
                };

                let inner_decoder = RelayDecoder::<$type>::new(
                    Arc::new(get_sprs_bit_matrix_from_python(py, check_matrix)?),
                    Arc::new(min_sum_config),
                    Arc::new(relay_config),
                );

                let dyn_decoder = DynDecoder(Box::new(inner_decoder));
                Ok((min_sum_decoder, dyn_decoder))
            }

            pub fn decode<'py>(
                mut self_: PyRefMut<'_, Self>,
                py: Python<'py>,
                detectors: PyReadonlyArray1<'_, Bit>,
            ) -> Bound<'py, PyArray1<Bit>> {
                self_
                    .as_super()
                    .inner()
                    .decode(detectors.as_array())
                    .into_pyarray(py)
            }

            pub fn decode_detailed(
                mut self_: PyRefMut<'_, Self>,
                detectors: PyReadonlyArray1<'_, Bit>,
            ) -> DecodeResult {
                DecodeResult::new(
                    self_
                        .as_super()
                        .inner()
                        .decode_detailed(detectors.as_array()),
                )
            }

            pub fn decode_batch<'py>(
                mut self_: PyRefMut<'_, Self>,
                py: Python<'py>,
                detectors: PyReadonlyArray2<'_, Bit>,
            ) -> Bound<'py, PyArray2<Bit>> {
                self_
                    .as_super()
                    .inner()
                    .decode_batch(detectors.as_array())
                    .into_pyarray(py)
            }

            pub fn decode_detailed_batch(
                mut self_: PyRefMut<'_, Self>,
                detectors: PyReadonlyArray2<'_, Bit>,
            ) -> Vec<DecodeResult> {
                self_
                    .as_super()
                    .inner()
                    .decode_detailed_batch(detectors.as_array())
                    .into_iter()
                    .map(|result| DecodeResult::new(result))
                    .collect()
            }
        }
    };
}

create_bp_interface!(RelayDecoderF32, f32);
create_bp_interface!(RelayDecoderF64, f64);
create_bp_interface!(RelayDecoderI32, i32);
create_bp_interface!(RelayDecoderI64, i64);

fn gamma_sampler_from_args(
    gamma_dist_interval: (f64, f64),
    gamma_bernoulli: Option<(f64, f64, f64)>,
) -> PyResult<GammaSampler> {
    if let Some((negative, positive, p_positive)) = gamma_bernoulli {
        return GammaSampler::bernoulli_two_point(negative, positive, p_positive)
            .map_err(pyo3::exceptions::PyValueError::new_err);
    }
    let sampler = GammaSampler::uniform_interval(gamma_dist_interval);
    sampler
        .validate()
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(sampler)
}

fn validate_damping_args(
    c_damp: Option<f64>,
    has_explicit_c_damp_messages: bool,
    c_damp_dist_interval: Option<(f64, f64)>,
    has_message_mix_bernoulli: bool,
) -> PyResult<()> {
    let specified = (c_damp.is_some() as usize)
        + (has_explicit_c_damp_messages as usize)
        + (c_damp_dist_interval.is_some() as usize)
        + (has_message_mix_bernoulli as usize);
    if specified > 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "message_mix_bernoulli cannot be combined with c_damp, explicit_c_damp_messages, or c_damp_dist_interval.",
        ));
    }
    if let Some(value) = c_damp {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "c_damp must be finite and in [0, 1].",
            ));
        }
    }
    Ok(())
}

fn message_mix_sampler_from_args(
    message_mix_bernoulli: Option<(f64, f64, f64, f64, f64, f64)>,
) -> PyResult<Option<MessageMixBernoulliSampler>> {
    match message_mix_bernoulli {
        Some((
            fresh_negative,
            fresh_positive,
            fresh_p_positive,
            previous_negative,
            previous_positive,
            previous_p_positive,
        )) => MessageMixBernoulliSampler::new(
            fresh_negative,
            fresh_positive,
            fresh_p_positive,
            previous_negative,
            previous_positive,
            previous_p_positive,
        )
        .map(Some)
        .map_err(pyo3::exceptions::PyValueError::new_err),
        None => Ok(None),
    }
}
