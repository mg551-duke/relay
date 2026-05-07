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
use relay_bp::bp::min_sum::{MinSumBPDecoder, MinSumDecoderConfig};
use relay_bp::decoder::{Bit, Decoder};

#[allow(clippy::too_many_arguments)]
fn build_min_sum_config(
    error_priors: &Bound<'_, PyArray1<f64>>,
    max_iter: usize,
    alpha: Option<f64>,
    alpha_iteration_scaling_factor: f64,
    c_damp: Option<f64>,
    explicit_c_damp_messages: Option<&Bound<'_, PyArray1<f64>>>,
    explicit_edge_message_weights: Option<&Bound<'_, PyArray1<f64>>>,
    gamma0: Option<f64>,
    data_scale_value: Option<f64>,
    max_data_value: Option<f64>,
    int_bits: Option<isize>,
    frac_bits: Option<isize>,
) -> MinSumDecoderConfig {
    MinSumDecoderConfig {
        error_priors: unsafe { error_priors.as_array() }.to_owned(),
        max_iter,
        alpha,
        alpha_iteration_scaling_factor,
        c_damp,
        explicit_c_damp_messages: explicit_c_damp_messages
            .map(|values| unsafe { values.as_array() }.to_owned()),
        explicit_message_mix_fresh_coefficients: None,
        explicit_message_mix_previous_coefficients: None,
        explicit_edge_message_weights: explicit_edge_message_weights
            .map(|values| unsafe { values.as_array() }.to_owned()),
        gamma0,
        data_scale_value,
        max_data_value,
        int_bits,
        frac_bits,
    }
}

macro_rules! create_bp_interface {
    ($name: ident, $type: ident) => {
        #[pyclass(extends=DynDecoder, subclass, module = "bp")]
        #[allow(dead_code)]
        pub struct $name {}

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (check_matrix, error_priors, max_iter=200, alpha=None, alpha_iteration_scaling_factor=1.0, c_damp=None, explicit_c_damp_messages=None, explicit_edge_message_weights=None, gamma0=None, data_scale_value=None, max_data_value=None, int_bits=None, frac_bits=None))]
            #[allow(clippy::missing_transmute_annotations, clippy::too_many_arguments)]
            pub fn new(
                py: Python<'_>,
                check_matrix: &Bound<'_, PyAny>,
                error_priors: &Bound<'_, PyArray1<f64>>,
                max_iter: usize,
                alpha: Option<f64>,
                alpha_iteration_scaling_factor: f64,
                c_damp: Option<f64>,
                explicit_c_damp_messages: Option<&Bound<'_, PyArray1<f64>>>,
                explicit_edge_message_weights: Option<&Bound<'_, PyArray1<f64>>>,
                gamma0: Option<f64>,
                data_scale_value: Option<f64>,
                max_data_value: Option<f64>,
                int_bits: Option<isize>,
                frac_bits: Option<isize>,
            ) -> PyResult<(Self, DynDecoder)> {
                let min_sum_decoder = Self {};

                let config = build_min_sum_config(
                    error_priors,
                    max_iter,
                    alpha,
                    alpha_iteration_scaling_factor,
                    c_damp,
                    explicit_c_damp_messages,
                    explicit_edge_message_weights,
                    gamma0,
                    data_scale_value,
                    max_data_value,
                    int_bits,
                    frac_bits,
                );


                let inner_decoder = MinSumBPDecoder::<$type>::new(
                    Arc::new(get_sprs_bit_matrix_from_python(py, check_matrix)?),
                    Arc::new(config),
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

create_bp_interface!(MinSumBPDecoderF32, f32);
create_bp_interface!(MinSumBPDecoderF64, f64);

create_bp_interface!(MinSumBPDecoderI8, i8);
create_bp_interface!(MinSumBPDecoderI16, i16);
create_bp_interface!(MinSumBPDecoderI32, i32);
create_bp_interface!(MinSumBPDecoderI64, i64);

#[pyclass(module = "bp")]
#[allow(dead_code)]
pub struct MinSumBPDecoderTraceF64 {
    inner_decoder: MinSumBPDecoder<f64>,
}

#[pymethods]
impl MinSumBPDecoderTraceF64 {
    #[new]
    #[pyo3(signature = (check_matrix, error_priors, max_iter=200, alpha=None, alpha_iteration_scaling_factor=1.0, c_damp=None, explicit_c_damp_messages=None, explicit_edge_message_weights=None, gamma0=None, data_scale_value=None, max_data_value=None, int_bits=None, frac_bits=None))]
    #[allow(clippy::missing_transmute_annotations, clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
        check_matrix: &Bound<'_, PyAny>,
        error_priors: &Bound<'_, PyArray1<f64>>,
        max_iter: usize,
        alpha: Option<f64>,
        alpha_iteration_scaling_factor: f64,
        c_damp: Option<f64>,
        explicit_c_damp_messages: Option<&Bound<'_, PyArray1<f64>>>,
        explicit_edge_message_weights: Option<&Bound<'_, PyArray1<f64>>>,
        gamma0: Option<f64>,
        data_scale_value: Option<f64>,
        max_data_value: Option<f64>,
        int_bits: Option<isize>,
        frac_bits: Option<isize>,
    ) -> PyResult<Self> {
        let config = build_min_sum_config(
            error_priors,
            max_iter,
            alpha,
            alpha_iteration_scaling_factor,
            c_damp,
            explicit_c_damp_messages,
            explicit_edge_message_weights,
            gamma0,
            data_scale_value,
            max_data_value,
            int_bits,
            frac_bits,
        );

        let inner_decoder = MinSumBPDecoder::<f64>::new(
            Arc::new(get_sprs_bit_matrix_from_python(py, check_matrix)?),
            Arc::new(config),
        );

        Ok(Self { inner_decoder })
    }

    pub fn reset(&mut self) {
        self.inner_decoder.initialize_decoder();
    }

    pub fn set_memory_strengths(&mut self, memory_strengths: PyReadonlyArray1<'_, f64>) {
        self.inner_decoder
            .set_memory_strengths_f64(memory_strengths.as_array().to_owned());
    }

    pub fn set_explicit_c_damp_messages(
        &mut self,
        explicit_c_damp_messages: PyReadonlyArray1<'_, f64>,
    ) {
        self.inner_decoder
            .set_explicit_c_damp_messages_f64(explicit_c_damp_messages.as_array().to_owned());
    }

    pub fn clear_explicit_c_damp_messages(&mut self) {
        self.inner_decoder.clear_explicit_c_damp_messages();
    }

    pub fn set_explicit_message_mix_coefficients(
        &mut self,
        fresh_coefficients: PyReadonlyArray1<'_, f64>,
        previous_coefficients: PyReadonlyArray1<'_, f64>,
    ) {
        self.inner_decoder
            .set_explicit_message_mix_coefficients_f64(
                fresh_coefficients.as_array().to_owned(),
                previous_coefficients.as_array().to_owned(),
            );
    }

    pub fn clear_explicit_message_mix_coefficients(&mut self) {
        self.inner_decoder.clear_explicit_message_mix_coefficients();
    }

    pub fn set_explicit_edge_message_weights(
        &mut self,
        explicit_edge_message_weights: PyReadonlyArray1<'_, f64>,
    ) {
        self.inner_decoder.set_explicit_edge_message_weights_f64(
            explicit_edge_message_weights.as_array().to_owned(),
        );
    }

    pub fn clear_explicit_edge_message_weights(&mut self) {
        self.inner_decoder.clear_explicit_edge_message_weights();
    }

    pub fn set_log_prior_ratios(&mut self, log_prior_ratios: PyReadonlyArray1<'_, f64>) {
        self.inner_decoder
            .set_log_prior_ratio_f64(log_prior_ratios.as_array().to_owned());
    }

    pub fn snapshot(&mut self, detectors: PyReadonlyArray1<'_, Bit>) -> DecodeResult {
        let decoded_detectors = self.inner_decoder.compute_decoded_detectors();
        let success = self
            .inner_decoder
            .check_convergence(detectors.as_array(), decoded_detectors.view());
        DecodeResult::new(self.inner_decoder.build_result(
            success,
            decoded_detectors,
            self.inner_decoder.config.max_iter,
        ))
    }

    pub fn run_iteration(&mut self, detectors: PyReadonlyArray1<'_, Bit>) -> DecodeResult {
        self.inner_decoder.run_iteration(detectors.as_array());
        self.inner_decoder.current_iteration += 1;
        self.snapshot(detectors)
    }

    pub fn decode_detailed(&mut self, detectors: PyReadonlyArray1<'_, Bit>) -> DecodeResult {
        DecodeResult::new(self.inner_decoder.decode_detailed(detectors.as_array()))
    }

    #[getter]
    pub fn current_iteration(&self) -> usize {
        self.inner_decoder.current_iteration
    }
}
