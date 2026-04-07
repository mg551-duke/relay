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

use crate::decoder::{get_sprs_bit_matrix_from_python, DecodeResult, DynDecoder};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use relay_bp::bp::bpgd::{BPGDDecoder, BPGDDecoderConfig, BPGDFallbackConfig};
use relay_bp::bp::relay::{RelayDecoderConfig, StoppingCriterion};
use relay_bp::decoder::Bit;

#[pyclass(extends=DynDecoder, subclass, module = "bp")]
#[allow(dead_code)]
pub struct BPGDDecoderF64 {}

#[pymethods]
impl BPGDDecoderF64 {
    #[new]
    #[pyo3(signature = (
        check_matrix,
        error_priors,
        pre_iter=0,
        initial_decimation_percentage=0.0,
        r_low=5,
        t_low=3,
        r_high=1000,
        t_high=5,
        alpha=None,
        alpha_iteration_scaling_factor=1.0,
        c_damp=None,
        fallback_kind="none".to_string(),
        fallback_gamma0=0.1,
        fallback_pre_iter=80,
        fallback_num_sets=60,
        fallback_set_max_iter=60,
        fallback_gamma_dist_interval=(-0.24, 0.66),
        fallback_explicit_gammas=None,
        fallback_stop_nconv=5,
        fallback_stopping_criterion="nconv".to_string(),
        fallback_logging=false,
        fallback_seed=0,
    ))]
    #[allow(clippy::missing_transmute_annotations, clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
        check_matrix: &Bound<'_, PyAny>,
        error_priors: &Bound<'_, PyArray1<f64>>,
        pre_iter: usize,
        initial_decimation_percentage: f64,
        r_low: usize,
        t_low: usize,
        r_high: usize,
        t_high: usize,
        alpha: Option<f64>,
        alpha_iteration_scaling_factor: f64,
        c_damp: Option<f64>,
        fallback_kind: String,
        fallback_gamma0: f64,
        fallback_pre_iter: usize,
        fallback_num_sets: usize,
        fallback_set_max_iter: usize,
        fallback_gamma_dist_interval: (f64, f64),
        fallback_explicit_gammas: Option<&Bound<'_, PyArray2<f64>>>,
        fallback_stop_nconv: usize,
        fallback_stopping_criterion: String,
        fallback_logging: bool,
        fallback_seed: u64,
    ) -> PyResult<(Self, DynDecoder)> {
        let bpgd_decoder = Self {};

        let fallback = match fallback_kind.as_str() {
            "none" => BPGDFallbackConfig::None,
            "relay_BP" | "relay-bp" | "relay" => {
                let stopping_criterion = match fallback_stopping_criterion.as_str() {
                    "pre_iter" => StoppingCriterion::PreIter,
                    "nconv" => StoppingCriterion::NConv {
                        stop_after: fallback_stop_nconv,
                    },
                    "all" => StoppingCriterion::All,
                    _ => StoppingCriterion::default(),
                };
                let relay_config = RelayDecoderConfig {
                    pre_iter: fallback_pre_iter,
                    num_sets: fallback_num_sets,
                    set_max_iter: fallback_set_max_iter,
                    gamma_dist_interval: fallback_gamma_dist_interval,
                    explicit_gammas: fallback_explicit_gammas
                        .map(|gammas| unsafe { gammas.as_array() }.to_owned()),
                    stopping_criterion,
                    logging: fallback_logging,
                    seed: fallback_seed,
                };
                BPGDFallbackConfig::Relay {
                    relay_config: Arc::new(relay_config),
                    gamma0: Some(fallback_gamma0),
                }
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unsupported BPGD fallback kind: {fallback_kind}"
                )))
            }
        };

        let inner_decoder = BPGDDecoder::new(
            Arc::new(get_sprs_bit_matrix_from_python(py, check_matrix)?),
            Arc::new(BPGDDecoderConfig {
                error_priors: unsafe { error_priors.as_array() }.to_owned(),
                alpha,
                alpha_iteration_scaling_factor,
                c_damp,
                pre_iter,
                initial_decimation_percentage,
                r_low,
                t_low,
                r_high,
                t_high,
                fallback,
            }),
        );

        let dyn_decoder = DynDecoder(Box::new(inner_decoder));
        Ok((bpgd_decoder, dyn_decoder))
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
            .map(DecodeResult::new)
            .collect()
    }
}
