// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Native BP-guided decimation (BPGD) wrapper.
//!
//! The repository's existing fast message-passing kernels already live in
//! [`MinSumBPDecoder`] and [`RelayDecoder`].  This module keeps the expensive
//! belief-propagation work in Rust and only adds a thin controller that:
//!
//! 1. runs min-sum for a configurable number of iterations,
//! 2. uses the returned posterior ratios to choose a variable to clamp,
//! 3. repeats for the requested number of rounds,
//! 4. optionally hands the clamped problem to a native Relay fallback.
//!
//! The design deliberately mirrors the Python experiments discussed in the
//! surrounding project chat, but keeps the hot loops and the detailed
//! statistics in the native layer.

use super::min_sum::{MinSumBPDecoder, MinSumDecoderConfig};
use super::relay::{RelayDecoder, RelayDecoderConfig};
use crate::decoder::{
    Bit, BPGDExtraResult, BPExtraResult, BPStageResult, DecodeResult, Decoder, DecoderRunner,
    Mod2Mul, SparseBitMatrix,
};
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum BPGDFallbackConfig {
    None,
    Relay {
        relay_config: Arc<RelayDecoderConfig>,
        gamma0: Option<f64>,
    },
}

#[derive(Clone, Debug)]
pub struct BPGDDecoderConfig {
    pub error_priors: Array1<f64>,
    pub alpha: Option<f64>,
    pub alpha_iteration_scaling_factor: f64,
    pub c_damp: Option<f64>,
    pub explicit_c_damp_messages: Option<Array1<f64>>,
    pub random_decimation_candidates: usize,
    pub random_seed: u64,
    pub pre_iter: usize,
    pub initial_decimation_percentage: f64,
    pub r_low: usize,
    pub t_low: usize,
    pub r_high: usize,
    pub t_high: usize,
    pub fallback: BPGDFallbackConfig,
}

impl Default for BPGDDecoderConfig {
    fn default() -> Self {
        Self {
            error_priors: Array1::zeros(0),
            alpha: None,
            alpha_iteration_scaling_factor: 1.0,
            c_damp: None,
            explicit_c_damp_messages: None,
            random_decimation_candidates: 1,
            random_seed: 0,
            pre_iter: 0,
            initial_decimation_percentage: 0.0,
            r_low: 5,
            t_low: 3,
            r_high: 1000,
            t_high: 5,
            fallback: BPGDFallbackConfig::None,
        }
    }
}

#[derive(Clone)]
pub struct BPGDDecoder {
    check_matrix: Arc<SparseBitMatrix>,
    config: Arc<BPGDDecoderConfig>,
    base_log_prior_ratios: Array1<f64>,
    stage_decoder: MinSumBPDecoder<f64>,
    decimation_rng: rand::rngs::StdRng,
}

impl BPGDDecoder {
    pub fn new(check_matrix: Arc<SparseBitMatrix>, config: Arc<BPGDDecoderConfig>) -> Self {
        let stage_decoder = MinSumBPDecoder::<f64>::new(
            check_matrix.clone(),
            Arc::new(MinSumDecoderConfig {
                error_priors: config.error_priors.clone(),
                max_iter: config.t_low,
                alpha: config.alpha,
                alpha_iteration_scaling_factor: config.alpha_iteration_scaling_factor,
                c_damp: config.c_damp,
                explicit_c_damp_messages: config.explicit_c_damp_messages.clone(),
                explicit_edge_message_weights: None,
                gamma0: None,
                data_scale_value: None,
                max_data_value: None,
                int_bits: None,
                frac_bits: None,
            }),
        );
        let base_log_prior_ratios = stage_decoder.config.log_prior_ratios();
        let decimation_rng = rand::rngs::StdRng::seed_from_u64(config.random_seed);
        Self {
            check_matrix,
            config,
            base_log_prior_ratios,
            stage_decoder,
            decimation_rng,
        }
    }

    fn choose_decimation_index(
        &mut self,
        posterior_ratios: &Array1<f64>,
        fixed_bits: &[Option<Bit>],
        choose_low_llr: bool,
    ) -> Option<usize> {
        let mut candidates: Vec<(usize, f64)> = posterior_ratios
            .iter()
            .enumerate()
            .filter_map(|(index, posterior)| {
                if fixed_bits[index].is_some() {
                    None
                } else {
                    Some((index, posterior.abs()))
                }
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        if choose_low_llr {
            candidates.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        } else {
            candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        }
        let pool_size = self
            .config
            .random_decimation_candidates
            .max(1)
            .min(candidates.len());
        let choice = if pool_size == 1 {
            0
        } else {
            self.decimation_rng.gen_range(0..pool_size)
        };
        Some(candidates[choice].0)
    }

    fn choose_high_confidence_indices(
        &self,
        posterior_ratios: &Array1<f64>,
        fixed_bits: &[Option<Bit>],
        count: usize,
    ) -> Vec<usize> {
        let mut candidates: Vec<(usize, f64)> = posterior_ratios
            .iter()
            .enumerate()
            .filter_map(|(index, posterior)| {
                if fixed_bits[index].is_some() {
                    None
                } else {
                    Some((index, posterior.abs()))
                }
            })
            .collect();
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        candidates
            .into_iter()
            .take(count)
            .map(|(index, _)| index)
            .collect()
    }

    fn normalized_initial_decimation_fraction(&self) -> f64 {
        let raw = self.config.initial_decimation_percentage;
        if !raw.is_finite() || raw <= 0.0 {
            0.0
        } else if raw <= 1.0 {
            raw
        } else if raw <= 100.0 {
            raw / 100.0
        } else {
            1.0
        }
    }

    fn apply_initial_high_decimation(
        &self,
        result: &DecodeResult,
        fixed_bits: &mut [Option<Bit>],
    ) -> usize {
        let fraction = self.normalized_initial_decimation_fraction();
        if fraction <= 0.0 {
            return 0;
        }
        let remaining = fixed_bits.iter().filter(|bit| bit.is_none()).count();
        if remaining == 0 {
            return 0;
        }
        let count = ((remaining as f64) * fraction).floor() as usize;
        if count == 0 {
            return 0;
        }
        let indices = self.choose_high_confidence_indices(&result.posterior_ratios, fixed_bits, count);
        for index in &indices {
            fixed_bits[*index] = Some(result.decoding[*index]);
        }
        indices.len()
    }

    fn fixed_log_priors(&self, fixed_bits: &[Option<Bit>]) -> Array1<f64> {
        let mut priors = self.base_log_prior_ratios.clone();
        for (index, bit) in fixed_bits.iter().enumerate() {
            match *bit {
                Some(0) => priors[index] = f64::INFINITY,
                Some(1) => priors[index] = f64::NEG_INFINITY,
                _ => {}
            }
        }
        priors
    }

    fn set_stage_max_iter(&mut self, max_iter: usize) {
        Arc::make_mut(&mut self.stage_decoder.config).max_iter = max_iter;
    }

    fn run_stage(
        &mut self,
        detectors: ArrayView1<Bit>,
        fixed_bits: &[Option<Bit>],
        max_iter: usize,
    ) -> DecodeResult {
        self.set_stage_max_iter(max_iter);
        self.stage_decoder
            .set_log_prior_ratio_f64(self.fixed_log_priors(fixed_bits));
        self.stage_decoder.decode_detailed(detectors)
    }

    fn build_fallback_decoder(&self, fixed_bits: &[Option<Bit>]) -> Option<RelayDecoder<f64>> {
        match &self.config.fallback {
            BPGDFallbackConfig::None => None,
            BPGDFallbackConfig::Relay {
                relay_config,
                gamma0,
            } => {
                let mut decoder = RelayDecoder::<f64>::new(
                    self.check_matrix.clone(),
                    Arc::new(MinSumDecoderConfig {
                        error_priors: self.config.error_priors.clone(),
                        max_iter: relay_config.pre_iter,
                        alpha: self.config.alpha,
                        alpha_iteration_scaling_factor: self.config.alpha_iteration_scaling_factor,
                        c_damp: None,
                        explicit_c_damp_messages: None,
                        explicit_edge_message_weights: None,
                        gamma0: *gamma0,
                        data_scale_value: None,
                        max_data_value: None,
                        int_bits: None,
                        frac_bits: None,
                    }),
                    relay_config.clone(),
                );
                decoder.set_log_prior_ratio_f64(self.fixed_log_priors(fixed_bits));
                Some(decoder)
            }
        }
    }

    fn original_decoding_quality(&self, decoding: &Array1<Bit>) -> f64 {
        let mut decoding_quality = 0.0;
        for i in 0..decoding.len() {
            if decoding[i] == 1 && f64::is_finite(self.base_log_prior_ratios[i]) {
                decoding_quality += self.base_log_prior_ratios[i];
            }
        }
        decoding_quality
    }

    fn max_total_iterations(&self) -> usize {
        let num_vars = self.base_log_prior_ratios.len();
        let low_rounds = self.config.r_low.min(num_vars);
        let high_rounds = self.config.r_high.min(num_vars);
        let pre_iters = if self.config.pre_iter > 0 && num_vars > 0 { 2 } else { 0 };
        let fallback_max = match &self.config.fallback {
            BPGDFallbackConfig::None => 0,
            BPGDFallbackConfig::Relay { relay_config, .. } => {
                relay_config.pre_iter + relay_config.num_sets * relay_config.set_max_iter
            }
        };
        pre_iters * self.config.pre_iter
            + low_rounds * self.config.t_low
            + high_rounds * self.config.t_high
            + fallback_max
    }
}

impl Decoder for BPGDDecoder {
    fn check_matrix(&self) -> Arc<SparseBitMatrix> {
        self.check_matrix.clone()
    }

    fn log_prior_ratios(&mut self) -> Array1<f64> {
        self.base_log_prior_ratios.clone()
    }

    fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult {
        let num_vars = self.base_log_prior_ratios.len();
        let mut fixed_bits = vec![None; num_vars];
        let mut stage_results: Vec<BPStageResult> = Vec::new();
        let mut converged_stage_index: Option<usize> = None;
        let mut fallback_used = false;
        let mut total_iterations = 0usize;
        let mut final_result: Option<DecodeResult> = None;

        let low_rounds = self.config.r_low.min(num_vars);
        let mut low_total_iterations = 0usize;
        let mut low_converged = false;
        if self.config.pre_iter > 0 {
            let result = self.run_stage(detectors, &fixed_bits, self.config.pre_iter);
            low_total_iterations += result.iterations;
            total_iterations += result.iterations;
            if result.success {
                low_converged = true;
                converged_stage_index = Some(0);
                final_result = Some(result);
            } else {
                final_result = Some(result);
            }
        }
        for _ in 0..low_rounds {
            if low_converged {
                break;
            }
            let result = self.run_stage(detectors, &fixed_bits, self.config.t_low);
            low_total_iterations += result.iterations;
            total_iterations += result.iterations;
            if result.success {
                low_converged = true;
                converged_stage_index = Some(0);
                final_result = Some(result);
                break;
            }
            if let Some(index) = self.choose_decimation_index(
                &result.posterior_ratios,
                &fixed_bits,
                true,
            ) {
                fixed_bits[index] = Some(result.decoding[index]);
            } else {
                final_result = Some(result);
                break;
            }
            final_result = Some(result);
        }
        stage_results.push(BPStageResult {
            name: "BPGD_low".to_string(),
            damping_mode: final_result
                .as_ref()
                .map(|result| result.damping_mode.clone())
                .unwrap_or_else(|| self.stage_decoder.damping_mode().to_string()),
            iterations: low_total_iterations,
            converged: low_converged,
        });

        if !low_converged {
            let high_rounds = self.config.r_high.min(num_vars);
            let mut high_total_iterations = 0usize;
            let mut high_converged = false;
            if self.config.pre_iter > 0 {
                let result = self.run_stage(detectors, &fixed_bits, self.config.pre_iter);
                high_total_iterations += result.iterations;
                total_iterations += result.iterations;
                if result.success {
                    high_converged = true;
                    converged_stage_index = Some(1);
                    final_result = Some(result);
                } else {
                    self.apply_initial_high_decimation(&result, &mut fixed_bits);
                    final_result = Some(result);
                }
            }
            for _ in 0..high_rounds {
                if high_converged {
                    break;
                }
                let result = self.run_stage(detectors, &fixed_bits, self.config.t_high);
                high_total_iterations += result.iterations;
                total_iterations += result.iterations;
                if result.success {
                    high_converged = true;
                    converged_stage_index = Some(1);
                    final_result = Some(result);
                    break;
                }
                if let Some(index) = self.choose_decimation_index(
                    &result.posterior_ratios,
                    &fixed_bits,
                    false,
                ) {
                    fixed_bits[index] = Some(result.decoding[index]);
                } else {
                    final_result = Some(result);
                    break;
                }
                final_result = Some(result);
            }
            stage_results.push(BPStageResult {
                name: "BPGD_high".to_string(),
                damping_mode: final_result
                    .as_ref()
                    .map(|result| result.damping_mode.clone())
                    .unwrap_or_else(|| self.stage_decoder.damping_mode().to_string()),
                iterations: high_total_iterations,
                converged: high_converged,
            });

            if !high_converged {
                if let Some(mut fallback_decoder) = self.build_fallback_decoder(&fixed_bits) {
                    fallback_used = true;
                    let fallback_result = fallback_decoder.decode_detailed(detectors);
                    total_iterations += fallback_result.iterations;
                    let fallback_converged = fallback_result.success;
                    if fallback_converged {
                        converged_stage_index = Some(stage_results.len());
                    }
                    stage_results.push(BPStageResult {
                        name: "relay-bp".to_string(),
                        damping_mode: fallback_result.damping_mode.clone(),
                        iterations: fallback_result.iterations,
                        converged: fallback_converged,
                    });
                    final_result = Some(fallback_result);
                }
            }
        }

        let mut result = final_result.unwrap_or_else(|| {
            self.run_stage(detectors, &fixed_bits, self.config.t_high.max(self.config.t_low))
        });
        if total_iterations == 0 {
            total_iterations = result.iterations;
        }
        result.iterations = total_iterations;
        result.max_iter = self.max_total_iterations();
        result.decoded_detectors = self.check_matrix.mul_mod2(&result.decoding);
        if result.success {
            result.decoding_quality = self.original_decoding_quality(&result.decoding);
        }
        result.extra = BPExtraResult::BPGD(BPGDExtraResult {
            fallback_used,
            converged_stage_index,
            stage_results,
        });
        result
    }
}

impl DecoderRunner for BPGDDecoder {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bipartite_graph::{BipartiteGraph, SparseBipartiteGraph};
    use ndarray::array;

    #[test]
    fn high_phase_stops_when_no_variables_left_to_decimate() {
        let check_matrix = SparseBipartiteGraph::from_dense(array![[1, 1], [1, 1]]);
        let mut decoder = BPGDDecoder::new(
            Arc::new(check_matrix),
            Arc::new(BPGDDecoderConfig {
                error_priors: array![0.0, 0.0],
                pre_iter: 1,
                initial_decimation_percentage: 100.0,
                r_low: 0,
                t_low: 1,
                r_high: 100,
                t_high: 1,
                fallback: BPGDFallbackConfig::None,
                ..Default::default()
            }),
        );

        let result = decoder.decode_detailed(array![0, 1].view());

        assert!(!result.success);
        let BPExtraResult::BPGD(extra) = result.extra else {
            panic!("expected BPGD extra result");
        };
        assert_eq!(extra.stage_results.len(), 2);
        assert_eq!(extra.stage_results[0].name, "BPGD_low");
        assert_eq!(extra.stage_results[0].iterations, 1);
        assert!(!extra.stage_results[0].converged);
        assert_eq!(extra.stage_results[1].name, "BPGD_high");
        assert_eq!(extra.stage_results[1].iterations, 2);
        assert!(!extra.stage_results[1].converged);
    }
}
