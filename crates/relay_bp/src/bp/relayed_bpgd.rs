// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Relay-style BP with temporary per-leg BPGD.
//!
//! Each Relay leg keeps the usual posterior relay behavior, but may spend that
//! leg's iteration budget on:
//!
//! 1. an initial `decimation_pre_iter` warm-up,
//! 2. low-LLR decimation rounds,
//! 3. optional high-LLR decimation rounds.
//!
//! The clamps are temporary: they are cleared before the next Relay leg starts,
//! while the relayed posterior state is preserved.

use super::min_sum::{MinSumBPDecoder, MinSumDecoderConfig};
use super::relay::{RelayDecoderConfig, StoppingCriterion};
use crate::decoder::{
    Bit, BPGDExtraResult, BPExtraResult, BPStageResult, DecodeResult, Decoder, DecoderRunner,
    Mod2Mul, SparseBitMatrix,
};
use ndarray::{Array1, ArrayView1};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use rand::SeedableRng;
use std::sync::Arc;

#[derive(Clone)]
struct PosteriorUpdateState {
    rng_std: rand::rngs::StdRng,
    uniform: rand::distributions::Uniform<f64>,
    c_damp_uniform: Option<rand::distributions::Uniform<f64>>,
}

#[derive(Clone, Debug)]
pub struct RelayedBPGDDecoderConfig {
    pub error_priors: Array1<f64>,
    pub alpha: Option<f64>,
    pub alpha_iteration_scaling_factor: f64,
    pub gamma0: Option<f64>,
    pub c_damp: Option<f64>,
    pub explicit_edge_message_weights: Option<Array1<f64>>,
    pub random_decimation_candidates: usize,
    pub relay_config: Arc<RelayDecoderConfig>,
    pub decimation_pre_iter: usize,
    pub initial_decimation_percentage: f64,
    pub r_low: usize,
    pub t_low: usize,
    pub r_high: usize,
    pub t_high: usize,
}

impl Default for RelayedBPGDDecoderConfig {
    fn default() -> Self {
        Self {
            error_priors: Array1::zeros(0),
            alpha: None,
            alpha_iteration_scaling_factor: 1.0,
            gamma0: Some(0.1),
            c_damp: None,
            explicit_edge_message_weights: None,
            random_decimation_candidates: 1,
            relay_config: Arc::new(RelayDecoderConfig::default()),
            decimation_pre_iter: 0,
            initial_decimation_percentage: 0.0,
            r_low: 0,
            t_low: 3,
            r_high: 0,
            t_high: 5,
        }
    }
}

#[derive(Clone)]
pub struct RelayedBPGDDecoder {
    check_matrix: Arc<SparseBitMatrix>,
    config: Arc<RelayedBPGDDecoderConfig>,
    base_log_prior_ratios: Array1<f64>,
    bp_decoder: MinSumBPDecoder<f64>,
    posterior_update_state: PosteriorUpdateState,
    decimation_rng: rand::rngs::StdRng,
}

impl RelayedBPGDDecoder {
    pub fn new(check_matrix: Arc<SparseBitMatrix>, config: Arc<RelayedBPGDDecoderConfig>) -> Self {
        assert!(
            !(config.c_damp.is_some()
                && (config.relay_config.explicit_c_damp_messages.is_some()
                    || config.relay_config.c_damp_dist_interval.is_some())),
            "Scalar c_damp cannot be combined with relay explicit_c_damp_messages or c_damp_dist_interval.",
        );
        if let Some(explicit_c_damp_messages) = config.relay_config.explicit_c_damp_messages.as_ref() {
            let shape = explicit_c_damp_messages.shape();
            assert!(
                shape[0] == config.relay_config.num_sets + 1,
                "explicit_c_damp_messages row count {} must match num_sets + 1 = {}.",
                shape[0],
                config.relay_config.num_sets + 1,
            );
            assert!(
                shape[1] == check_matrix.nnz(),
                "explicit_c_damp_messages column count {} must match check-matrix nnz {}.",
                shape[1],
                check_matrix.nnz(),
            );
            assert!(
                explicit_c_damp_messages
                    .iter()
                    .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
                "explicit_c_damp_messages values must be finite and in [0, 1].",
            );
        }
        let bp_decoder = MinSumBPDecoder::<f64>::new(
            check_matrix.clone(),
            Arc::new(MinSumDecoderConfig {
                error_priors: config.error_priors.clone(),
                max_iter: config
                    .relay_config
                    .pre_iter
                    .max(config.relay_config.set_max_iter),
                alpha: config.alpha,
                alpha_iteration_scaling_factor: config.alpha_iteration_scaling_factor,
                c_damp: config.c_damp,
                explicit_c_damp_messages: None,
                explicit_edge_message_weights: config.explicit_edge_message_weights.clone(),
                gamma0: config.gamma0,
                data_scale_value: None,
                max_data_value: None,
                int_bits: None,
                frac_bits: None,
            }),
        );
        let base_log_prior_ratios = bp_decoder.config.log_prior_ratios();
        let posterior_update_state = Self::init_relay_state(&config.relay_config);
        let decimation_rng = rand::rngs::StdRng::seed_from_u64(config.relay_config.seed);
        Self {
            check_matrix,
            config,
            base_log_prior_ratios,
            bp_decoder,
            posterior_update_state,
            decimation_rng,
        }
    }

    fn init_relay_state(relay_config: &RelayDecoderConfig) -> PosteriorUpdateState {
        let rng_std = rand::rngs::StdRng::seed_from_u64(relay_config.seed);
        let uniform = Uniform::new(
            relay_config.gamma_dist_interval.0,
            relay_config.gamma_dist_interval.1,
        );
        let c_damp_uniform = relay_config
            .c_damp_dist_interval
            .map(|(low, high)| Uniform::new(low, high));
        PosteriorUpdateState {
            rng_std,
            uniform,
            c_damp_uniform,
        }
    }

    fn apply_stage_edge_damping(&mut self, stage_idx: usize) {
        if let Some(explicit_c_damp_messages) = self.config.relay_config.explicit_c_damp_messages.as_ref() {
            let row = explicit_c_damp_messages.row(stage_idx).to_owned();
            self.bp_decoder.set_explicit_c_damp_messages_f64(row);
            return;
        }
        if let Some(c_damp_uniform) = self.posterior_update_state.c_damp_uniform.as_ref() {
            let mut c_damp_messages = Array1::zeros(self.check_matrix.nnz());
            for c_damp in &mut c_damp_messages {
                *c_damp = c_damp_uniform.sample(&mut self.posterior_update_state.rng_std);
            }
            self.bp_decoder
                .set_explicit_c_damp_messages_f64(c_damp_messages);
            return;
        }
        self.bp_decoder.clear_explicit_c_damp_messages();
    }

    fn init_next_set(&mut self, set_idx: usize) {
        let mut gammas = Array1::zeros(self.check_matrix.cols());
        if let Some(explicit_gammas) = self.config.relay_config.explicit_gammas.as_ref() {
            let gammas_num_sets = explicit_gammas.shape()[0];
            for (i, gamma_ref) in gammas.iter_mut().enumerate() {
                *gamma_ref = *explicit_gammas
                    .get((set_idx % gammas_num_sets, i))
                    .expect("index within explicit_gammas bounds");
            }
            self.bp_decoder.set_memory_strengths_f64(gammas);
            return;
        }
        for gamma in &mut gammas {
            *gamma = self
                .posterior_update_state
                .uniform
                .sample(&mut self.posterior_update_state.rng_std);
        }
        self.bp_decoder.set_memory_strengths_f64(gammas);
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
        let count = ((remaining as f64) * fraction).ceil() as usize;
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
        self.config.relay_config.pre_iter
            + self.config.relay_config.num_sets * self.config.relay_config.set_max_iter
    }

    fn prepare_initial_leg(&mut self) {
        self.bp_decoder
            .set_log_prior_ratio_f64(self.base_log_prior_ratios.clone());
        self.bp_decoder.initialize_decoder();
        self.apply_stage_edge_damping(0);
    }

    fn prepare_next_leg(&mut self, set_idx: usize) {
        self.bp_decoder
            .set_log_prior_ratio_f64(self.base_log_prior_ratios.clone());
        self.init_next_set(set_idx);
        self.apply_stage_edge_damping(set_idx);
        self.bp_decoder.current_iteration = 0;
        self.bp_decoder.initialize_check_to_variable();
        self.bp_decoder.initialize_variable_to_check();
        if !self.config.relay_config.relay_posteriors {
            self.bp_decoder.set_posterior_ratios_to_priors();
        }
    }

    fn prepare_clamped_stage(&mut self, fixed_bits: &[Option<Bit>]) {
        self.bp_decoder
            .set_log_prior_ratio_f64(self.fixed_log_priors(fixed_bits));
        self.bp_decoder.current_iteration = 0;
        self.bp_decoder.initialize_check_to_variable();
        self.bp_decoder.initialize_variable_to_check();
    }

    fn run_stage(&mut self, detectors: ArrayView1<Bit>, max_iter: usize) -> DecodeResult {
        let mut success = false;
        let mut decoded_detectors = Array1::default(detectors.dim());
        for _ in 0..max_iter {
            self.bp_decoder.run_iteration(detectors);
            self.bp_decoder.current_iteration += 1;
            decoded_detectors = self.bp_decoder.compute_decoded_detectors();
            success = self
                .bp_decoder
                .check_convergence(detectors, decoded_detectors.view());
            if success {
                break;
            }
        }
        self.bp_decoder.build_result(success, decoded_detectors, max_iter)
    }

    fn run_leg(&mut self, detectors: ArrayView1<Bit>, leg_budget: usize) -> DecodeResult {
        let num_vars = self.base_log_prior_ratios.len();
        let mut fixed_bits = vec![None; num_vars];
        let mut remaining_budget = leg_budget;
        let mut last_result: Option<DecodeResult> = None;
        let mut ran_any_stage = false;

        if self.config.decimation_pre_iter > 0 && remaining_budget > 0 {
            let stage_budget = remaining_budget.min(self.config.decimation_pre_iter);
            let result = self.run_stage(detectors, stage_budget);
            remaining_budget = remaining_budget.saturating_sub(result.iterations);
            ran_any_stage = true;
            if result.success {
                return result;
            }
            last_result = Some(result);
        }

        for _ in 0..self.config.r_low.min(num_vars) {
            if remaining_budget == 0 {
                break;
            }
            self.prepare_clamped_stage(&fixed_bits);
            let stage_budget = remaining_budget.min(self.config.t_low);
            let result = self.run_stage(detectors, stage_budget);
            remaining_budget = remaining_budget.saturating_sub(result.iterations);
            ran_any_stage = true;
            if result.success {
                return result;
            }
            if let Some(index) =
                self.choose_decimation_index(&result.posterior_ratios, &fixed_bits, true)
            {
                fixed_bits[index] = Some(result.decoding[index]);
            } else {
                return result;
            }
            last_result = Some(result);
        }

        if self.config.r_high > 0 {
            if self.config.initial_decimation_percentage > 0.0
                && fixed_bits.iter().any(|bit| bit.is_none())
            {
                if let Some(result) = last_result.as_ref() {
                    self.apply_initial_high_decimation(result, &mut fixed_bits);
                } else if remaining_budget > 0 {
                    self.prepare_clamped_stage(&fixed_bits);
                    let stage_budget = remaining_budget.min(self.config.t_high);
                    let result = self.run_stage(detectors, stage_budget);
                    remaining_budget = remaining_budget.saturating_sub(result.iterations);
                    ran_any_stage = true;
                    if result.success {
                        return result;
                    }
                    self.apply_initial_high_decimation(&result, &mut fixed_bits);
                    last_result = Some(result);
                }
            }

            for _ in 0..self.config.r_high.min(num_vars) {
                if remaining_budget == 0 {
                    break;
                }
                self.prepare_clamped_stage(&fixed_bits);
                let stage_budget = remaining_budget.min(self.config.t_high);
                let result = self.run_stage(detectors, stage_budget);
                remaining_budget = remaining_budget.saturating_sub(result.iterations);
                ran_any_stage = true;
                if result.success {
                    return result;
                }
                if let Some(index) =
                    self.choose_decimation_index(&result.posterior_ratios, &fixed_bits, false)
                {
                    fixed_bits[index] = Some(result.decoding[index]);
                } else {
                    return result;
                }
                last_result = Some(result);
            }
        }

        if !ran_any_stage && leg_budget > 0 {
            return self.run_stage(detectors, leg_budget);
        }

        last_result.unwrap_or_else(|| {
            self.bp_decoder
                .build_result(false, self.bp_decoder.compute_decoded_detectors(), leg_budget)
        })
    }
}

impl Decoder for RelayedBPGDDecoder {
    fn check_matrix(&self) -> Arc<SparseBitMatrix> {
        self.check_matrix.clone()
    }

    fn log_prior_ratios(&mut self) -> Array1<f64> {
        self.base_log_prior_ratios.clone()
    }

    fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult {
        let stopping_criterion = self.config.relay_config.stopping_criterion.clone();
        let mut stage_results: Vec<BPStageResult> = Vec::new();
        let mut total_iterations = 0usize;
        let mut num_conv = 0usize;
        let mut min_pm = f64::MAX;
        let mut best_result: Option<DecodeResult> = None;
        let mut last_result: Option<DecodeResult> = None;
        let mut converged_stage_index: Option<usize> = None;

        let leg_budgets = std::iter::once(self.config.relay_config.pre_iter)
            .chain(std::iter::repeat(self.config.relay_config.set_max_iter).take(self.config.relay_config.num_sets));

        for (leg_index, leg_budget) in leg_budgets.enumerate() {
            if leg_index == 0 {
                self.prepare_initial_leg();
            } else {
                self.prepare_next_leg(leg_index);
            }

            let leg_result = self.run_leg(detectors, leg_budget);
            total_iterations += leg_result.iterations;
            stage_results.push(BPStageResult {
                name: format!("relay_leg_{leg_index}"),
                damping_mode: leg_result.damping_mode.clone(),
                iterations: leg_result.iterations,
                converged: leg_result.success,
            });
            last_result = Some(leg_result.clone());

            if leg_result.success {
                num_conv += 1;
                let pm = self.original_decoding_quality(&leg_result.decoding);
                if best_result.is_none() || pm < min_pm {
                    min_pm = pm;
                    converged_stage_index = Some(leg_index);
                    best_result = Some(leg_result);
                }

                let mut done = false;
                if leg_index == 0 && stopping_criterion == StoppingCriterion::PreIter {
                    done = true;
                } else if let StoppingCriterion::NConv { stop_after } = stopping_criterion {
                    if num_conv >= stop_after {
                        done = true;
                    }
                }

                if done {
                    break;
                }
            } else {
                continue;
            }

            if best_result.is_some() {
                break;
            }
        }

        let mut result = best_result.or(last_result).unwrap_or_else(|| {
            self.bp_decoder
                .build_result(false, self.bp_decoder.compute_decoded_detectors(), 0)
        });
        result.iterations = total_iterations;
        result.max_iter = self.max_total_iterations();
        result.decoded_detectors = self.check_matrix.mul_mod2(&result.decoding);
        if result.success {
            result.decoding_quality = self.original_decoding_quality(&result.decoding);
        }
        result.extra = BPExtraResult::RelayedBPGD(BPGDExtraResult {
            fallback_used: false,
            converged_stage_index,
            stage_results,
        });
        result
    }
}

impl DecoderRunner for RelayedBPGDDecoder {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bipartite_graph::{BipartiteGraph, SparseBipartiteGraph};
    use crate::bp::relay::RelayDecoderConfig;
    use ndarray::array;

    #[test]
    fn relayed_bpgd_decode_repetition_code() {
        let check_matrix = SparseBipartiteGraph::from_dense(array![[1, 1, 0], [0, 1, 1],]);
        let mut decoder = RelayedBPGDDecoder::new(
            Arc::new(check_matrix),
            Arc::new(RelayedBPGDDecoderConfig {
                error_priors: array![0.003, 0.003, 0.003],
                alpha: Some(1.0),
                gamma0: Some(0.15),
                relay_config: Arc::new(RelayDecoderConfig {
                    pre_iter: 20,
                    num_sets: 2,
                    set_max_iter: 10,
                    ..Default::default()
                }),
                decimation_pre_iter: 5,
                r_low: 1,
                t_low: 5,
                r_high: 0,
                t_high: 5,
                ..Default::default()
            }),
        );

        let result = decoder.decode_detailed(array![1, 1].view());

        assert!(result.success);
        let BPExtraResult::RelayedBPGD(extra) = result.extra else {
            panic!("expected RelayedBPGD extra result");
        };
        assert_eq!(extra.stage_results.len(), 1);
        assert_eq!(extra.stage_results[0].name, "relay_leg_0");
        assert!(extra.stage_results[0].converged);
        assert_eq!(extra.converged_stage_index, Some(0));
    }
}
