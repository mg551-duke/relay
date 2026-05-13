// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use super::min_sum::{MinSumBPDecoder, MinSumDecoderConfig};
use crate::decoder::{Bit, SparseBitMatrix};
use crate::decoder::{DecodeResult, Decoder, DecoderRunner};
use log::debug;

use ndarray::{Array1, Array2, ArrayView1};
use num_traits::{Bounded, FromPrimitive, Signed, ToPrimitive};
use std::fmt::Debug;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
//use std::string;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use rand::SeedableRng;
use std::process::exit;
use std::sync::Arc;

#[derive(Clone, PartialEq, Debug)]
pub enum StoppingCriterion {
    PreIter,
    NConv { stop_after: usize },
    All,
}

impl Default for StoppingCriterion {
    fn default() -> StoppingCriterion {
        StoppingCriterion::NConv { stop_after: 1 }
    }
}

#[derive(Clone, Debug)]
pub enum GammaSampler {
    UniformInterval {
        low: f64,
        high: f64,
    },
    BernoulliTwoPoint {
        negative: f64,
        positive: f64,
        p_positive: f64,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct MessageMixBernoulliSampler {
    pub fresh_negative: f64,
    pub fresh_positive: f64,
    pub fresh_p_positive: f64,
    pub previous_negative: f64,
    pub previous_positive: f64,
    pub previous_p_positive: f64,
    pub second_previous_negative: Option<f64>,
    pub second_previous_positive: Option<f64>,
    pub second_previous_p_positive: Option<f64>,
}

impl GammaSampler {
    pub fn uniform_interval(interval: (f64, f64)) -> Self {
        Self::UniformInterval {
            low: interval.0,
            high: interval.1,
        }
    }

    pub fn bernoulli_two_point(
        negative: f64,
        positive: f64,
        p_positive: f64,
    ) -> Result<Self, String> {
        let sampler = Self::BernoulliTwoPoint {
            negative,
            positive,
            p_positive,
        };
        sampler.validate()?;
        Ok(sampler)
    }

    pub fn validate(&self) -> Result<(), String> {
        match *self {
            Self::UniformInterval { low, high } => {
                if !low.is_finite() || !high.is_finite() {
                    return Err("gamma interval endpoints must be finite.".to_string());
                }
                if high < low {
                    return Err("gamma interval high must be >= low.".to_string());
                }
            }
            Self::BernoulliTwoPoint {
                negative,
                positive,
                p_positive,
            } => {
                if !negative.is_finite() || !positive.is_finite() || !p_positive.is_finite() {
                    return Err("Bernoulli gamma parameters must be finite.".to_string());
                }
                if negative > 0.0 {
                    return Err("Bernoulli negative gamma value must be <= 0.".to_string());
                }
                if positive < 0.0 {
                    return Err("Bernoulli positive gamma value must be >= 0.".to_string());
                }
                if !(0.0..=1.0).contains(&p_positive) {
                    return Err("Bernoulli p_positive must be in [0, 1].".to_string());
                }
            }
        }
        Ok(())
    }

    pub fn sample(&self, rng: &mut rand::rngs::StdRng) -> f64 {
        match *self {
            Self::UniformInterval { low, high } => {
                if high <= low {
                    low
                } else {
                    rng.gen_range(low..high)
                }
            }
            Self::BernoulliTwoPoint {
                negative,
                positive,
                p_positive,
            } => {
                if rng.gen::<f64>() < p_positive {
                    positive
                } else {
                    negative
                }
            }
        }
    }
}

impl MessageMixBernoulliSampler {
    pub fn new(
        fresh_negative: f64,
        fresh_positive: f64,
        fresh_p_positive: f64,
        previous_negative: f64,
        previous_positive: f64,
        previous_p_positive: f64,
    ) -> Result<Self, String> {
        let sampler = Self {
            fresh_negative,
            fresh_positive,
            fresh_p_positive,
            previous_negative,
            previous_positive,
            previous_p_positive,
            second_previous_negative: None,
            second_previous_positive: None,
            second_previous_p_positive: None,
        };
        sampler.validate()?;
        Ok(sampler)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_second_previous(
        fresh_negative: f64,
        fresh_positive: f64,
        fresh_p_positive: f64,
        previous_negative: f64,
        previous_positive: f64,
        previous_p_positive: f64,
        second_previous_negative: f64,
        second_previous_positive: f64,
        second_previous_p_positive: f64,
    ) -> Result<Self, String> {
        let sampler = Self {
            fresh_negative,
            fresh_positive,
            fresh_p_positive,
            previous_negative,
            previous_positive,
            previous_p_positive,
            second_previous_negative: Some(second_previous_negative),
            second_previous_positive: Some(second_previous_positive),
            second_previous_p_positive: Some(second_previous_p_positive),
        };
        sampler.validate()?;
        Ok(sampler)
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_signed_two_point(
            "fresh message-mix",
            self.fresh_negative,
            self.fresh_positive,
            self.fresh_p_positive,
        )?;
        validate_signed_two_point(
            "previous message-mix",
            self.previous_negative,
            self.previous_positive,
            self.previous_p_positive,
        )?;
        match (
            self.second_previous_negative,
            self.second_previous_positive,
            self.second_previous_p_positive,
        ) {
            (Some(negative), Some(positive), Some(p_positive)) => {
                validate_signed_two_point(
                    "second previous message-mix",
                    negative,
                    positive,
                    p_positive,
                )?;
            }
            (None, None, None) => {}
            _ => {
                return Err(
                    "second previous message-mix Bernoulli parameters must be all present or all absent."
                        .to_string(),
                );
            }
        }
        Ok(())
    }

    pub fn has_second_previous(&self) -> bool {
        self.second_previous_negative.is_some()
    }

    pub fn sample(&self, rng: &mut rand::rngs::StdRng) -> (f64, f64, Option<f64>) {
        let second_previous = match (
            self.second_previous_negative,
            self.second_previous_positive,
            self.second_previous_p_positive,
        ) {
            (Some(negative), Some(positive), Some(p_positive)) => {
                Some(sample_signed_two_point(negative, positive, p_positive, rng))
            }
            _ => None,
        };
        (
            sample_signed_two_point(
                self.fresh_negative,
                self.fresh_positive,
                self.fresh_p_positive,
                rng,
            ),
            sample_signed_two_point(
                self.previous_negative,
                self.previous_positive,
                self.previous_p_positive,
                rng,
            ),
            second_previous,
        )
    }
}

fn validate_signed_two_point(
    label: &str,
    negative: f64,
    positive: f64,
    p_positive: f64,
) -> Result<(), String> {
    if !negative.is_finite() || !positive.is_finite() || !p_positive.is_finite() {
        return Err(format!("{label} Bernoulli parameters must be finite."));
    }
    if negative > 0.0 {
        return Err(format!("{label} negative value must be <= 0."));
    }
    if positive < 0.0 {
        return Err(format!("{label} positive value must be >= 0."));
    }
    if !(0.0..=1.0).contains(&p_positive) {
        return Err(format!("{label} p_positive must be in [0, 1]."));
    }
    Ok(())
}

fn sample_signed_two_point(
    negative: f64,
    positive: f64,
    p_positive: f64,
    rng: &mut rand::rngs::StdRng,
) -> f64 {
    if rng.gen::<f64>() < p_positive {
        positive
    } else {
        negative
    }
}

#[derive(Clone, Debug)]
pub struct RelayDecoderConfig {
    pub pre_iter: usize,
    pub num_sets: usize,
    pub set_max_iter: usize,
    pub gamma_dist_interval: (f64, f64),
    pub gamma_sampler: GammaSampler,
    pub explicit_gammas: Option<Array2<f64>>,
    pub explicit_c_damp_messages: Option<Array2<f64>>,
    pub c_damp_dist_interval: Option<(f64, f64)>,
    pub message_mix_bernoulli: Option<MessageMixBernoulliSampler>,
    pub relay_posteriors: bool,
    pub stopping_criterion: StoppingCriterion,
    pub logging: bool,
    pub seed: u64,
}

impl Default for RelayDecoderConfig {
    fn default() -> Self {
        Self {
            pre_iter: 80,
            num_sets: 300,
            set_max_iter: 60,
            gamma_dist_interval: (-0.24, 0.66),
            gamma_sampler: GammaSampler::uniform_interval((-0.24, 0.66)),
            explicit_gammas: None,
            explicit_c_damp_messages: None,
            c_damp_dist_interval: None,
            message_mix_bernoulli: None,
            relay_posteriors: true,
            stopping_criterion: StoppingCriterion::default(),
            logging: false,
            seed: 0,
        }
    }
}

impl RelayDecoderConfig {
    pub fn effective_gamma_sampler(&self) -> GammaSampler {
        match self.gamma_sampler {
            GammaSampler::UniformInterval { low, high }
                if low == -0.24 && high == 0.66 && self.gamma_dist_interval != (-0.24, 0.66) =>
            {
                GammaSampler::uniform_interval(self.gamma_dist_interval)
            }
            ref sampler => sampler.clone(),
        }
    }
}

#[derive(Clone)]
struct PosteriorUpdateState {
    rng_std: rand::rngs::StdRng,
    c_damp_uniform: Option<rand::distributions::Uniform<f64>>,
}

/// An ensemble decoder which controls an inner BP min-sum decoder.
#[derive(Clone)]
pub struct RelayDecoder<N: PartialEq + Default + Clone + Copy> {
    bp_decoder: MinSumBPDecoder<N>,
    relay_config: Arc<RelayDecoderConfig>,
    posterior_update_state: PosteriorUpdateState,
    sets_quality: Array1<f64>,
    sets_iter: Array1<usize>,
    sets_conv: Array1<bool>,
    sets_best: Array1<bool>,
    num_executed_sets: usize,
}

impl<N> RelayDecoder<N>
where
    N: PartialEq
        + Debug
        + Default
        + Clone
        + Copy
        + Signed
        + Bounded
        + FromPrimitive
        + ToPrimitive
        + std::cmp::PartialOrd
        + std::ops::Add
        + std::ops::AddAssign
        + std::ops::DivAssign
        + std::ops::Mul<N>
        + std::ops::MulAssign
        + Send
        + Sync
        + std::fmt::Display
        + 'static,
{
    pub fn new(
        check_matrix: Arc<SparseBitMatrix>,
        min_sum_config: Arc<MinSumDecoderConfig>,
        relay_config: Arc<RelayDecoderConfig>,
    ) -> RelayDecoder<N> {
        relay_config
            .effective_gamma_sampler()
            .validate()
            .expect("Invalid relay gamma sampler.");
        if let Some(message_mix) = relay_config.message_mix_bernoulli {
            message_mix
                .validate()
                .expect("Invalid relay message-mix sampler.");
        }
        assert!(
            !(min_sum_config.c_damp.is_some()
                && (relay_config.explicit_c_damp_messages.is_some()
                    || relay_config.c_damp_dist_interval.is_some())),
            "Scalar c_damp cannot be combined with relay explicit_c_damp_messages or c_damp_dist_interval.",
        );
        assert!(
            !(relay_config.message_mix_bernoulli.is_some()
                && (min_sum_config.c_damp.is_some()
                    || min_sum_config.explicit_c_damp_messages.is_some()
                    || relay_config.explicit_c_damp_messages.is_some()
                    || relay_config.c_damp_dist_interval.is_some())),
            "message_mix_bernoulli cannot be combined with c_damp, explicit_c_damp_messages, or c_damp_dist_interval.",
        );
        assert!(
            min_sum_config.explicit_c_damp_messages.is_none(),
            "RelayDecoder expects per-leg edge damping in RelayDecoderConfig, not MinSumDecoderConfig.",
        );
        assert!(
            min_sum_config
                .explicit_message_mix_fresh_coefficients
                .is_none()
                && min_sum_config
                    .explicit_message_mix_previous_coefficients
                    .is_none()
                && min_sum_config
                    .explicit_message_mix_second_previous_coefficients
                    .is_none(),
            "RelayDecoder expects per-leg message-mix coefficients in RelayDecoderConfig, not MinSumDecoderConfig.",
        );
        if relay_config.logging {
            let log_line = format!(
                "# pre_iter: {}: sets: {} set_max_iter: {}\n\
                # gamma_distribution: {:?} # set_idx, num_iter, converged, unique_best_solution\n",
                relay_config.pre_iter,
                relay_config.num_sets,
                relay_config.set_max_iter,
                relay_config.effective_gamma_sampler(),
            );
            let mut file =
                File::create("relay_logging.out").expect("Unable to create file for logging.");
            file.write_all(log_line.as_bytes())
                .expect("Unable to write Relay logging data.");
        }

        // Create logging variables if applicable
        let (sets_quality, sets_iter, sets_conv, sets_best);
        if relay_config.logging {
            sets_quality = Array1::<f64>::from_elem(relay_config.num_sets + 1, f64::MAX);
            sets_iter =
                Array1::<usize>::from_elem(relay_config.num_sets + 1, relay_config.set_max_iter);
            sets_conv = Array1::<bool>::from_elem(relay_config.num_sets + 1, false);
            sets_best = Array1::<bool>::from_elem(relay_config.num_sets + 1, false);
        } else {
            sets_quality = Array1::<f64>::zeros(1);
            sets_iter = Array1::<usize>::zeros(1);
            sets_conv = Array1::<bool>::from_elem(1, false);
            sets_best = Array1::<bool>::from_elem(1, false);
        }

        if let Some(gammas) = relay_config.explicit_gammas.as_ref() {
            let gammas_shape = gammas.shape();
            let num_variable_nodes = check_matrix.cols();
            if num_variable_nodes != gammas_shape[1] {
                eprintln!("ERROR: Number of specified gammas {} does not match the number of variable nodes {}.", gammas_shape[1], num_variable_nodes);
                exit(1);
            };
            if relay_config.num_sets > gammas_shape[0] {
                eprintln!("WARNING: Number of different gamma sets {} is smaller than the number of Relay legs {}. Legs will be reused.", gammas_shape[0], relay_config.num_sets)
            }
        }
        if let Some(explicit_c_damp_messages) = relay_config.explicit_c_damp_messages.as_ref() {
            let shape = explicit_c_damp_messages.shape();
            assert!(
                shape[0] == relay_config.num_sets + 1,
                "explicit_c_damp_messages row count {} must match num_sets + 1 = {}.",
                shape[0],
                relay_config.num_sets + 1,
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

        // The actual number of sets Relay ran, depends on the stopping criterion
        let num_executed_sets = 0;

        let bp_decoder = MinSumBPDecoder::new(check_matrix, min_sum_config);

        let posterior_update_state = Self::init_dismem_state(&relay_config);

        RelayDecoder {
            bp_decoder,
            relay_config,
            posterior_update_state,
            sets_quality,
            sets_iter,
            sets_conv,
            sets_best,
            num_executed_sets,
        }
    }

    /// Override the inner BP decoder's log-prior ratios.
    ///
    /// This is primarily used by wrapper decoders such as BPGD that keep
    /// the message-passing kernel in Rust while externally deciding which
    /// variables to clamp between stages.
    pub fn set_log_prior_ratio_f64(&mut self, log_prior_ratios: Array1<f64>) {
        self.bp_decoder.set_log_prior_ratio_f64(log_prior_ratios);
    }

    fn init_dismem_state(relay_config: &RelayDecoderConfig) -> PosteriorUpdateState {
        let rng_std: rand::prelude::StdRng = rand::rngs::StdRng::seed_from_u64(relay_config.seed);
        let c_damp_uniform = relay_config
            .c_damp_dist_interval
            .map(|(low, high)| Uniform::new(low, high));
        PosteriorUpdateState {
            rng_std,
            c_damp_uniform,
        }
    }

    fn apply_stage_edge_damping(&mut self, stage_idx: usize) {
        if let Some(message_mix) = self.relay_config.message_mix_bernoulli {
            self.bp_decoder.clear_explicit_c_damp_messages();
            let mut fresh_coefficients = Array1::zeros(self.check_matrix().nnz());
            let mut previous_coefficients = Array1::zeros(self.check_matrix().nnz());
            let mut second_previous_coefficients = message_mix
                .has_second_previous()
                .then(|| Array1::zeros(self.check_matrix().nnz()));
            for edge_idx in 0..fresh_coefficients.len() {
                let (sampled_fresh, sampled_previous, sampled_second_previous) =
                    message_mix.sample(&mut self.posterior_update_state.rng_std);
                fresh_coefficients[edge_idx] = sampled_fresh;
                previous_coefficients[edge_idx] = sampled_previous;
                if let (Some(coefficients), Some(sampled)) = (
                    second_previous_coefficients.as_mut(),
                    sampled_second_previous,
                ) {
                    coefficients[edge_idx] = sampled;
                }
            }
            if let Some(second_previous_coefficients) = second_previous_coefficients {
                self.bp_decoder
                    .set_explicit_second_order_message_mix_coefficients_f64(
                        fresh_coefficients,
                        previous_coefficients,
                        second_previous_coefficients,
                    );
            } else {
                self.bp_decoder.set_explicit_message_mix_coefficients_f64(
                    fresh_coefficients,
                    previous_coefficients,
                );
            }
            return;
        }
        self.bp_decoder.clear_explicit_message_mix_coefficients();
        if let Some(explicit_c_damp_messages) = self.relay_config.explicit_c_damp_messages.as_ref()
        {
            let row = explicit_c_damp_messages.row(stage_idx).to_owned();
            self.bp_decoder.set_explicit_c_damp_messages_f64(row);
            return;
        }
        if let Some(c_damp_uniform) = self.posterior_update_state.c_damp_uniform.as_ref() {
            let mut c_damp_messages = Array1::zeros(self.check_matrix().nnz());
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
        let mut gammas = Array1::zeros(self.check_matrix().cols());
        if let Some(explicit_gammas) = self.relay_config.explicit_gammas.as_ref() {
            let gammas_num_sets = explicit_gammas.shape()[0];
            for (i, gamma_ref) in gammas.iter_mut().enumerate() {
                *gamma_ref = *explicit_gammas
                    .get((set_idx % gammas_num_sets, i))
                    .expect("index within explicit_gammas bounds");
            }
            self.bp_decoder.set_memory_strengths_f64(gammas);
            return;
        }
        let gamma_sampler = self.relay_config.effective_gamma_sampler();
        for i in 0..gammas.len() {
            gammas[i] = gamma_sampler.sample(&mut self.posterior_update_state.rng_std);
        }
        self.bp_decoder.set_memory_strengths_f64(gammas);
    }

    fn prepare_initial_stage(&mut self) {
        self.bp_decoder.initialize_decoder();
        self.apply_stage_edge_damping(0);
    }

    fn prepare_next_leg(&mut self, set_idx: usize) {
        self.init_next_set(set_idx);
        self.apply_stage_edge_damping(set_idx);
        self.bp_decoder.current_iteration = 0;
        self.bp_decoder.initialize_check_to_variable();
        self.bp_decoder.initialize_variable_to_check();
        if !self.relay_config.relay_posteriors {
            self.bp_decoder.set_posterior_ratios_to_priors();
        }
    }

    /// Decode with the inner decoder
    fn decode_inner(&mut self, detectors: ArrayView1<Bit>, max_iter: usize) -> DecodeResult {
        let mut success: bool = false;
        let mut decoded_detectors = Array1::default(detectors.dim());

        for _ in 0..max_iter {
            self.bp_decoder.run_iteration(detectors);
            decoded_detectors = self.bp_decoder.compute_decoded_detectors();
            success = self
                .bp_decoder
                .check_convergence(detectors, decoded_detectors.view());

            // If we have converged may now exit
            if success {
                debug!(
                    "Succeeded on iteration {:?}",
                    self.bp_decoder.current_iteration
                );
                break;
            }
            self.bp_decoder.current_iteration += 1;
        }

        self.bp_decoder
            .build_result(success, decoded_detectors, max_iter)
    }

    fn write_log(&mut self, file: File) {
        let mut buf_writer = BufWriter::new(file);
        for set in 0..self.num_executed_sets {
            let log_line = format!(
                "{}, {}, {}, {}\n",
                (set - 1) as i32,
                self.sets_iter[set],
                self.sets_conv[set] as u8,
                self.sets_best[set] as u8
            );
            buf_writer
                .write_all(log_line.as_bytes())
                .expect("Unable to write Relay logging data.");
        }
        buf_writer
            .flush()
            .expect("Unable to write Relay logging data.");
    }
}

impl<N> Decoder for RelayDecoder<N>
where
    N: PartialEq
        + Debug
        + Default
        + Clone
        + Copy
        + Signed
        + Bounded
        + FromPrimitive
        + ToPrimitive
        + std::cmp::PartialOrd
        + std::ops::Add
        + std::ops::AddAssign
        + std::ops::DivAssign
        + std::ops::Mul<N>
        + std::ops::MulAssign
        + Send
        + Sync
        + std::fmt::Display
        + 'static,
{
    fn check_matrix(&self) -> Arc<SparseBitMatrix> {
        self.bp_decoder.check_matrix()
    }

    fn log_prior_ratios(&mut self) -> Array1<f64> {
        self.bp_decoder.log_prior_ratios()
    }

    fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult {
        // Initialization
        let mut num_conv = 0;
        let mut min_pm = f64::MAX;
        let mut num_sets_best = 0;
        let mut best_set_idx = 0;
        let mut total_iterations: usize = 0;
        self.num_executed_sets = 0;
        let stopping_criterion = self.relay_config.stopping_criterion.clone();

        // First Mem-BP
        self.prepare_initial_stage();
        let mut result = self.decode_inner(detectors, self.relay_config.pre_iter);
        self.num_executed_sets = 1;

        // Create logging variables and log first set if applicable
        if self.relay_config.logging {
            self.sets_iter[0] = result.iterations;
            self.sets_conv[0] = result.success;
        }

        // Check early stopping criteria
        if result.success {
            num_conv += 1;
            min_pm = result.decoding_quality;
            num_sets_best += 1;
            if self.relay_config.logging {
                self.sets_quality[0] = result.decoding_quality
            };

            let mut done = false;
            if stopping_criterion == StoppingCriterion::PreIter {
                done = true;
            } else if let StoppingCriterion::NConv { stop_after } = stopping_criterion {
                if num_conv >= stop_after {
                    done = true;
                }
            }
            // If stopping criterion has been met: Log (if applicable) and return
            if done {
                if self.relay_config.logging {
                    self.sets_best[0] = true;
                    let file = OpenOptions::new()
                        .append(true)
                        .open("relay_logging.out")
                        .unwrap();
                    self.write_log(file);
                }
                return result;
            }
        }

        // Init and loop over all Relay sets
        total_iterations += result.iterations;
        for set in 1..=self.relay_config.num_sets {
            // Do not completely initialize decoder as we wish to relay
            // posterior marginals with new memory strengths.
            self.prepare_next_leg(set);
            let temp_result = self.decode_inner(detectors, self.relay_config.set_max_iter);

            self.num_executed_sets += 1;
            total_iterations += temp_result.iterations;
            if temp_result.success {
                num_conv += 1;
                let pm = temp_result.decoding_quality;
                if self.relay_config.logging {
                    self.sets_conv[set] = true;
                    self.sets_iter[set] = temp_result.iterations;
                    self.sets_quality[set] = pm;
                }
                if pm == min_pm {
                    // Count how often we found the best solution
                    num_sets_best += 1;
                }
                if pm < min_pm {
                    // Found a new best solution
                    num_sets_best = 1;
                    best_set_idx = set;
                    min_pm = pm;
                    result = temp_result;
                }
                if let StoppingCriterion::NConv { stop_after } = stopping_criterion {
                    if num_conv >= stop_after {
                        break;
                    }
                }
            }
        }
        result.iterations = total_iterations;

        // Rest of the function is just logging
        if self.relay_config.logging {
            if num_sets_best == 1 {
                self.sets_best[best_set_idx] = true;
            }
            let file = OpenOptions::new()
                .append(true)
                .open("relay_logging.out")
                .unwrap();
            self.write_log(file);
        }

        result
    }

    fn get_decoding_quality(&mut self, errors: ArrayView1<u8>) -> f64 {
        self.bp_decoder.get_decoding_quality(errors)
    }
}

impl<N> DecoderRunner for RelayDecoder<N> where
    N: PartialEq
        + Debug
        + Default
        + Clone
        + Copy
        + Signed
        + Bounded
        + FromPrimitive
        + ToPrimitive
        + std::cmp::PartialOrd
        + std::ops::Add
        + std::ops::AddAssign
        + std::ops::DivAssign
        + std::ops::Mul<N>
        + std::ops::MulAssign
        + Send
        + Sync
        + std::fmt::Display
        + 'static
{
}

#[cfg(test)]
mod tests {

    use super::*;

    use crate::bipartite_graph::{BipartiteGraph, SparseBipartiteGraph};
    use env_logger;
    use ndarray::prelude::*;

    use crate::dem::DetectorErrorModel;
    use crate::utilities::test::get_test_data_path;
    use ndarray::Array2;
    use ndarray_npy::read_npy;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    // Basic test where Relay is called but only runs 1 BP iteration
    #[test]
    fn min_sum_decode_repetition_code() {
        init();

        // Build 3, 2 qubit repetition code with weight 2 checks
        let check_matrix = array![[1, 1, 0], [0, 1, 1],];

        let check_matrix: SparseBipartiteGraph<_> = SparseBipartiteGraph::from_dense(check_matrix);
        let check_matrix_arc = Arc::new(check_matrix);

        let iterations = 10;
        let bp_config = MinSumDecoderConfig {
            error_priors: array![0.003, 0.003, 0.003],
            max_iter: iterations,
            alpha: Some(1.),
            alpha_iteration_scaling_factor: 1.,
            gamma0: None,
            ..Default::default()
        };
        let bp_config_arc = Arc::new(bp_config);

        let relay_config = RelayDecoderConfig {
            pre_iter: iterations,
            num_sets: 0,
            set_max_iter: 150,
            stopping_criterion: StoppingCriterion::PreIter,
            explicit_gammas: None,
            ..Default::default()
        };
        let relay_config_arc = Arc::new(relay_config);

        let mut decoder: RelayDecoder<f32> =
            RelayDecoder::new(check_matrix_arc, bp_config_arc, relay_config_arc);

        let error = array![0, 0, 0];
        let detectors: Array1<Bit> = array![0, 0];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);

        let error = array![1, 0, 0];
        let detectors: Array1<Bit> = array![1, 0];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);

        let error = array![0, 1, 0];
        let detectors: Array1<Bit> = array![1, 1];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);

        let error = array![0, 0, 1];
        let detectors: Array1<Bit> = array![0, 1];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);
    }

    // Basic test where Relay runs 40 sets
    #[test]
    fn decode_144_12_12() {
        let resources = get_test_data_path();
        let code_144_12_12 =
            DetectorErrorModel::load(resources.join("144_12_12")).expect("Unable to load the code");
        let detectors_144_12_12: Array2<Bit> =
            read_npy(resources.join("144_12_12_detectors.npy")).expect("Unable to open file");
        let bp_config_144_12_12 = MinSumDecoderConfig {
            error_priors: code_144_12_12.error_priors,
            max_iter: 200,
            alpha: None,
            alpha_iteration_scaling_factor: 0.,
            gamma0: Some(0.9),
            ..Default::default()
        };
        let relay_config = RelayDecoderConfig::default();
        let check_matrix = Arc::new(code_144_12_12.detector_error_matrix);
        let bp_config = Arc::new(bp_config_144_12_12);
        let config = Arc::new(relay_config);
        let mut decoder_144_12_12: RelayDecoder<f64> =
            RelayDecoder::new(check_matrix, bp_config, config);
        let num_errors = 100;
        let detectors_slice = detectors_144_12_12.slice(s![..num_errors, ..]);
        let results = decoder_144_12_12.par_decode_detailed_batch(detectors_slice);

        // All should pass for Relay.
        assert!(
            results.iter().map(|x| x.success as usize).sum::<usize>()
                == (detectors_slice.shape()[0])
        );

        assert_eq!(results[0].decoding.len(), 8785);
    }

    // Basic test where Relay runs 40 sets
    #[test]
    fn decode_144_12_12_int() {
        let resources = get_test_data_path();
        let code_144_12_12 =
            DetectorErrorModel::load(resources.join("144_12_12")).expect("Unable to load the code");
        let detectors_144_12_12: Array2<Bit> =
            read_npy(resources.join("144_12_12_detectors.npy")).expect("Unable to open file");

        let bits = 16;
        let scale = 8.0;

        let bp_config_144_12_12 = MinSumDecoderConfig {
            error_priors: code_144_12_12.error_priors,
            max_iter: 200,
            alpha: None,
            alpha_iteration_scaling_factor: 0.,
            gamma0: Some(0.9),
            max_data_value: Some(((1 << bits) - 1) as f64),
            data_scale_value: Some(scale),
            ..Default::default()
        };
        let relay_config = RelayDecoderConfig {
            ..Default::default()
        };
        let check_matrix = Arc::new(code_144_12_12.detector_error_matrix);
        let bp_config = Arc::new(bp_config_144_12_12);
        let config = Arc::new(relay_config);
        let mut decoder_144_12_12: RelayDecoder<isize> =
            RelayDecoder::new(check_matrix, bp_config, config);
        let num_errors = 100;
        let detectors_slice = detectors_144_12_12.slice(s![..num_errors, ..]);
        let results = decoder_144_12_12.par_decode_detailed_batch(detectors_slice);

        // All should pass for Relay.
        assert!(
            results.iter().map(|x| x.success as usize).sum::<usize>()
                == (detectors_slice.shape()[0])
        );

        assert_eq!(results[0].decoding.len(), 8785);
    }

    #[test]
    fn independent_ensembling_resets_posteriors_to_priors() {
        let check_matrix = array![[1, 1, 0], [0, 1, 1]];
        let check_matrix: SparseBipartiteGraph<_> = SparseBipartiteGraph::from_dense(check_matrix);
        let check_matrix_arc = Arc::new(check_matrix);

        let bp_config = Arc::new(MinSumDecoderConfig {
            error_priors: array![0.1, 0.2, 0.3],
            max_iter: 2,
            alpha: Some(1.0),
            alpha_iteration_scaling_factor: 1.0,
            gamma0: Some(0.15),
            ..Default::default()
        });
        let relay_config = Arc::new(RelayDecoderConfig {
            pre_iter: 2,
            num_sets: 1,
            set_max_iter: 2,
            relay_posteriors: false,
            ..Default::default()
        });

        let mut decoder: RelayDecoder<f64> =
            RelayDecoder::new(check_matrix_arc, bp_config.clone(), relay_config);

        let detectors: Array1<Bit> = array![1, 0];
        decoder.bp_decoder.initialize_decoder();
        decoder.bp_decoder.run_iteration(detectors.view());
        let before_reset = decoder.bp_decoder.build_result(
            false,
            decoder.bp_decoder.compute_decoded_detectors(),
            2,
        );
        decoder.prepare_next_leg(1);
        let after_reset = decoder.bp_decoder.build_result(
            false,
            decoder.bp_decoder.compute_decoded_detectors(),
            2,
        );

        assert_ne!(before_reset.posterior_ratios, bp_config.log_prior_ratios());
        assert_eq!(after_reset.posterior_ratios, bp_config.log_prior_ratios());
    }

    #[test]
    fn bernoulli_gamma_sampler_extremes_are_deterministic() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let always_negative = GammaSampler::bernoulli_two_point(-0.2, 0.4, 0.0).unwrap();
        let always_positive = GammaSampler::bernoulli_two_point(-0.2, 0.4, 1.0).unwrap();
        for _ in 0..16 {
            assert_eq!(always_negative.sample(&mut rng), -0.2);
            assert_eq!(always_positive.sample(&mut rng), 0.4);
        }
    }

    #[test]
    fn bernoulli_gamma_sampler_rejects_invalid_values() {
        assert!(GammaSampler::bernoulli_two_point(0.1, 0.4, 0.5).is_err());
        assert!(GammaSampler::bernoulli_two_point(-0.1, -0.4, 0.5).is_err());
        assert!(GammaSampler::bernoulli_two_point(-0.1, 0.4, 1.5).is_err());
    }

    #[test]
    fn message_mix_sampler_extremes_are_independent_and_deterministic() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let sampler = MessageMixBernoulliSampler::new(-0.2, 0.4, 1.0, -0.8, 0.1, 0.0).unwrap();
        for _ in 0..16 {
            assert_eq!(sampler.sample(&mut rng), (0.4, -0.8, None));
        }
    }

    #[test]
    fn message_mix_sampler_second_previous_extremes_are_deterministic() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let sampler = MessageMixBernoulliSampler::with_second_previous(
            -0.2, 0.4, 1.0, -0.8, 0.1, 0.0, -0.3, 0.2, 1.0,
        )
        .unwrap();
        assert!(sampler.has_second_previous());
        for _ in 0..16 {
            assert_eq!(sampler.sample(&mut rng), (0.4, -0.8, Some(0.2)));
        }
    }

    #[test]
    fn message_mix_sampler_rejects_invalid_values() {
        assert!(MessageMixBernoulliSampler::new(0.1, 0.4, 0.5, -0.1, 0.2, 0.5).is_err());
        assert!(MessageMixBernoulliSampler::new(-0.1, -0.4, 0.5, -0.1, 0.2, 0.5).is_err());
        assert!(MessageMixBernoulliSampler::new(-0.1, 0.4, 1.5, -0.1, 0.2, 0.5).is_err());
        assert!(MessageMixBernoulliSampler::new(-0.1, 0.4, 0.5, 0.1, 0.2, 0.5).is_err());
        assert!(MessageMixBernoulliSampler::new(-0.1, 0.4, 0.5, -0.1, -0.2, 0.5).is_err());
        assert!(MessageMixBernoulliSampler::new(-0.1, 0.4, 0.5, -0.1, 0.2, 1.5).is_err());
        assert!(MessageMixBernoulliSampler::with_second_previous(
            -0.1, 0.4, 0.5, -0.1, 0.2, 0.5, 0.1, 0.2, 0.5
        )
        .is_err());
    }
}
