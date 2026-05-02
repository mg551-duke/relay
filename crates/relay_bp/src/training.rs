use crate::bp::relay::{GammaSampler, RelayDecoderConfig, StoppingCriterion};
use crate::bp::relayed_bpgd::{RelayedBPGDDecoder, RelayedBPGDDecoderConfig};
use crate::decoder::{Bit, Decoder, Mod2Mul, SparseBitMatrix};
use ndarray::{Array1, Array2, ArrayView1, Axis};
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct RelayedBpgdTrainingDecoderConfig {
    pub alpha: Option<f64>,
    pub alpha_iteration_scaling_factor: f64,
    pub gamma0: Option<f64>,
    pub c_damp: Option<f64>,
    pub random_decimation_candidates: usize,
    pub pre_iter: usize,
    pub num_sets: usize,
    pub set_max_iter: usize,
    pub relay_posteriors: bool,
    pub stopping_criterion: StoppingCriterion,
    pub stop_nconv: usize,
    pub decimation_pre_iter: usize,
    pub initial_decimation_percentage: f64,
    pub r_low: usize,
    pub t_low: usize,
    pub r_high: usize,
    pub t_high: usize,
}

impl Default for RelayedBpgdTrainingDecoderConfig {
    fn default() -> Self {
        Self {
            alpha: None,
            alpha_iteration_scaling_factor: 1.0,
            gamma0: Some(0.1),
            c_damp: None,
            random_decimation_candidates: 1,
            pre_iter: 80,
            num_sets: 300,
            set_max_iter: 60,
            relay_posteriors: true,
            stopping_criterion: StoppingCriterion::NConv { stop_after: 5 },
            stop_nconv: 5,
            decimation_pre_iter: 0,
            initial_decimation_percentage: 0.0,
            r_low: 0,
            t_low: 3,
            r_high: 0,
            t_high: 5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BernoulliCemTrainingConfig {
    pub negative_min: f64,
    pub negative_max: f64,
    pub positive_min: f64,
    pub positive_max: f64,
    pub probability_min: f64,
    pub probability_max: f64,
    pub candidate_count: usize,
    pub elite_count: usize,
    pub generations: usize,
    pub distribution_repeats: usize,
    pub local_refinement_steps: usize,
    pub initial_std: f64,
    pub std_floor: f64,
    pub smoothing: f64,
    pub train_max_logical_failures: Option<usize>,
    pub validation_max_logical_failures: Option<usize>,
    pub seed: u64,
}

impl Default for BernoulliCemTrainingConfig {
    fn default() -> Self {
        Self {
            negative_min: -0.3,
            negative_max: 0.0,
            positive_min: 0.0,
            positive_max: 0.66,
            probability_min: 1.0e-3,
            probability_max: 1.0 - 1.0e-3,
            candidate_count: 64,
            elite_count: 8,
            generations: 20,
            distribution_repeats: 1,
            local_refinement_steps: 3,
            initial_std: 1.0,
            std_floor: 0.05,
            smoothing: 0.7,
            train_max_logical_failures: None,
            validation_max_logical_failures: None,
            seed: 0,
        }
    }
}

#[derive(Clone)]
pub struct BernoulliTrainingProblem {
    pub check_matrix: Arc<SparseBitMatrix>,
    pub observable_matrix: Arc<SparseBitMatrix>,
    pub error_priors: Array1<f64>,
    pub train_detectors: Array2<Bit>,
    pub train_observables: Array2<Bit>,
    pub validation_detectors: Array2<Bit>,
    pub validation_observables: Array2<Bit>,
}

#[derive(Clone, Copy, Debug)]
pub struct BernoulliGammaParams {
    pub negative: f64,
    pub positive: f64,
    pub p_positive: f64,
}

#[derive(Clone, Debug)]
pub struct BernoulliEvaluationMetrics {
    pub shots: usize,
    pub repeats: usize,
    pub trials: usize,
    pub logical_failures: usize,
    pub max_logical_failures: Option<usize>,
    pub stopped_early: bool,
    pub logical_failure_rate: f64,
    pub convergence_rate: f64,
    pub mean_iterations: f64,
}

#[derive(Clone, Debug)]
pub struct BernoulliCandidateResult {
    pub raw_params: [f64; 3],
    pub params: BernoulliGammaParams,
    pub metrics: BernoulliEvaluationMetrics,
}

#[derive(Clone, Debug)]
pub struct BernoulliGenerationRecord {
    pub generation: usize,
    pub raw_mean: [f64; 3],
    pub raw_std: [f64; 3],
    pub mean_params: BernoulliGammaParams,
    pub best_candidate: BernoulliCandidateResult,
}

#[derive(Clone, Debug)]
pub struct BernoulliTrainingResult {
    pub best_candidate: BernoulliCandidateResult,
    pub validation_metrics: BernoulliEvaluationMetrics,
    pub generation_records: Vec<BernoulliGenerationRecord>,
    pub final_raw_mean: [f64; 3],
    pub final_raw_std: [f64; 3],
}

#[derive(Clone, Debug)]
pub struct BernoulliTrainingResumeState {
    pub completed_generations: usize,
    pub raw_mean: [f64; 3],
    pub raw_std: [f64; 3],
    pub best_candidate: Option<BernoulliCandidateResult>,
    pub generation_records: Vec<BernoulliGenerationRecord>,
}

#[derive(Clone, Debug)]
pub struct BernoulliTrainingProgress {
    pub completed_generations: usize,
    pub generation_record: BernoulliGenerationRecord,
    pub best_candidate: BernoulliCandidateResult,
    pub generation_records: Vec<BernoulliGenerationRecord>,
    pub final_raw_mean: [f64; 3],
    pub final_raw_std: [f64; 3],
}

pub fn train_relayed_bpgd_bernoulli_memory(
    problem: BernoulliTrainingProblem,
    decoder_config: RelayedBpgdTrainingDecoderConfig,
    training_config: BernoulliCemTrainingConfig,
) -> Result<BernoulliTrainingResult, String> {
    train_relayed_bpgd_bernoulli_memory_with_progress(
        problem,
        decoder_config,
        training_config,
        None,
        None,
    )
}

pub fn train_relayed_bpgd_bernoulli_memory_with_progress(
    problem: BernoulliTrainingProblem,
    decoder_config: RelayedBpgdTrainingDecoderConfig,
    training_config: BernoulliCemTrainingConfig,
    resume_state: Option<BernoulliTrainingResumeState>,
    mut progress_callback: Option<&mut dyn FnMut(&BernoulliTrainingProgress) -> Result<(), String>>,
) -> Result<BernoulliTrainingResult, String> {
    validate_problem(&problem)?;
    validate_training_config(&training_config)?;

    let start_generation = resume_state
        .as_ref()
        .map(|state| state.completed_generations)
        .unwrap_or(0);
    if start_generation > training_config.generations {
        return Err("resume completed_generations exceeds requested generations.".to_string());
    }
    let mut raw_mean = resume_state
        .as_ref()
        .map(|state| state.raw_mean)
        .unwrap_or([0.0_f64; 3]);
    let mut raw_std = resume_state
        .as_ref()
        .map(|state| state.raw_std)
        .unwrap_or([training_config.initial_std.max(training_config.std_floor); 3]);
    for value in raw_mean.iter().chain(raw_std.iter()) {
        if !value.is_finite() {
            return Err("resume raw_mean/raw_std values must be finite.".to_string());
        }
    }
    for value in &mut raw_std {
        *value = value.max(training_config.std_floor);
    }
    let mut generation_records = resume_state
        .as_ref()
        .map(|state| state.generation_records.clone())
        .unwrap_or_default();
    let mut best_overall: Option<BernoulliCandidateResult> =
        resume_state.and_then(|state| state.best_candidate);

    for generation in start_generation..training_config.generations {
        let candidates = sample_generation_candidates(&raw_mean, &raw_std, generation, &training_config);
        let mut results = evaluate_candidates(
            &problem,
            &decoder_config,
            &training_config,
            &candidates,
            generation,
            false,
        );
        results.sort_by(compare_candidate_results);
        let generation_best = results
            .first()
            .cloned()
            .ok_or_else(|| "No Bernoulli candidates were evaluated.".to_string())?;
        if best_overall
            .as_ref()
            .map(|best| is_better_candidate(&generation_best, best))
            .unwrap_or(true)
        {
            best_overall = Some(generation_best.clone());
        }

        let elite_count = training_config.elite_count.max(1).min(results.len());
        let elites = &results[..elite_count];
        let (elite_mean, elite_std) = fit_elites(elites, training_config.std_floor);
        let rho = training_config.smoothing;
        for dim in 0..3 {
            raw_mean[dim] = (1.0 - rho) * raw_mean[dim] + rho * elite_mean[dim];
            raw_std[dim] = ((1.0 - rho) * raw_std[dim] + rho * elite_std[dim])
                .max(training_config.std_floor);
        }
        generation_records.push(BernoulliGenerationRecord {
            generation,
            raw_mean,
            raw_std,
            mean_params: params_from_raw(raw_mean, &training_config),
            best_candidate: generation_best,
        });
        if let (Some(callback), Some(best_candidate), Some(generation_record)) = (
            progress_callback.as_deref_mut(),
            best_overall.as_ref(),
            generation_records.last(),
        ) {
            callback(&BernoulliTrainingProgress {
                completed_generations: generation + 1,
                generation_record: generation_record.clone(),
                best_candidate: best_candidate.clone(),
                generation_records: generation_records.clone(),
                final_raw_mean: raw_mean,
                final_raw_std: raw_std,
            })?;
        }
    }

    let mut best = best_overall.ok_or_else(|| "Training produced no candidate.".to_string())?;
    if training_config.local_refinement_steps > 0 {
        best = refine_candidate(
            &problem,
            &decoder_config,
            &training_config,
            best,
            raw_std,
        );
    }
    let validation_metrics = evaluate_params(
        &problem,
        &decoder_config,
        &training_config,
        best.params,
        training_config.seed.wrapping_add(9_000_000),
        true,
    );

    Ok(BernoulliTrainingResult {
        best_candidate: best,
        validation_metrics,
        generation_records,
        final_raw_mean: raw_mean,
        final_raw_std: raw_std,
    })
}

fn validate_problem(problem: &BernoulliTrainingProblem) -> Result<(), String> {
    let n_detectors = problem.check_matrix.rows();
    let n_observables = problem.observable_matrix.rows();
    if problem.error_priors.len() != problem.check_matrix.cols() {
        return Err("error_priors length must match check-matrix columns.".to_string());
    }
    for (label, detectors, observables) in [
        (
            "train",
            &problem.train_detectors,
            &problem.train_observables,
        ),
        (
            "validation",
            &problem.validation_detectors,
            &problem.validation_observables,
        ),
    ] {
        if detectors.nrows() == 0 {
            return Err(format!("{label} detectors must contain at least one shot."));
        }
        if detectors.ncols() != n_detectors {
            return Err(format!(
                "{label} detector columns must match check-matrix rows."
            ));
        }
        if observables.nrows() != detectors.nrows() {
            return Err(format!(
                "{label} observables row count must match detector row count."
            ));
        }
        if observables.ncols() != n_observables {
            return Err(format!(
                "{label} observable columns must match observable-matrix rows."
            ));
        }
    }
    Ok(())
}

fn validate_training_config(config: &BernoulliCemTrainingConfig) -> Result<(), String> {
    if !(config.negative_min.is_finite()
        && config.negative_max.is_finite()
        && config.positive_min.is_finite()
        && config.positive_max.is_finite())
    {
        return Err("Bernoulli weight bounds must be finite.".to_string());
    }
    if config.negative_min > config.negative_max || config.negative_max > 0.0 {
        return Err("negative bounds must satisfy negative_min <= negative_max <= 0.".to_string());
    }
    if config.positive_min < 0.0 || config.positive_min > config.positive_max {
        return Err("positive bounds must satisfy 0 <= positive_min <= positive_max.".to_string());
    }
    if config.probability_min < 0.0
        || config.probability_max > 1.0
        || config.probability_min > config.probability_max
    {
        return Err("probability bounds must lie inside [0, 1].".to_string());
    }
    if config.candidate_count == 0 {
        return Err("candidate_count must be positive.".to_string());
    }
    if config.elite_count == 0 {
        return Err("elite_count must be positive.".to_string());
    }
    if config.generations == 0 {
        return Err("generations must be positive.".to_string());
    }
    if config.initial_std <= 0.0 || !config.initial_std.is_finite() {
        return Err("initial_std must be positive and finite.".to_string());
    }
    if config.std_floor <= 0.0 || !config.std_floor.is_finite() {
        return Err("std_floor must be positive and finite.".to_string());
    }
    if !(0.0..=1.0).contains(&config.smoothing) || !config.smoothing.is_finite() {
        return Err("smoothing must be finite and in [0, 1].".to_string());
    }
    if config.train_max_logical_failures == Some(0) {
        return Err("train_max_logical_failures must be positive when set.".to_string());
    }
    if config.validation_max_logical_failures == Some(0) {
        return Err("validation_max_logical_failures must be positive when set.".to_string());
    }
    Ok(())
}

fn sample_generation_candidates(
    raw_mean: &[f64; 3],
    raw_std: &[f64; 3],
    generation: usize,
    config: &BernoulliCemTrainingConfig,
) -> Vec<[f64; 3]> {
    let mut candidates = Vec::with_capacity(config.candidate_count);
    candidates.push(*raw_mean);
    for candidate_idx in 1..config.candidate_count {
        let mut rng = rand::rngs::StdRng::seed_from_u64(
            config
                .seed
                .wrapping_add(1_000_003 * generation as u64)
                .wrapping_add(97_531 * candidate_idx as u64),
        );
        let mut raw = [0.0; 3];
        for dim in 0..3 {
            raw[dim] = raw_mean[dim] + raw_std[dim] * sample_standard_normal(&mut rng);
        }
        candidates.push(raw);
    }
    candidates
}

fn evaluate_candidates(
    problem: &BernoulliTrainingProblem,
    decoder_config: &RelayedBpgdTrainingDecoderConfig,
    training_config: &BernoulliCemTrainingConfig,
    candidates: &[[f64; 3]],
    generation: usize,
    validation: bool,
) -> Vec<BernoulliCandidateResult> {
    candidates
        .par_iter()
        .map(|raw| {
            let params = params_from_raw(*raw, training_config);
            let seed = training_config
                .seed
                .wrapping_add(5_000_003 * generation as u64);
            let metrics = evaluate_params(
                problem,
                decoder_config,
                training_config,
                params,
                seed,
                validation,
            );
            BernoulliCandidateResult {
                raw_params: *raw,
                params,
                metrics,
            }
        })
        .collect()
}

fn evaluate_params(
    problem: &BernoulliTrainingProblem,
    decoder_config: &RelayedBpgdTrainingDecoderConfig,
    training_config: &BernoulliCemTrainingConfig,
    params: BernoulliGammaParams,
    base_seed: u64,
    validation: bool,
) -> BernoulliEvaluationMetrics {
    let detectors = if validation {
        &problem.validation_detectors
    } else {
        &problem.train_detectors
    };
    let observables = if validation {
        &problem.validation_observables
    } else {
        &problem.train_observables
    };
    let repeats = training_config.distribution_repeats.max(1);
    let max_logical_failures = if validation {
        training_config.validation_max_logical_failures
    } else {
        training_config.train_max_logical_failures
    };
    let mut logical_failures = 0usize;
    let mut converged = 0usize;
    let mut iterations = 0usize;
    let mut trials = 0usize;
    let mut stopped_early = false;

    'repeat_loop: for repeat_idx in 0..repeats {
        let repeat_seed = base_seed.wrapping_add(10_007 * repeat_idx as u64);
        let mut decoder = build_decoder(problem, decoder_config, params, repeat_seed);
        for (detectors_row, truth_row) in detectors.axis_iter(Axis(0)).zip(observables.axis_iter(Axis(0))) {
            let decode_result = decoder.decode_detailed(detectors_row);
            let predicted = problem
                .observable_matrix
                .mul_mod2(&decode_result.decoding);
            if observable_mismatch(predicted.view(), truth_row) {
                logical_failures += 1;
            }
            if decode_result.success {
                converged += 1;
            }
            iterations += decode_result.iterations;
            trials += 1;
            if let Some(limit) = max_logical_failures {
                if logical_failures >= limit {
                    stopped_early = true;
                    break 'repeat_loop;
                }
            }
        }
    }
    let trials_f64 = trials.max(1) as f64;
    BernoulliEvaluationMetrics {
        shots: detectors.nrows(),
        repeats,
        trials,
        logical_failures,
        max_logical_failures,
        stopped_early,
        logical_failure_rate: logical_failures as f64 / trials_f64,
        convergence_rate: converged as f64 / trials_f64,
        mean_iterations: iterations as f64 / trials_f64,
    }
}

fn build_decoder(
    problem: &BernoulliTrainingProblem,
    decoder_config: &RelayedBpgdTrainingDecoderConfig,
    params: BernoulliGammaParams,
    seed: u64,
) -> RelayedBPGDDecoder {
    let stopping_criterion = match decoder_config.stopping_criterion {
        StoppingCriterion::NConv { .. } => StoppingCriterion::NConv {
            stop_after: decoder_config.stop_nconv,
        },
        ref other => other.clone(),
    };
    RelayedBPGDDecoder::new(
        problem.check_matrix.clone(),
        Arc::new(RelayedBPGDDecoderConfig {
            error_priors: problem.error_priors.clone(),
            alpha: decoder_config.alpha,
            alpha_iteration_scaling_factor: decoder_config.alpha_iteration_scaling_factor,
            gamma0: decoder_config.gamma0,
            c_damp: decoder_config.c_damp,
            random_decimation_candidates: decoder_config.random_decimation_candidates,
            relay_config: Arc::new(RelayDecoderConfig {
                pre_iter: decoder_config.pre_iter,
                num_sets: decoder_config.num_sets,
                set_max_iter: decoder_config.set_max_iter,
                gamma_dist_interval: (params.negative, params.positive),
                gamma_sampler: GammaSampler::BernoulliTwoPoint {
                    negative: params.negative,
                    positive: params.positive,
                    p_positive: params.p_positive,
                },
                relay_posteriors: decoder_config.relay_posteriors,
                stopping_criterion,
                seed,
                ..Default::default()
            }),
            decimation_pre_iter: decoder_config.decimation_pre_iter,
            initial_decimation_percentage: decoder_config.initial_decimation_percentage,
            r_low: decoder_config.r_low,
            t_low: decoder_config.t_low,
            r_high: decoder_config.r_high,
            t_high: decoder_config.t_high,
            ..Default::default()
        }),
    )
}

fn observable_mismatch(predicted: ArrayView1<Bit>, truth: ArrayView1<Bit>) -> bool {
    predicted
        .iter()
        .zip(truth.iter())
        .any(|(predicted_bit, truth_bit)| predicted_bit != truth_bit)
}

fn fit_elites(elites: &[BernoulliCandidateResult], std_floor: f64) -> ([f64; 3], [f64; 3]) {
    let mut mean = [0.0; 3];
    for elite in elites {
        for dim in 0..3 {
            mean[dim] += elite.raw_params[dim];
        }
    }
    for dim in 0..3 {
        mean[dim] /= elites.len().max(1) as f64;
    }
    let mut variance = [0.0; 3];
    for elite in elites {
        for dim in 0..3 {
            let delta = elite.raw_params[dim] - mean[dim];
            variance[dim] += delta * delta;
        }
    }
    let mut std = [0.0; 3];
    for dim in 0..3 {
        std[dim] = (variance[dim] / elites.len().max(1) as f64)
            .sqrt()
            .max(std_floor);
    }
    (mean, std)
}

fn refine_candidate(
    problem: &BernoulliTrainingProblem,
    decoder_config: &RelayedBpgdTrainingDecoderConfig,
    training_config: &BernoulliCemTrainingConfig,
    mut best: BernoulliCandidateResult,
    raw_std: [f64; 3],
) -> BernoulliCandidateResult {
    let mut radius = raw_std
        .iter()
        .copied()
        .fold(training_config.std_floor, f64::max);
    for step in 0..training_config.local_refinement_steps {
        let mut raws = Vec::with_capacity(7);
        raws.push(best.raw_params);
        for dim in 0..3 {
            let mut minus = best.raw_params;
            minus[dim] -= radius;
            raws.push(minus);
            let mut plus = best.raw_params;
            plus[dim] += radius;
            raws.push(plus);
        }
        let mut results = evaluate_candidates(
            problem,
            decoder_config,
            training_config,
            &raws,
            training_config.generations + step,
            false,
        );
        results.sort_by(compare_candidate_results);
        if let Some(candidate) = results.into_iter().next() {
            if is_better_candidate(&candidate, &best) {
                best = candidate;
            }
        }
        radius = (radius * 0.5).max(training_config.std_floor);
    }
    best
}

fn compare_candidate_results(
    left: &BernoulliCandidateResult,
    right: &BernoulliCandidateResult,
) -> std::cmp::Ordering {
    left.metrics
        .logical_failure_rate
        .total_cmp(&right.metrics.logical_failure_rate)
        .then_with(|| {
            left.metrics
                .mean_iterations
                .total_cmp(&right.metrics.mean_iterations)
        })
        .then_with(|| left.params.negative.total_cmp(&right.params.negative))
        .then_with(|| left.params.positive.total_cmp(&right.params.positive))
        .then_with(|| left.params.p_positive.total_cmp(&right.params.p_positive))
}

fn is_better_candidate(
    left: &BernoulliCandidateResult,
    right: &BernoulliCandidateResult,
) -> bool {
    compare_candidate_results(left, right).is_lt()
}

fn params_from_raw(raw: [f64; 3], config: &BernoulliCemTrainingConfig) -> BernoulliGammaParams {
    BernoulliGammaParams {
        negative: bounded_sigmoid(raw[0], config.negative_min, config.negative_max),
        positive: bounded_sigmoid(raw[1], config.positive_min, config.positive_max),
        p_positive: bounded_sigmoid(raw[2], config.probability_min, config.probability_max),
    }
}

fn bounded_sigmoid(raw: f64, low: f64, high: f64) -> f64 {
    if high <= low {
        return low;
    }
    low + (high - low) * sigmoid(raw)
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

fn sample_standard_normal(rng: &mut rand::rngs::StdRng) -> f64 {
    let u1 = rng.gen::<f64>().clamp(f64::MIN_POSITIVE, 1.0);
    let u2 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_sigmoid_maps_inside_bounds() {
        let value = bounded_sigmoid(0.0, -0.3, 0.0);
        assert!(value > -0.3);
        assert!(value < 0.0);
    }

    #[test]
    fn validates_invalid_probability_bounds() {
        let config = BernoulliCemTrainingConfig {
            probability_min: -0.1,
            ..Default::default()
        };
        assert!(validate_training_config(&config).is_err());
    }
}
