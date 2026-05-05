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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StaticMemoryOptimizer {
    Cem,
    Nes,
}

#[derive(Clone, Debug)]
pub struct StaticDiscreteMemoryTrainingConfig {
    pub optimizer: StaticMemoryOptimizer,
    pub negative: f64,
    pub positive: f64,
    pub p_positive: f64,
    pub candidate_count: usize,
    pub elite_count: usize,
    pub generations: usize,
    pub distribution_repeats: usize,
    pub probability_floor: f64,
    pub smoothing: f64,
    pub train_max_logical_failures: Option<usize>,
    pub validation_max_logical_failures: Option<usize>,
    pub seed: u64,
    pub nes_learning_rate: f64,
    pub initial_probability_vector: Option<Array1<f64>>,
    pub initial_mask: Option<Array1<Bit>>,
}

impl Default for StaticDiscreteMemoryTrainingConfig {
    fn default() -> Self {
        Self {
            optimizer: StaticMemoryOptimizer::Cem,
            negative: -0.0687506131581227,
            positive: 0.3557972204473916,
            p_positive: 0.5451025293701163,
            candidate_count: 64,
            elite_count: 8,
            generations: 20,
            distribution_repeats: 1,
            probability_floor: 1.0e-3,
            smoothing: 0.7,
            train_max_logical_failures: None,
            validation_max_logical_failures: None,
            seed: 0,
            nes_learning_rate: 0.15,
            initial_probability_vector: None,
            initial_mask: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StaticContinuousMemoryTrainingConfig {
    pub optimizer: StaticMemoryOptimizer,
    pub negative: f64,
    pub positive: f64,
    pub p_positive: f64,
    pub memory_min: f64,
    pub memory_max: f64,
    pub candidate_count: usize,
    pub elite_count: usize,
    pub generations: usize,
    pub distribution_repeats: usize,
    pub initialization_candidates: usize,
    pub initial_std: f64,
    pub std_floor: f64,
    pub smoothing: f64,
    pub train_max_logical_failures: Option<usize>,
    pub validation_max_logical_failures: Option<usize>,
    pub seed: u64,
    pub initial_gammas: Option<Array1<f64>>,
    pub nes_learning_rate: f64,
    pub nes_sigma_learning_rate: f64,
    pub nes_impact_decay: f64,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub initial_impact_gammas: Option<Array1<f64>>,
}

impl Default for StaticContinuousMemoryTrainingConfig {
    fn default() -> Self {
        Self {
            optimizer: StaticMemoryOptimizer::Cem,
            negative: -0.0687506131581227,
            positive: 0.3557972204473916,
            p_positive: 0.5451025293701163,
            memory_min: -0.3,
            memory_max: 0.66,
            candidate_count: 64,
            elite_count: 8,
            generations: 20,
            distribution_repeats: 1,
            initialization_candidates: 64,
            initial_std: 0.05,
            std_floor: 1.0e-3,
            smoothing: 0.7,
            train_max_logical_failures: None,
            validation_max_logical_failures: None,
            seed: 0,
            initial_gammas: None,
            nes_learning_rate: 0.2,
            nes_sigma_learning_rate: 0.05,
            nes_impact_decay: 0.9,
            sigma_min: 1.0e-4,
            sigma_max: 0.25,
            initial_impact_gammas: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StaticDiscreteCandidateResult {
    pub mask: Array1<Bit>,
    pub gammas: Array1<f64>,
    pub metrics: BernoulliEvaluationMetrics,
}

#[derive(Clone, Debug)]
pub struct StaticDiscreteGenerationRecord {
    pub generation: usize,
    pub probability_vector: Array1<f64>,
    pub best_candidate: StaticDiscreteCandidateResult,
}

#[derive(Clone, Debug)]
pub struct StaticDiscreteTrainingResult {
    pub best_candidate: StaticDiscreteCandidateResult,
    pub validation_metrics: BernoulliEvaluationMetrics,
    pub generation_records: Vec<StaticDiscreteGenerationRecord>,
    pub final_probability_vector: Array1<f64>,
}

#[derive(Clone, Debug)]
pub struct StaticDiscreteTrainingResumeState {
    pub completed_generations: usize,
    pub probability_vector: Array1<f64>,
    pub best_candidate: Option<StaticDiscreteCandidateResult>,
    pub generation_records: Vec<StaticDiscreteGenerationRecord>,
}

#[derive(Clone, Debug)]
pub struct StaticDiscreteTrainingProgress {
    pub completed_generations: usize,
    pub generation_record: StaticDiscreteGenerationRecord,
    pub best_candidate: StaticDiscreteCandidateResult,
    pub generation_records: Vec<StaticDiscreteGenerationRecord>,
    pub final_probability_vector: Array1<f64>,
}

#[derive(Clone, Debug)]
pub struct StaticContinuousCandidateResult {
    pub gammas: Array1<f64>,
    pub metrics: BernoulliEvaluationMetrics,
}

#[derive(Clone, Debug)]
pub struct StaticContinuousGenerationRecord {
    pub generation: usize,
    pub mean_gammas: Array1<f64>,
    pub std_gammas: Array1<f64>,
    pub impact_gammas: Array1<f64>,
    pub best_candidate: StaticContinuousCandidateResult,
}

#[derive(Clone, Debug)]
pub struct StaticContinuousTrainingResult {
    pub best_candidate: StaticContinuousCandidateResult,
    pub validation_metrics: BernoulliEvaluationMetrics,
    pub generation_records: Vec<StaticContinuousGenerationRecord>,
    pub final_mean_gammas: Array1<f64>,
    pub final_std_gammas: Array1<f64>,
    pub final_impact_gammas: Array1<f64>,
}

#[derive(Clone, Debug)]
pub struct StaticContinuousTrainingResumeState {
    pub completed_generations: usize,
    pub mean_gammas: Array1<f64>,
    pub std_gammas: Array1<f64>,
    pub impact_gammas: Array1<f64>,
    pub best_candidate: Option<StaticContinuousCandidateResult>,
    pub generation_records: Vec<StaticContinuousGenerationRecord>,
}

#[derive(Clone, Debug)]
pub struct StaticContinuousTrainingProgress {
    pub completed_generations: usize,
    pub generation_record: StaticContinuousGenerationRecord,
    pub best_candidate: StaticContinuousCandidateResult,
    pub generation_records: Vec<StaticContinuousGenerationRecord>,
    pub final_mean_gammas: Array1<f64>,
    pub final_std_gammas: Array1<f64>,
    pub final_impact_gammas: Array1<f64>,
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
        let candidates =
            sample_generation_candidates(&raw_mean, &raw_std, generation, &training_config);
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
            raw_std[dim] =
                ((1.0 - rho) * raw_std[dim] + rho * elite_std[dim]).max(training_config.std_floor);
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
        best = refine_candidate(&problem, &decoder_config, &training_config, best, raw_std);
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

pub fn train_relayed_bpgd_static_discrete_memory(
    problem: BernoulliTrainingProblem,
    decoder_config: RelayedBpgdTrainingDecoderConfig,
    training_config: StaticDiscreteMemoryTrainingConfig,
) -> Result<StaticDiscreteTrainingResult, String> {
    train_relayed_bpgd_static_discrete_memory_with_progress(
        problem,
        decoder_config,
        training_config,
        None,
        None,
    )
}

pub fn train_relayed_bpgd_static_discrete_memory_with_progress(
    problem: BernoulliTrainingProblem,
    decoder_config: RelayedBpgdTrainingDecoderConfig,
    training_config: StaticDiscreteMemoryTrainingConfig,
    resume_state: Option<StaticDiscreteTrainingResumeState>,
    mut progress_callback: Option<
        &mut dyn FnMut(&StaticDiscreteTrainingProgress) -> Result<(), String>,
    >,
) -> Result<StaticDiscreteTrainingResult, String> {
    if training_config.optimizer == StaticMemoryOptimizer::Nes {
        return train_relayed_bpgd_static_discrete_memory_nes_with_progress(
            problem,
            decoder_config,
            training_config,
            resume_state,
            progress_callback,
        );
    }
    validate_problem(&problem)?;
    validate_static_discrete_training_config(&training_config)?;
    let num_variables = problem.check_matrix.cols();

    let start_generation = resume_state
        .as_ref()
        .map(|state| state.completed_generations)
        .unwrap_or(0);
    if start_generation > training_config.generations {
        return Err("resume completed_generations exceeds requested generations.".to_string());
    }
    let mut probability_vector = match resume_state.as_ref() {
        Some(state) => {
            validate_static_vector_len(
                "resume probability_vector",
                state.probability_vector.len(),
                num_variables,
            )?;
            if !state
                .probability_vector
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
            {
                return Err("resume probability_vector values must lie in [0, 1].".to_string());
            }
            state.probability_vector.clone()
        }
        None => initial_static_discrete_probability_vector(&training_config, num_variables)?,
    };
    clamp_probability_vector(&mut probability_vector, training_config.probability_floor);
    let mut generation_records = resume_state
        .as_ref()
        .map(|state| state.generation_records.clone())
        .unwrap_or_default();
    let mut best_overall = resume_state.and_then(|state| state.best_candidate);
    if best_overall.is_none() {
        best_overall = evaluate_initial_static_discrete_mask(
            &problem,
            &decoder_config,
            &training_config,
            training_config.seed.wrapping_add(4_000_003),
        )?;
    }

    for generation in start_generation..training_config.generations {
        let masks = sample_static_discrete_masks(&probability_vector, generation, &training_config);
        let mut results = evaluate_static_discrete_masks(
            &problem,
            &decoder_config,
            &training_config,
            &masks,
            generation,
            false,
        );
        results.sort_by(compare_static_discrete_candidate_results);
        let generation_best = results
            .first()
            .cloned()
            .ok_or_else(|| "No static discrete candidates were evaluated.".to_string())?;
        if best_overall
            .as_ref()
            .map(|best| is_better_static_discrete_candidate(&generation_best, best))
            .unwrap_or(true)
        {
            best_overall = Some(generation_best.clone());
        }

        let elite_count = training_config.elite_count.max(1).min(results.len());
        let elite_probabilities =
            fit_static_discrete_elites(&results[..elite_count], num_variables);
        let rho = training_config.smoothing;
        for idx in 0..num_variables {
            probability_vector[idx] =
                (1.0 - rho) * probability_vector[idx] + rho * elite_probabilities[idx];
        }
        clamp_probability_vector(&mut probability_vector, training_config.probability_floor);

        generation_records.push(StaticDiscreteGenerationRecord {
            generation,
            probability_vector: probability_vector.clone(),
            best_candidate: generation_best,
        });
        if let (Some(callback), Some(best_candidate), Some(generation_record)) = (
            progress_callback.as_deref_mut(),
            best_overall.as_ref(),
            generation_records.last(),
        ) {
            callback(&StaticDiscreteTrainingProgress {
                completed_generations: generation + 1,
                generation_record: generation_record.clone(),
                best_candidate: best_candidate.clone(),
                generation_records: generation_records.clone(),
                final_probability_vector: probability_vector.clone(),
            })?;
        }
    }

    let best = best_overall.ok_or_else(|| "Training produced no candidate.".to_string())?;
    let validation_metrics = evaluate_static_gammas(
        &problem,
        &decoder_config,
        best.gammas.view(),
        training_config.distribution_repeats,
        training_config.train_max_logical_failures,
        training_config.validation_max_logical_failures,
        training_config.seed.wrapping_add(9_000_000),
        true,
    );

    Ok(StaticDiscreteTrainingResult {
        best_candidate: best,
        validation_metrics,
        generation_records,
        final_probability_vector: probability_vector,
    })
}

pub fn train_relayed_bpgd_static_continuous_memory(
    problem: BernoulliTrainingProblem,
    decoder_config: RelayedBpgdTrainingDecoderConfig,
    training_config: StaticContinuousMemoryTrainingConfig,
) -> Result<StaticContinuousTrainingResult, String> {
    train_relayed_bpgd_static_continuous_memory_with_progress(
        problem,
        decoder_config,
        training_config,
        None,
        None,
    )
}

pub fn train_relayed_bpgd_static_continuous_memory_with_progress(
    problem: BernoulliTrainingProblem,
    decoder_config: RelayedBpgdTrainingDecoderConfig,
    training_config: StaticContinuousMemoryTrainingConfig,
    resume_state: Option<StaticContinuousTrainingResumeState>,
    mut progress_callback: Option<
        &mut dyn FnMut(&StaticContinuousTrainingProgress) -> Result<(), String>,
    >,
) -> Result<StaticContinuousTrainingResult, String> {
    if training_config.optimizer == StaticMemoryOptimizer::Nes {
        return train_relayed_bpgd_static_continuous_memory_nes_with_progress(
            problem,
            decoder_config,
            training_config,
            resume_state,
            progress_callback,
        );
    }
    validate_problem(&problem)?;
    validate_static_continuous_training_config(&training_config)?;
    let num_variables = problem.check_matrix.cols();

    let start_generation = resume_state
        .as_ref()
        .map(|state| state.completed_generations)
        .unwrap_or(0);
    if start_generation > training_config.generations {
        return Err("resume completed_generations exceeds requested generations.".to_string());
    }
    let mut mean_gammas = match resume_state.as_ref() {
        Some(state) => {
            validate_static_vector_len(
                "resume mean_gammas",
                state.mean_gammas.len(),
                num_variables,
            )?;
            state.mean_gammas.clone()
        }
        None => initialize_static_continuous_mean(&problem, &decoder_config, &training_config)?,
    };
    let mut std_gammas = match resume_state.as_ref() {
        Some(state) => {
            validate_static_vector_len("resume std_gammas", state.std_gammas.len(), num_variables)?;
            state.std_gammas.clone()
        }
        None => Array1::from_elem(num_variables, training_config.initial_std),
    };
    let impact_gammas = match resume_state.as_ref() {
        Some(state) => {
            validate_static_vector_len(
                "resume impact_gammas",
                state.impact_gammas.len(),
                num_variables,
            )?;
            state.impact_gammas.clone()
        }
        None => Array1::zeros(num_variables),
    };
    validate_finite_vector("mean_gammas", mean_gammas.view())?;
    validate_finite_vector("std_gammas", std_gammas.view())?;
    validate_finite_vector("impact_gammas", impact_gammas.view())?;
    clamp_gamma_vector(
        &mut mean_gammas,
        training_config.memory_min,
        training_config.memory_max,
    );
    for std in &mut std_gammas {
        *std = std.max(training_config.std_floor);
    }
    let mut generation_records = resume_state
        .as_ref()
        .map(|state| state.generation_records.clone())
        .unwrap_or_default();
    let mut best_overall = resume_state.and_then(|state| state.best_candidate);
    if best_overall.is_none() {
        best_overall = Some(evaluate_static_continuous_candidate(
            &problem,
            &decoder_config,
            &training_config,
            mean_gammas.view(),
            training_config.seed.wrapping_add(4_000_003),
            false,
        ));
    }

    for generation in start_generation..training_config.generations {
        let candidates = sample_static_continuous_candidates(
            &mean_gammas,
            &std_gammas,
            generation,
            &training_config,
        );
        let mut results = evaluate_static_continuous_gammas(
            &problem,
            &decoder_config,
            &training_config,
            &candidates,
            generation,
            false,
        );
        results.sort_by(compare_static_continuous_candidate_results);
        let generation_best = results
            .first()
            .cloned()
            .ok_or_else(|| "No static continuous candidates were evaluated.".to_string())?;
        if best_overall
            .as_ref()
            .map(|best| is_better_static_continuous_candidate(&generation_best, best))
            .unwrap_or(true)
        {
            best_overall = Some(generation_best.clone());
        }

        let elite_count = training_config.elite_count.max(1).min(results.len());
        let (elite_mean, elite_std) = fit_static_continuous_elites(
            &results[..elite_count],
            num_variables,
            training_config.std_floor,
        );
        let rho = training_config.smoothing;
        for idx in 0..num_variables {
            mean_gammas[idx] = (1.0 - rho) * mean_gammas[idx] + rho * elite_mean[idx];
            std_gammas[idx] = ((1.0 - rho) * std_gammas[idx] + rho * elite_std[idx])
                .max(training_config.std_floor);
        }
        clamp_gamma_vector(
            &mut mean_gammas,
            training_config.memory_min,
            training_config.memory_max,
        );

        generation_records.push(StaticContinuousGenerationRecord {
            generation,
            mean_gammas: mean_gammas.clone(),
            std_gammas: std_gammas.clone(),
            impact_gammas: impact_gammas.clone(),
            best_candidate: generation_best,
        });
        if let (Some(callback), Some(best_candidate), Some(generation_record)) = (
            progress_callback.as_deref_mut(),
            best_overall.as_ref(),
            generation_records.last(),
        ) {
            callback(&StaticContinuousTrainingProgress {
                completed_generations: generation + 1,
                generation_record: generation_record.clone(),
                best_candidate: best_candidate.clone(),
                generation_records: generation_records.clone(),
                final_mean_gammas: mean_gammas.clone(),
                final_std_gammas: std_gammas.clone(),
                final_impact_gammas: impact_gammas.clone(),
            })?;
        }
    }

    let best = best_overall.ok_or_else(|| "Training produced no candidate.".to_string())?;
    let validation_metrics = evaluate_static_gammas(
        &problem,
        &decoder_config,
        best.gammas.view(),
        training_config.distribution_repeats,
        training_config.train_max_logical_failures,
        training_config.validation_max_logical_failures,
        training_config.seed.wrapping_add(9_000_000),
        true,
    );

    Ok(StaticContinuousTrainingResult {
        best_candidate: best,
        validation_metrics,
        generation_records,
        final_mean_gammas: mean_gammas,
        final_std_gammas: std_gammas,
        final_impact_gammas: impact_gammas,
    })
}

fn train_relayed_bpgd_static_discrete_memory_nes_with_progress(
    problem: BernoulliTrainingProblem,
    decoder_config: RelayedBpgdTrainingDecoderConfig,
    training_config: StaticDiscreteMemoryTrainingConfig,
    resume_state: Option<StaticDiscreteTrainingResumeState>,
    mut progress_callback: Option<
        &mut dyn FnMut(&StaticDiscreteTrainingProgress) -> Result<(), String>,
    >,
) -> Result<StaticDiscreteTrainingResult, String> {
    validate_problem(&problem)?;
    validate_static_discrete_training_config(&training_config)?;
    let num_variables = problem.check_matrix.cols();
    let start_generation = resume_state
        .as_ref()
        .map(|state| state.completed_generations)
        .unwrap_or(0);
    if start_generation > training_config.generations {
        return Err("resume completed_generations exceeds requested generations.".to_string());
    }

    let mut probability_vector = match resume_state.as_ref() {
        Some(state) => {
            validate_static_vector_len(
                "resume probability_vector",
                state.probability_vector.len(),
                num_variables,
            )?;
            state.probability_vector.clone()
        }
        None => initial_static_discrete_probability_vector(&training_config, num_variables)?,
    };
    clamp_probability_vector(&mut probability_vector, training_config.probability_floor);
    let mut logits = probability_vector.mapv(logit_from_probability);
    let mut generation_records = resume_state
        .as_ref()
        .map(|state| state.generation_records.clone())
        .unwrap_or_default();
    let mut best_overall = resume_state.and_then(|state| state.best_candidate);
    if best_overall.is_none() {
        best_overall = evaluate_initial_static_discrete_mask(
            &problem,
            &decoder_config,
            &training_config,
            training_config.seed.wrapping_add(4_000_003),
        )?;
    }

    for generation in start_generation..training_config.generations {
        let anchor_mask = best_overall
            .as_ref()
            .map(|candidate| candidate.mask.clone())
            .unwrap_or_else(|| static_discrete_threshold_mask(&probability_vector));
        let masks = sample_static_discrete_nes_masks(
            &probability_vector,
            &anchor_mask,
            generation,
            &training_config,
        );
        let results = evaluate_static_discrete_masks(
            &problem,
            &decoder_config,
            &training_config,
            &masks,
            generation,
            false,
        );
        let mut ranked_results = results.clone();
        ranked_results.sort_by(compare_static_discrete_candidate_results);
        let generation_best = ranked_results
            .first()
            .cloned()
            .ok_or_else(|| "No static discrete NES candidates were evaluated.".to_string())?;
        let canonical_candidates = ranked_results
            .iter()
            .take(training_config.elite_count.min(3))
            .map(|candidate| candidate.mask.clone())
            .collect::<Vec<_>>();
        for candidate_mask in canonical_candidates {
            let candidate = evaluate_static_discrete_mask_candidate(
                &problem,
                &decoder_config,
                &training_config,
                candidate_mask.view(),
                static_discrete_selection_seed(&training_config),
                false,
            );
            if best_overall
                .as_ref()
                .map(|best| is_better_static_discrete_candidate(&candidate, best))
                .unwrap_or(true)
            {
                best_overall = Some(candidate);
            }
        }

        let utilities = rank_utilities_for_discrete_results(&results);
        update_static_discrete_logits_from_utilities(
            &mut logits,
            &probability_vector,
            &results,
            &utilities,
            training_config.nes_learning_rate,
        );
        probability_vector = logits.mapv(probability_from_logit);
        clamp_probability_vector(&mut probability_vector, training_config.probability_floor);
        if let Some(best_candidate) = best_overall.as_ref() {
            pull_static_discrete_probabilities_toward_anchor(
                &mut probability_vector,
                best_candidate.mask.view(),
                training_config.probability_floor,
                training_config.nes_learning_rate,
            );
        }
        logits = probability_vector.mapv(logit_from_probability);

        generation_records.push(StaticDiscreteGenerationRecord {
            generation,
            probability_vector: probability_vector.clone(),
            best_candidate: generation_best,
        });
        if let (Some(callback), Some(best_candidate), Some(generation_record)) = (
            progress_callback.as_deref_mut(),
            best_overall.as_ref(),
            generation_records.last(),
        ) {
            callback(&StaticDiscreteTrainingProgress {
                completed_generations: generation + 1,
                generation_record: generation_record.clone(),
                best_candidate: best_candidate.clone(),
                generation_records: generation_records.clone(),
                final_probability_vector: probability_vector.clone(),
            })?;
        }
    }

    let best = best_overall.ok_or_else(|| "Training produced no candidate.".to_string())?;
    let validation_metrics = evaluate_static_gammas(
        &problem,
        &decoder_config,
        best.gammas.view(),
        training_config.distribution_repeats,
        training_config.train_max_logical_failures,
        training_config.validation_max_logical_failures,
        training_config.seed.wrapping_add(9_000_000),
        true,
    );
    Ok(StaticDiscreteTrainingResult {
        best_candidate: best,
        validation_metrics,
        generation_records,
        final_probability_vector: probability_vector,
    })
}

fn train_relayed_bpgd_static_continuous_memory_nes_with_progress(
    problem: BernoulliTrainingProblem,
    decoder_config: RelayedBpgdTrainingDecoderConfig,
    training_config: StaticContinuousMemoryTrainingConfig,
    resume_state: Option<StaticContinuousTrainingResumeState>,
    mut progress_callback: Option<
        &mut dyn FnMut(&StaticContinuousTrainingProgress) -> Result<(), String>,
    >,
) -> Result<StaticContinuousTrainingResult, String> {
    validate_problem(&problem)?;
    validate_static_continuous_training_config(&training_config)?;
    let num_variables = problem.check_matrix.cols();
    let start_generation = resume_state
        .as_ref()
        .map(|state| state.completed_generations)
        .unwrap_or(0);
    if start_generation > training_config.generations {
        return Err("resume completed_generations exceeds requested generations.".to_string());
    }

    let mut mean_gammas = match resume_state.as_ref() {
        Some(state) => {
            validate_static_vector_len(
                "resume mean_gammas",
                state.mean_gammas.len(),
                num_variables,
            )?;
            state.mean_gammas.clone()
        }
        None => initialize_static_continuous_mean(&problem, &decoder_config, &training_config)?,
    };
    let mut std_gammas = match resume_state.as_ref() {
        Some(state) => {
            validate_static_vector_len("resume std_gammas", state.std_gammas.len(), num_variables)?;
            state.std_gammas.clone()
        }
        None => Array1::from_elem(
            num_variables,
            training_config
                .initial_std
                .clamp(training_config.sigma_min, training_config.sigma_max),
        ),
    };
    let mut impact_gammas = match resume_state.as_ref() {
        Some(state) => {
            validate_static_vector_len(
                "resume impact_gammas",
                state.impact_gammas.len(),
                num_variables,
            )?;
            state.impact_gammas.clone()
        }
        None => initialize_static_continuous_impact(&training_config, num_variables)?,
    };
    validate_finite_vector("mean_gammas", mean_gammas.view())?;
    validate_finite_vector("std_gammas", std_gammas.view())?;
    validate_finite_vector("impact_gammas", impact_gammas.view())?;
    clamp_gamma_vector(
        &mut mean_gammas,
        training_config.memory_min,
        training_config.memory_max,
    );
    clamp_gamma_vector(
        &mut std_gammas,
        training_config.sigma_min,
        training_config.sigma_max,
    );

    let mut generation_records = resume_state
        .as_ref()
        .map(|state| state.generation_records.clone())
        .unwrap_or_default();
    let mut best_overall = resume_state.and_then(|state| state.best_candidate);
    if best_overall.is_none() {
        best_overall = Some(evaluate_static_continuous_candidate(
            &problem,
            &decoder_config,
            &training_config,
            mean_gammas.view(),
            training_config.seed.wrapping_add(4_000_003),
            false,
        ));
    }

    for generation in start_generation..training_config.generations {
        let (candidates, pairs) = sample_static_continuous_nes_candidates(
            &mean_gammas,
            &std_gammas,
            generation,
            &training_config,
        );
        let results = evaluate_static_continuous_gammas(
            &problem,
            &decoder_config,
            &training_config,
            &candidates,
            generation,
            false,
        );
        let mut ranked_results = results.clone();
        ranked_results.sort_by(compare_static_continuous_candidate_results);
        let generation_best = ranked_results
            .first()
            .cloned()
            .ok_or_else(|| "No static continuous NES candidates were evaluated.".to_string())?;
        if best_overall
            .as_ref()
            .map(|best| is_better_static_continuous_candidate(&generation_best, best))
            .unwrap_or(true)
        {
            best_overall = Some(generation_best.clone());
        }

        let utilities = rank_utilities_for_continuous_results(&results);
        let delta = continuous_nes_delta(num_variables, &pairs, &utilities);
        apply_continuous_nes_update(
            &mut mean_gammas,
            &mut std_gammas,
            &mut impact_gammas,
            &delta,
            &training_config,
        );

        generation_records.push(StaticContinuousGenerationRecord {
            generation,
            mean_gammas: mean_gammas.clone(),
            std_gammas: std_gammas.clone(),
            impact_gammas: impact_gammas.clone(),
            best_candidate: generation_best,
        });
        if let (Some(callback), Some(best_candidate), Some(generation_record)) = (
            progress_callback.as_deref_mut(),
            best_overall.as_ref(),
            generation_records.last(),
        ) {
            callback(&StaticContinuousTrainingProgress {
                completed_generations: generation + 1,
                generation_record: generation_record.clone(),
                best_candidate: best_candidate.clone(),
                generation_records: generation_records.clone(),
                final_mean_gammas: mean_gammas.clone(),
                final_std_gammas: std_gammas.clone(),
                final_impact_gammas: impact_gammas.clone(),
            })?;
        }
    }

    let best = best_overall.ok_or_else(|| "Training produced no candidate.".to_string())?;
    let validation_metrics = evaluate_static_gammas(
        &problem,
        &decoder_config,
        best.gammas.view(),
        training_config.distribution_repeats,
        training_config.train_max_logical_failures,
        training_config.validation_max_logical_failures,
        training_config.seed.wrapping_add(9_000_000),
        true,
    );
    Ok(StaticContinuousTrainingResult {
        best_candidate: best,
        validation_metrics,
        generation_records,
        final_mean_gammas: mean_gammas,
        final_std_gammas: std_gammas,
        final_impact_gammas: impact_gammas,
    })
}

fn validate_static_discrete_training_config(
    config: &StaticDiscreteMemoryTrainingConfig,
) -> Result<(), String> {
    validate_static_seed(config.negative, config.positive, config.p_positive)?;
    validate_static_cem_counts(
        config.candidate_count,
        config.elite_count,
        config.generations,
    )?;
    if !(0.0..0.5).contains(&config.probability_floor) || !config.probability_floor.is_finite() {
        return Err("probability_floor must be finite and in [0, 0.5).".to_string());
    }
    if !(0.0..=1.0).contains(&config.smoothing) || !config.smoothing.is_finite() {
        return Err("smoothing must be finite and in [0, 1].".to_string());
    }
    if config.nes_learning_rate <= 0.0 || !config.nes_learning_rate.is_finite() {
        return Err("nes_learning_rate must be positive and finite.".to_string());
    }
    if let Some(probability_vector) = config.initial_probability_vector.as_ref() {
        validate_finite_vector("initial_probability_vector", probability_vector.view())?;
        if probability_vector
            .iter()
            .any(|value| !(0.0..=1.0).contains(value))
        {
            return Err("initial_probability_vector values must lie in [0, 1].".to_string());
        }
    }
    if let Some(mask) = config.initial_mask.as_ref() {
        if mask.iter().any(|bit| *bit > 1) {
            return Err("initial_mask values must be 0 or 1.".to_string());
        }
    }
    validate_failure_limits(
        config.train_max_logical_failures,
        config.validation_max_logical_failures,
    )
}

fn validate_static_continuous_training_config(
    config: &StaticContinuousMemoryTrainingConfig,
) -> Result<(), String> {
    validate_static_seed(config.negative, config.positive, config.p_positive)?;
    validate_static_cem_counts(
        config.candidate_count,
        config.elite_count,
        config.generations,
    )?;
    if !config.memory_min.is_finite()
        || !config.memory_max.is_finite()
        || config.memory_min > config.memory_max
    {
        return Err(
            "memory bounds must be finite and satisfy memory_min <= memory_max.".to_string(),
        );
    }
    if config.initialization_candidates == 0 {
        return Err("initialization_candidates must be positive.".to_string());
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
    if config.nes_learning_rate <= 0.0 || !config.nes_learning_rate.is_finite() {
        return Err("nes_learning_rate must be positive and finite.".to_string());
    }
    if config.nes_sigma_learning_rate < 0.0 || !config.nes_sigma_learning_rate.is_finite() {
        return Err("nes_sigma_learning_rate must be non-negative and finite.".to_string());
    }
    if !(0.0..1.0).contains(&config.nes_impact_decay) || !config.nes_impact_decay.is_finite() {
        return Err("nes_impact_decay must be finite and in [0, 1).".to_string());
    }
    if config.sigma_min <= 0.0
        || !config.sigma_min.is_finite()
        || config.sigma_max < config.sigma_min
        || !config.sigma_max.is_finite()
    {
        return Err(
            "sigma bounds must be finite and satisfy 0 < sigma_min <= sigma_max.".to_string(),
        );
    }
    if let Some(initial_gammas) = config.initial_gammas.as_ref() {
        validate_finite_vector("initial_gammas", initial_gammas.view())?;
    }
    if let Some(initial_impact_gammas) = config.initial_impact_gammas.as_ref() {
        validate_finite_vector("initial_impact_gammas", initial_impact_gammas.view())?;
    }
    validate_failure_limits(
        config.train_max_logical_failures,
        config.validation_max_logical_failures,
    )
}

fn validate_static_seed(negative: f64, positive: f64, p_positive: f64) -> Result<(), String> {
    if !negative.is_finite() || !positive.is_finite() {
        return Err("static memory coefficients must be finite.".to_string());
    }
    if negative > positive {
        return Err("static memory coefficients must satisfy negative <= positive.".to_string());
    }
    if !(0.0..=1.0).contains(&p_positive) || !p_positive.is_finite() {
        return Err("p_positive must be finite and in [0, 1].".to_string());
    }
    Ok(())
}

fn validate_static_cem_counts(
    candidate_count: usize,
    elite_count: usize,
    generations: usize,
) -> Result<(), String> {
    if candidate_count == 0 {
        return Err("candidate_count must be positive.".to_string());
    }
    if elite_count == 0 {
        return Err("elite_count must be positive.".to_string());
    }
    if elite_count > candidate_count {
        return Err("elite_count must be <= candidate_count.".to_string());
    }
    if generations == 0 {
        return Err("generations must be positive.".to_string());
    }
    Ok(())
}

fn validate_failure_limits(
    train_max_logical_failures: Option<usize>,
    validation_max_logical_failures: Option<usize>,
) -> Result<(), String> {
    if train_max_logical_failures == Some(0) {
        return Err("train_max_logical_failures must be positive when set.".to_string());
    }
    if validation_max_logical_failures == Some(0) {
        return Err("validation_max_logical_failures must be positive when set.".to_string());
    }
    Ok(())
}

fn validate_static_vector_len(label: &str, actual: usize, expected: usize) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "{label} length {actual} must match check-matrix columns {expected}."
        ));
    }
    Ok(())
}

fn validate_finite_vector(label: &str, values: ArrayView1<f64>) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} values must be finite."));
    }
    Ok(())
}

fn initial_static_discrete_probability_vector(
    config: &StaticDiscreteMemoryTrainingConfig,
    num_variables: usize,
) -> Result<Array1<f64>, String> {
    if let Some(probability_vector) = config.initial_probability_vector.as_ref() {
        validate_static_vector_len(
            "initial_probability_vector",
            probability_vector.len(),
            num_variables,
        )?;
        let mut probabilities = probability_vector.clone();
        clamp_probability_vector(&mut probabilities, config.probability_floor);
        return Ok(probabilities);
    }
    if let Some(mask) = config.initial_mask.as_ref() {
        validate_static_vector_len("initial_mask", mask.len(), num_variables)?;
        let low = config.probability_floor.max(0.1).min(0.5);
        let high = (1.0 - config.probability_floor).min(0.9).max(0.5);
        return Ok(mask.mapv(|bit| if bit != 0 { high } else { low }));
    }
    Ok(Array1::from_elem(
        num_variables,
        config
            .p_positive
            .clamp(config.probability_floor, 1.0 - config.probability_floor),
    ))
}

fn evaluate_initial_static_discrete_mask(
    problem: &BernoulliTrainingProblem,
    decoder_config: &RelayedBpgdTrainingDecoderConfig,
    config: &StaticDiscreteMemoryTrainingConfig,
    seed: u64,
) -> Result<Option<StaticDiscreteCandidateResult>, String> {
    let Some(mask) = config.initial_mask.as_ref() else {
        return Ok(None);
    };
    validate_static_vector_len("initial_mask", mask.len(), problem.check_matrix.cols())?;
    let gammas = static_discrete_mask_to_gammas(mask.view(), config.negative, config.positive);
    let metrics = evaluate_static_gammas(
        problem,
        decoder_config,
        gammas.view(),
        config.distribution_repeats,
        config.train_max_logical_failures,
        config.validation_max_logical_failures,
        seed,
        false,
    );
    Ok(Some(StaticDiscreteCandidateResult {
        mask: mask.clone(),
        gammas,
        metrics,
    }))
}

fn static_discrete_selection_seed(config: &StaticDiscreteMemoryTrainingConfig) -> u64 {
    config.seed.wrapping_add(4_000_003)
}

fn static_discrete_threshold_mask(probability_vector: &Array1<f64>) -> Array1<Bit> {
    probability_vector.mapv(|probability| if probability >= 0.5 { 1 } else { 0 })
}

fn sample_static_discrete_masks(
    probability_vector: &Array1<f64>,
    generation: usize,
    config: &StaticDiscreteMemoryTrainingConfig,
) -> Vec<Array1<Bit>> {
    let mut candidates = Vec::with_capacity(config.candidate_count);
    candidates.push(static_discrete_threshold_mask(probability_vector));
    if generation == 0 && candidates.len() < config.candidate_count {
        if let Some(initial_mask) = config.initial_mask.as_ref() {
            if initial_mask.len() == probability_vector.len() {
                candidates.push(initial_mask.clone());
            }
        }
    }
    let mut candidate_idx = 1usize;
    while candidates.len() < config.candidate_count {
        let mut rng = rand::rngs::StdRng::seed_from_u64(
            config
                .seed
                .wrapping_add(1_000_003 * generation as u64)
                .wrapping_add(97_531 * candidate_idx as u64),
        );
        let mask = probability_vector.mapv(
            |probability| {
                if rng.gen::<f64>() < probability {
                    1
                } else {
                    0
                }
            },
        );
        candidates.push(mask);
        candidate_idx += 1;
    }
    candidates
}

fn sample_static_discrete_nes_masks(
    probability_vector: &Array1<f64>,
    anchor_mask: &Array1<Bit>,
    generation: usize,
    config: &StaticDiscreteMemoryTrainingConfig,
) -> Vec<Array1<Bit>> {
    let mut candidates = Vec::with_capacity(config.candidate_count);
    push_unique_static_discrete_mask(&mut candidates, anchor_mask.clone());
    if generation == 0 {
        if let Some(initial_mask) = config.initial_mask.as_ref() {
            if initial_mask.len() == probability_vector.len() {
                push_unique_static_discrete_mask(&mut candidates, initial_mask.clone());
            }
        }
    }
    push_unique_static_discrete_mask(
        &mut candidates,
        static_discrete_threshold_mask(probability_vector),
    );

    let max_flips = 1 + (generation / 3).min(3);
    let mut candidate_idx = 1usize;
    while candidates.len() < config.candidate_count {
        let mut rng = rand::rngs::StdRng::seed_from_u64(
            config
                .seed
                .wrapping_add(1_000_003 * generation as u64)
                .wrapping_add(97_531 * candidate_idx as u64),
        );
        let flips = 1 + (rng.gen::<usize>() % max_flips.max(1));
        let mut mask = anchor_mask.clone();
        let mut touched = vec![false; mask.len()];
        let mut flipped = 0usize;
        let mut attempts = 0usize;
        while flipped < flips && attempts < flips.saturating_mul(16).max(16) {
            attempts += 1;
            let Some(index) = sample_static_discrete_mutation_index(
                probability_vector,
                anchor_mask.view(),
                &mut rng,
            ) else {
                break;
            };
            if touched[index] {
                continue;
            }
            touched[index] = true;
            mask[index] ^= 1;
            flipped += 1;
        }
        candidates.push(mask);
        candidate_idx += 1;
    }
    candidates
}

fn push_unique_static_discrete_mask(candidates: &mut Vec<Array1<Bit>>, mask: Array1<Bit>) {
    if !candidates.iter().any(|candidate| candidate == &mask) {
        candidates.push(mask);
    }
}

fn sample_static_discrete_mutation_index(
    probability_vector: &Array1<f64>,
    anchor_mask: ArrayView1<Bit>,
    rng: &mut rand::rngs::StdRng,
) -> Option<usize> {
    if probability_vector.is_empty() {
        return None;
    }
    let total_weight = probability_vector
        .iter()
        .zip(anchor_mask.iter())
        .map(|(probability, anchor_bit)| {
            let anchor_probability = if *anchor_bit != 0 {
                *probability
            } else {
                1.0 - *probability
            };
            (1.0 - anchor_probability).max(1.0e-6)
        })
        .sum::<f64>();
    if total_weight <= 0.0 || !total_weight.is_finite() {
        return Some(rng.gen::<usize>() % probability_vector.len());
    }
    let mut draw = rng.gen::<f64>() * total_weight;
    for (idx, (probability, anchor_bit)) in probability_vector
        .iter()
        .zip(anchor_mask.iter())
        .enumerate()
    {
        let anchor_probability = if *anchor_bit != 0 {
            *probability
        } else {
            1.0 - *probability
        };
        draw -= (1.0 - anchor_probability).max(1.0e-6);
        if draw <= 0.0 {
            return Some(idx);
        }
    }
    Some(probability_vector.len() - 1)
}

fn static_discrete_mask_to_gammas(
    mask: ArrayView1<Bit>,
    negative: f64,
    positive: f64,
) -> Array1<f64> {
    mask.mapv(|bit| if bit != 0 { positive } else { negative })
}

fn evaluate_static_discrete_mask_candidate(
    problem: &BernoulliTrainingProblem,
    decoder_config: &RelayedBpgdTrainingDecoderConfig,
    training_config: &StaticDiscreteMemoryTrainingConfig,
    mask: ArrayView1<Bit>,
    seed: u64,
    validation: bool,
) -> StaticDiscreteCandidateResult {
    let gammas =
        static_discrete_mask_to_gammas(mask, training_config.negative, training_config.positive);
    let metrics = evaluate_static_gammas(
        problem,
        decoder_config,
        gammas.view(),
        training_config.distribution_repeats,
        training_config.train_max_logical_failures,
        training_config.validation_max_logical_failures,
        seed,
        validation,
    );
    StaticDiscreteCandidateResult {
        mask: mask.to_owned(),
        gammas,
        metrics,
    }
}

fn evaluate_static_discrete_masks(
    problem: &BernoulliTrainingProblem,
    decoder_config: &RelayedBpgdTrainingDecoderConfig,
    training_config: &StaticDiscreteMemoryTrainingConfig,
    masks: &[Array1<Bit>],
    generation: usize,
    validation: bool,
) -> Vec<StaticDiscreteCandidateResult> {
    masks
        .par_iter()
        .map(|mask| {
            let gammas = static_discrete_mask_to_gammas(
                mask.view(),
                training_config.negative,
                training_config.positive,
            );
            let metrics = evaluate_static_gammas(
                problem,
                decoder_config,
                gammas.view(),
                training_config.distribution_repeats,
                training_config.train_max_logical_failures,
                training_config.validation_max_logical_failures,
                training_config
                    .seed
                    .wrapping_add(5_000_003 * generation as u64),
                validation,
            );
            StaticDiscreteCandidateResult {
                mask: mask.clone(),
                gammas,
                metrics,
            }
        })
        .collect()
}

fn fit_static_discrete_elites(
    elites: &[StaticDiscreteCandidateResult],
    num_variables: usize,
) -> Array1<f64> {
    let mut probabilities = Array1::<f64>::zeros(num_variables);
    for elite in elites {
        for (idx, bit) in elite.mask.iter().enumerate() {
            probabilities[idx] += if *bit != 0 { 1.0 } else { 0.0 };
        }
    }
    let denom = elites.len().max(1) as f64;
    probabilities.mapv_inplace(|value| value / denom);
    probabilities
}

fn clamp_probability_vector(probability_vector: &mut Array1<f64>, probability_floor: f64) {
    for probability in probability_vector {
        *probability = probability.clamp(probability_floor, 1.0 - probability_floor);
    }
}

fn initialize_static_continuous_mean(
    problem: &BernoulliTrainingProblem,
    decoder_config: &RelayedBpgdTrainingDecoderConfig,
    training_config: &StaticContinuousMemoryTrainingConfig,
) -> Result<Array1<f64>, String> {
    let num_variables = problem.check_matrix.cols();
    if let Some(initial_gammas) = training_config.initial_gammas.as_ref() {
        validate_static_vector_len("initial_gammas", initial_gammas.len(), num_variables)?;
        let mut clipped = initial_gammas.clone();
        clamp_gamma_vector(
            &mut clipped,
            training_config.memory_min,
            training_config.memory_max,
        );
        return Ok(clipped);
    }

    let mut candidates = Vec::with_capacity(training_config.initialization_candidates);
    for candidate_idx in 0..training_config.initialization_candidates {
        let mut rng = rand::rngs::StdRng::seed_from_u64(
            training_config
                .seed
                .wrapping_add(13_000_003)
                .wrapping_add(97_531 * candidate_idx as u64),
        );
        let gammas = Array1::from_shape_fn(num_variables, |_| {
            if rng.gen::<f64>() < training_config.p_positive {
                training_config.positive
            } else {
                training_config.negative
            }
        });
        candidates.push(gammas);
    }
    let mut results = evaluate_static_continuous_gammas(
        problem,
        decoder_config,
        training_config,
        &candidates,
        0,
        false,
    );
    results.sort_by(compare_static_continuous_candidate_results);
    results
        .into_iter()
        .next()
        .map(|result| result.gammas)
        .ok_or_else(|| "No static continuous initialization candidates were evaluated.".to_string())
}

fn sample_static_continuous_candidates(
    mean_gammas: &Array1<f64>,
    std_gammas: &Array1<f64>,
    generation: usize,
    config: &StaticContinuousMemoryTrainingConfig,
) -> Vec<Array1<f64>> {
    let mut candidates = Vec::with_capacity(config.candidate_count);
    let mut mean_candidate = mean_gammas.clone();
    clamp_gamma_vector(&mut mean_candidate, config.memory_min, config.memory_max);
    candidates.push(mean_candidate);
    for candidate_idx in 1..config.candidate_count {
        let mut rng = rand::rngs::StdRng::seed_from_u64(
            config
                .seed
                .wrapping_add(1_000_003 * generation as u64)
                .wrapping_add(97_531 * candidate_idx as u64),
        );
        let mut gammas = Array1::zeros(mean_gammas.len());
        for idx in 0..mean_gammas.len() {
            gammas[idx] = mean_gammas[idx]
                + std_gammas[idx].max(config.std_floor) * sample_standard_normal(&mut rng);
        }
        clamp_gamma_vector(&mut gammas, config.memory_min, config.memory_max);
        candidates.push(gammas);
    }
    candidates
}

fn evaluate_static_continuous_gammas(
    problem: &BernoulliTrainingProblem,
    decoder_config: &RelayedBpgdTrainingDecoderConfig,
    training_config: &StaticContinuousMemoryTrainingConfig,
    candidates: &[Array1<f64>],
    generation: usize,
    validation: bool,
) -> Vec<StaticContinuousCandidateResult> {
    candidates
        .par_iter()
        .map(|gammas| {
            let metrics = evaluate_static_gammas(
                problem,
                decoder_config,
                gammas.view(),
                training_config.distribution_repeats,
                training_config.train_max_logical_failures,
                training_config.validation_max_logical_failures,
                training_config
                    .seed
                    .wrapping_add(5_000_003 * generation as u64),
                validation,
            );
            StaticContinuousCandidateResult {
                gammas: gammas.clone(),
                metrics,
            }
        })
        .collect()
}

fn evaluate_static_continuous_candidate(
    problem: &BernoulliTrainingProblem,
    decoder_config: &RelayedBpgdTrainingDecoderConfig,
    training_config: &StaticContinuousMemoryTrainingConfig,
    gammas: ArrayView1<f64>,
    seed: u64,
    validation: bool,
) -> StaticContinuousCandidateResult {
    let gammas = gammas.to_owned();
    let metrics = evaluate_static_gammas(
        problem,
        decoder_config,
        gammas.view(),
        training_config.distribution_repeats,
        training_config.train_max_logical_failures,
        training_config.validation_max_logical_failures,
        seed,
        validation,
    );
    StaticContinuousCandidateResult { gammas, metrics }
}

fn fit_static_continuous_elites(
    elites: &[StaticContinuousCandidateResult],
    num_variables: usize,
    std_floor: f64,
) -> (Array1<f64>, Array1<f64>) {
    let mut mean = Array1::<f64>::zeros(num_variables);
    for elite in elites {
        for idx in 0..num_variables {
            mean[idx] += elite.gammas[idx];
        }
    }
    let denom = elites.len().max(1) as f64;
    mean.mapv_inplace(|value| value / denom);

    let mut variance = Array1::<f64>::zeros(num_variables);
    for elite in elites {
        for idx in 0..num_variables {
            let delta = elite.gammas[idx] - mean[idx];
            variance[idx] += delta * delta;
        }
    }
    variance.mapv_inplace(|value| (value / denom).sqrt().max(std_floor));
    (mean, variance)
}

fn clamp_gamma_vector(gammas: &mut Array1<f64>, low: f64, high: f64) {
    for gamma in gammas {
        *gamma = gamma.clamp(low, high);
    }
}

#[derive(Clone, Debug)]
struct StaticContinuousNesPair {
    plus_index: usize,
    minus_index: usize,
    epsilon: Array1<f64>,
}

fn rank_utilities_for_discrete_results(results: &[StaticDiscreteCandidateResult]) -> Vec<f64> {
    let mut indices: Vec<usize> = (0..results.len()).collect();
    indices.sort_by(|left, right| {
        compare_static_discrete_candidate_results(&results[*left], &results[*right])
    });
    rank_utilities_from_sorted_indices(indices)
}

fn rank_utilities_for_continuous_results(results: &[StaticContinuousCandidateResult]) -> Vec<f64> {
    let mut indices: Vec<usize> = (0..results.len()).collect();
    indices.sort_by(|left, right| {
        compare_static_continuous_candidate_results(&results[*left], &results[*right])
    });
    rank_utilities_from_sorted_indices(indices)
}

fn rank_utilities_from_sorted_indices(indices: Vec<usize>) -> Vec<f64> {
    let count = indices.len();
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![0.0];
    }
    let mut utilities = vec![0.0; count];
    let denom = (count - 1) as f64;
    for (rank, index) in indices.into_iter().enumerate() {
        utilities[index] = (denom - 2.0 * rank as f64) / denom;
    }
    let abs_sum: f64 = utilities.iter().map(|value| value.abs()).sum();
    if abs_sum > 0.0 {
        for utility in utilities.iter_mut() {
            *utility /= abs_sum;
        }
    }
    utilities
}

fn update_static_discrete_logits_from_utilities(
    logits: &mut Array1<f64>,
    probability_vector: &Array1<f64>,
    results: &[StaticDiscreteCandidateResult],
    utilities: &[f64],
    learning_rate: f64,
) {
    let mut gradient = Array1::<f64>::zeros(logits.len());
    for (result, utility) in results.iter().zip(utilities.iter()) {
        for idx in 0..gradient.len() {
            let bit = if result.mask[idx] != 0 { 1.0 } else { 0.0 };
            gradient[idx] += utility * (bit - probability_vector[idx]);
        }
    }
    for idx in 0..logits.len() {
        logits[idx] += learning_rate * gradient[idx];
    }
}

fn pull_static_discrete_probabilities_toward_anchor(
    probability_vector: &mut Array1<f64>,
    anchor_mask: ArrayView1<Bit>,
    probability_floor: f64,
    learning_rate: f64,
) {
    let pull = learning_rate.clamp(0.05, 0.2);
    let low = probability_floor.max(0.05).min(0.5);
    let high = (1.0 - probability_floor).min(0.95).max(0.5);
    for (probability, anchor_bit) in probability_vector.iter_mut().zip(anchor_mask.iter()) {
        let target = if *anchor_bit != 0 { high } else { low };
        *probability = ((1.0 - pull) * *probability + pull * target)
            .clamp(probability_floor, 1.0 - probability_floor);
    }
}

fn initialize_static_continuous_impact(
    config: &StaticContinuousMemoryTrainingConfig,
    num_variables: usize,
) -> Result<Array1<f64>, String> {
    if let Some(initial_impact_gammas) = config.initial_impact_gammas.as_ref() {
        validate_static_vector_len(
            "initial_impact_gammas",
            initial_impact_gammas.len(),
            num_variables,
        )?;
        return Ok(initial_impact_gammas.clone());
    }
    Ok(Array1::zeros(num_variables))
}

fn sample_static_continuous_nes_candidates(
    mean_gammas: &Array1<f64>,
    std_gammas: &Array1<f64>,
    generation: usize,
    config: &StaticContinuousMemoryTrainingConfig,
) -> (Vec<Array1<f64>>, Vec<StaticContinuousNesPair>) {
    let mut candidates = Vec::with_capacity(config.candidate_count);
    let mut pairs = Vec::new();

    let mut mean_candidate = mean_gammas.clone();
    clamp_gamma_vector(&mut mean_candidate, config.memory_min, config.memory_max);
    candidates.push(mean_candidate);

    if generation == 0 && candidates.len() < config.candidate_count {
        if let Some(initial_gammas) = config.initial_gammas.as_ref() {
            if initial_gammas.len() == mean_gammas.len()
                && !vectors_are_nearly_equal(initial_gammas.view(), mean_gammas.view())
            {
                let mut candidate = initial_gammas.clone();
                clamp_gamma_vector(&mut candidate, config.memory_min, config.memory_max);
                candidates.push(candidate);
            }
        }
    }

    let pair_budget = config.candidate_count.saturating_sub(candidates.len()) / 2;
    for pair_idx in 0..pair_budget {
        let mut rng = rand::rngs::StdRng::seed_from_u64(
            config
                .seed
                .wrapping_add(1_000_003 * generation as u64)
                .wrapping_add(193_939 * pair_idx as u64),
        );
        let epsilon =
            Array1::from_shape_fn(mean_gammas.len(), |_| sample_standard_normal(&mut rng));
        let mut plus = Array1::zeros(mean_gammas.len());
        let mut minus = Array1::zeros(mean_gammas.len());
        for idx in 0..mean_gammas.len() {
            let sigma = std_gammas[idx].clamp(config.sigma_min, config.sigma_max);
            plus[idx] = mean_gammas[idx] + sigma * epsilon[idx];
            minus[idx] = mean_gammas[idx] - sigma * epsilon[idx];
        }
        clamp_gamma_vector(&mut plus, config.memory_min, config.memory_max);
        clamp_gamma_vector(&mut minus, config.memory_min, config.memory_max);

        let plus_index = candidates.len();
        candidates.push(plus);
        let minus_index = candidates.len();
        candidates.push(minus);
        pairs.push(StaticContinuousNesPair {
            plus_index,
            minus_index,
            epsilon,
        });
    }

    (candidates, pairs)
}

fn continuous_nes_delta(
    num_variables: usize,
    pairs: &[StaticContinuousNesPair],
    utilities: &[f64],
) -> Array1<f64> {
    let mut delta = Array1::<f64>::zeros(num_variables);
    if pairs.is_empty() {
        return delta;
    }
    for pair in pairs {
        let weight = utilities[pair.plus_index] - utilities[pair.minus_index];
        for idx in 0..num_variables {
            delta[idx] += weight * pair.epsilon[idx];
        }
    }
    delta.mapv_inplace(|value| value / pairs.len() as f64);
    delta
}

fn apply_continuous_nes_update(
    mean_gammas: &mut Array1<f64>,
    std_gammas: &mut Array1<f64>,
    impact_gammas: &mut Array1<f64>,
    delta: &Array1<f64>,
    config: &StaticContinuousMemoryTrainingConfig,
) {
    for idx in 0..mean_gammas.len() {
        let signal = delta[idx];
        mean_gammas[idx] += config.nes_learning_rate * std_gammas[idx] * signal;
        impact_gammas[idx] = config.nes_impact_decay * impact_gammas[idx]
            + (1.0 - config.nes_impact_decay) * signal.abs();
    }
    clamp_gamma_vector(mean_gammas, config.memory_min, config.memory_max);

    let mean_impact =
        impact_gammas.iter().copied().sum::<f64>() / impact_gammas.len().max(1) as f64;
    let baseline = mean_impact.max(1.0e-12);
    for idx in 0..std_gammas.len() {
        let normalized = (impact_gammas[idx] / baseline).clamp(0.0, 10.0);
        let log_scale = (config.nes_sigma_learning_rate * (normalized - 1.0)).clamp(-0.2, 0.2);
        std_gammas[idx] =
            (std_gammas[idx] * log_scale.exp()).clamp(config.sigma_min, config.sigma_max);
    }
}

fn vectors_are_nearly_equal(left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(left_value, right_value)| (left_value - right_value).abs() <= 1.0e-12)
}

fn logit_from_probability(probability: f64) -> f64 {
    let probability = probability.clamp(1.0e-12, 1.0 - 1.0e-12);
    (probability / (1.0 - probability)).ln()
}

fn probability_from_logit(logit: f64) -> f64 {
    sigmoid(logit)
}

fn evaluate_static_gammas(
    problem: &BernoulliTrainingProblem,
    decoder_config: &RelayedBpgdTrainingDecoderConfig,
    gammas: ArrayView1<f64>,
    repeats: usize,
    train_max_logical_failures: Option<usize>,
    validation_max_logical_failures: Option<usize>,
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
    let repeats = repeats.max(1);
    let max_logical_failures = if validation {
        validation_max_logical_failures
    } else {
        train_max_logical_failures
    };
    let mut logical_failures = 0usize;
    let mut converged = 0usize;
    let mut iterations = 0usize;
    let mut trials = 0usize;
    let mut stopped_early = false;

    'repeat_loop: for repeat_idx in 0..repeats {
        let repeat_seed = base_seed.wrapping_add(10_007 * repeat_idx as u64);
        let mut decoder = build_static_decoder(problem, decoder_config, gammas, repeat_seed);
        for (detectors_row, truth_row) in detectors
            .axis_iter(Axis(0))
            .zip(observables.axis_iter(Axis(0)))
        {
            let decode_result = decoder.decode_detailed(detectors_row);
            let predicted = problem.observable_matrix.mul_mod2(&decode_result.decoding);
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

fn build_static_decoder(
    problem: &BernoulliTrainingProblem,
    decoder_config: &RelayedBpgdTrainingDecoderConfig,
    gammas: ArrayView1<f64>,
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
                explicit_gammas: Some(explicit_gammas_from_static_vector(gammas)),
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

fn explicit_gammas_from_static_vector(gammas: ArrayView1<f64>) -> Array2<f64> {
    Array2::from_shape_vec((1, gammas.len()), gammas.to_vec())
        .expect("shape constructed from static gamma vector length")
}

fn compare_static_metrics(
    left: &BernoulliEvaluationMetrics,
    right: &BernoulliEvaluationMetrics,
) -> std::cmp::Ordering {
    left.logical_failure_rate
        .total_cmp(&right.logical_failure_rate)
        .then_with(|| left.mean_iterations.total_cmp(&right.mean_iterations))
        .then_with(|| right.convergence_rate.total_cmp(&left.convergence_rate))
}

fn compare_static_discrete_candidate_results(
    left: &StaticDiscreteCandidateResult,
    right: &StaticDiscreteCandidateResult,
) -> std::cmp::Ordering {
    compare_static_metrics(&left.metrics, &right.metrics)
        .then_with(|| left.mask.iter().cmp(right.mask.iter()))
}

fn is_better_static_discrete_candidate(
    left: &StaticDiscreteCandidateResult,
    right: &StaticDiscreteCandidateResult,
) -> bool {
    compare_static_discrete_candidate_results(left, right).is_lt()
}

fn compare_static_continuous_candidate_results(
    left: &StaticContinuousCandidateResult,
    right: &StaticContinuousCandidateResult,
) -> std::cmp::Ordering {
    compare_static_metrics(&left.metrics, &right.metrics).then_with(|| {
        left.gammas
            .iter()
            .zip(right.gammas.iter())
            .map(|(left_gamma, right_gamma)| left_gamma.total_cmp(right_gamma))
            .find(|ordering| !ordering.is_eq())
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn is_better_static_continuous_candidate(
    left: &StaticContinuousCandidateResult,
    right: &StaticContinuousCandidateResult,
) -> bool {
    compare_static_continuous_candidate_results(left, right).is_lt()
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
        for (detectors_row, truth_row) in detectors
            .axis_iter(Axis(0))
            .zip(observables.axis_iter(Axis(0)))
        {
            let decode_result = decoder.decode_detailed(detectors_row);
            let predicted = problem.observable_matrix.mul_mod2(&decode_result.decoding);
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

fn is_better_candidate(left: &BernoulliCandidateResult, right: &BernoulliCandidateResult) -> bool {
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

    #[test]
    fn static_discrete_elites_update_per_variable_probabilities() {
        let metrics = BernoulliEvaluationMetrics {
            shots: 1,
            repeats: 1,
            trials: 1,
            logical_failures: 0,
            max_logical_failures: None,
            stopped_early: false,
            logical_failure_rate: 0.0,
            convergence_rate: 1.0,
            mean_iterations: 1.0,
        };
        let elites = vec![
            StaticDiscreteCandidateResult {
                mask: Array1::from(vec![1, 0, 1]),
                gammas: Array1::from(vec![0.3, -0.1, 0.3]),
                metrics: metrics.clone(),
            },
            StaticDiscreteCandidateResult {
                mask: Array1::from(vec![1, 1, 0]),
                gammas: Array1::from(vec![0.3, 0.3, -0.1]),
                metrics,
            },
        ];
        let probabilities = fit_static_discrete_elites(&elites, 3);
        assert_eq!(probabilities.to_vec(), vec![1.0, 0.5, 0.5]);
    }

    #[test]
    fn static_gamma_vector_is_exported_as_single_reused_row() {
        let gammas = Array1::from(vec![-0.1, 0.2, 0.3]);
        let explicit = explicit_gammas_from_static_vector(gammas.view());
        assert_eq!(explicit.shape(), &[1, 3]);
        assert_eq!(explicit.row(0).to_vec(), gammas.to_vec());
    }

    #[test]
    fn validates_static_continuous_bounds() {
        let config = StaticContinuousMemoryTrainingConfig {
            memory_min: 1.0,
            memory_max: 0.0,
            ..Default::default()
        };
        assert!(validate_static_continuous_training_config(&config).is_err());
    }

    #[test]
    fn static_discrete_nes_logit_update_moves_toward_better_ranked_mask() {
        let good_metrics = BernoulliEvaluationMetrics {
            shots: 1,
            repeats: 1,
            trials: 1,
            logical_failures: 0,
            max_logical_failures: None,
            stopped_early: false,
            logical_failure_rate: 0.0,
            convergence_rate: 1.0,
            mean_iterations: 1.0,
        };
        let bad_metrics = BernoulliEvaluationMetrics {
            logical_failures: 1,
            logical_failure_rate: 1.0,
            ..good_metrics.clone()
        };
        let results = vec![
            StaticDiscreteCandidateResult {
                mask: Array1::from(vec![1]),
                gammas: Array1::from(vec![0.3]),
                metrics: good_metrics,
            },
            StaticDiscreteCandidateResult {
                mask: Array1::from(vec![0]),
                gammas: Array1::from(vec![-0.1]),
                metrics: bad_metrics,
            },
        ];
        let probability_vector = Array1::from(vec![0.5]);
        let mut logits = Array1::from(vec![0.0]);
        let utilities = rank_utilities_for_discrete_results(&results);
        update_static_discrete_logits_from_utilities(
            &mut logits,
            &probability_vector,
            &results,
            &utilities,
            1.0,
        );
        assert!(probability_from_logit(logits[0]) > 0.5);
    }

    #[test]
    fn static_continuous_nes_pair_update_moves_mean_toward_better_perturbation() {
        let mut mean_gammas = Array1::from(vec![0.0]);
        let mut std_gammas = Array1::from(vec![0.1]);
        let mut impact_gammas = Array1::zeros(1);
        let delta = Array1::from(vec![1.0]);
        let config = StaticContinuousMemoryTrainingConfig {
            memory_min: -1.0,
            memory_max: 1.0,
            nes_learning_rate: 1.0,
            nes_sigma_learning_rate: 0.0,
            nes_impact_decay: 0.5,
            sigma_min: 0.001,
            sigma_max: 1.0,
            ..Default::default()
        };
        apply_continuous_nes_update(
            &mut mean_gammas,
            &mut std_gammas,
            &mut impact_gammas,
            &delta,
            &config,
        );
        assert!(mean_gammas[0] > 0.0);
        assert!(impact_gammas[0] > 0.0);
        assert_eq!(std_gammas[0], 0.1);
    }

    #[test]
    fn static_warm_start_vectors_validate_shape_and_bounds() {
        let discrete = StaticDiscreteMemoryTrainingConfig {
            initial_probability_vector: Some(Array1::from(vec![0.25, 0.75])),
            initial_mask: Some(Array1::from(vec![0, 1])),
            ..Default::default()
        };
        assert!(validate_static_discrete_training_config(&discrete).is_ok());
        let bad_discrete = StaticDiscreteMemoryTrainingConfig {
            initial_probability_vector: Some(Array1::from(vec![1.25])),
            ..Default::default()
        };
        assert!(validate_static_discrete_training_config(&bad_discrete).is_err());

        let continuous = StaticContinuousMemoryTrainingConfig {
            initial_gammas: Some(Array1::from(vec![-0.1, 0.2])),
            initial_impact_gammas: Some(Array1::from(vec![0.0, 1.0])),
            ..Default::default()
        };
        assert!(validate_static_continuous_training_config(&continuous).is_ok());
        let bad_continuous = StaticContinuousMemoryTrainingConfig {
            initial_gammas: Some(Array1::from(vec![f64::NAN])),
            ..Default::default()
        };
        assert!(validate_static_continuous_training_config(&bad_continuous).is_err());
    }
}
