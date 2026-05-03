use std::sync::Arc;

use crate::decoder::get_sprs_bit_matrix_from_python;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use relay_bp::bp::relay::StoppingCriterion;
use relay_bp::decoder::Bit;
use relay_bp::training::{
    train_relayed_bpgd_bernoulli_memory_with_progress,
    train_relayed_bpgd_static_continuous_memory_with_progress,
    train_relayed_bpgd_static_discrete_memory_with_progress, BernoulliCandidateResult,
    BernoulliCemTrainingConfig, BernoulliEvaluationMetrics, BernoulliGammaParams,
    BernoulliGenerationRecord, BernoulliTrainingProblem, BernoulliTrainingProgress,
    BernoulliTrainingResumeState, RelayedBpgdTrainingDecoderConfig,
    StaticContinuousCandidateResult, StaticContinuousGenerationRecord,
    StaticContinuousMemoryTrainingConfig, StaticContinuousTrainingProgress,
    StaticContinuousTrainingResumeState, StaticDiscreteCandidateResult,
    StaticDiscreteGenerationRecord, StaticDiscreteMemoryTrainingConfig,
    StaticDiscreteTrainingProgress, StaticDiscreteTrainingResumeState,
};

#[pyfunction]
#[pyo3(signature = (
    check_matrix,
    observable_matrix,
    error_priors,
    train_detectors,
    train_observables,
    validation_detectors,
    validation_observables,
    alpha=None,
    alpha_iteration_scaling_factor=1.0,
    gamma0=0.1,
    c_damp=None,
    random_decimation_candidates=1,
    pre_iter=80,
    num_sets=300,
    set_max_iter=60,
    relay_posteriors=true,
    stop_nconv=5,
    stopping_criterion="nconv".to_string(),
    decimation_pre_iter=0,
    initial_decimation_percentage=0.0,
    r_low=0,
    t_low=3,
    r_high=0,
    t_high=5,
    negative_min=-0.3,
    negative_max=0.0,
    positive_min=0.0,
    positive_max=0.66,
    probability_min=0.001,
    probability_max=0.999,
    candidate_count=64,
    elite_count=8,
    generations=20,
    distribution_repeats=1,
    local_refinement_steps=3,
    initial_std=1.0,
    std_floor=0.05,
    smoothing=0.7,
    train_max_logical_failures=None,
    validation_max_logical_failures=None,
    seed=0,
    resume_state=None,
    progress_callback=None
))]
#[allow(clippy::too_many_arguments)]
pub fn train_relayed_bpgd_bernoulli_memory_py<'py>(
    py: Python<'py>,
    check_matrix: &Bound<'_, PyAny>,
    observable_matrix: &Bound<'_, PyAny>,
    error_priors: PyReadonlyArray1<'_, f64>,
    train_detectors: PyReadonlyArray2<'_, Bit>,
    train_observables: PyReadonlyArray2<'_, Bit>,
    validation_detectors: PyReadonlyArray2<'_, Bit>,
    validation_observables: PyReadonlyArray2<'_, Bit>,
    alpha: Option<f64>,
    alpha_iteration_scaling_factor: f64,
    gamma0: Option<f64>,
    c_damp: Option<f64>,
    random_decimation_candidates: usize,
    pre_iter: usize,
    num_sets: usize,
    set_max_iter: usize,
    relay_posteriors: bool,
    stop_nconv: usize,
    stopping_criterion: String,
    decimation_pre_iter: usize,
    initial_decimation_percentage: f64,
    r_low: usize,
    t_low: usize,
    r_high: usize,
    t_high: usize,
    negative_min: f64,
    negative_max: f64,
    positive_min: f64,
    positive_max: f64,
    probability_min: f64,
    probability_max: f64,
    candidate_count: usize,
    elite_count: usize,
    generations: usize,
    distribution_repeats: usize,
    local_refinement_steps: usize,
    initial_std: f64,
    std_floor: f64,
    smoothing: f64,
    train_max_logical_failures: Option<usize>,
    validation_max_logical_failures: Option<usize>,
    seed: u64,
    resume_state: Option<&Bound<'_, PyDict>>,
    progress_callback: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let problem = BernoulliTrainingProblem {
        check_matrix: Arc::new(get_sprs_bit_matrix_from_python(py, check_matrix)?),
        observable_matrix: Arc::new(get_sprs_bit_matrix_from_python(py, observable_matrix)?),
        error_priors: error_priors.as_array().to_owned(),
        train_detectors: train_detectors.as_array().to_owned(),
        train_observables: train_observables.as_array().to_owned(),
        validation_detectors: validation_detectors.as_array().to_owned(),
        validation_observables: validation_observables.as_array().to_owned(),
    };
    let decoder_config = RelayedBpgdTrainingDecoderConfig {
        alpha,
        alpha_iteration_scaling_factor,
        gamma0,
        c_damp,
        random_decimation_candidates,
        pre_iter,
        num_sets,
        set_max_iter,
        relay_posteriors,
        stopping_criterion: parse_stopping_criterion(&stopping_criterion, stop_nconv),
        stop_nconv,
        decimation_pre_iter,
        initial_decimation_percentage,
        r_low,
        t_low,
        r_high,
        t_high,
    };
    let training_config = BernoulliCemTrainingConfig {
        negative_min,
        negative_max,
        positive_min,
        positive_max,
        probability_min,
        probability_max,
        candidate_count,
        elite_count,
        generations,
        distribution_repeats,
        local_refinement_steps,
        initial_std,
        std_floor,
        smoothing,
        train_max_logical_failures,
        validation_max_logical_failures,
        seed,
    };
    let resume_state = match resume_state {
        Some(state) => Some(resume_state_from_dict(state)?),
        None => None,
    };
    let mut callback_holder = progress_callback.map(|callback| {
        move |progress: &BernoulliTrainingProgress| -> Result<(), String> {
            Python::with_gil(|py| {
                let payload = progress_to_dict(py, progress).map_err(|err| err.to_string())?;
                callback
                    .call1(py, (payload,))
                    .map_err(|err| err.to_string())?;
                Ok(())
            })
        }
    });
    let callback_ref = callback_holder.as_mut().map(|callback| {
        callback as &mut dyn FnMut(&BernoulliTrainingProgress) -> Result<(), String>
    });
    let result = train_relayed_bpgd_bernoulli_memory_with_progress(
        problem,
        decoder_config,
        training_config,
        resume_state,
        callback_ref,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let payload = PyDict::new(py);
    payload.set_item(
        "best_candidate",
        candidate_to_dict(py, &result.best_candidate)?,
    )?;
    payload.set_item(
        "validation_metrics",
        metrics_to_dict(py, &result.validation_metrics)?,
    )?;
    payload.set_item(
        "generation_records",
        generation_records_to_list(py, &result.generation_records)?,
    )?;
    payload.set_item("final_raw_mean", result.final_raw_mean.to_vec())?;
    payload.set_item("final_raw_std", result.final_raw_std.to_vec())?;
    Ok(payload)
}

#[pyfunction]
#[pyo3(signature = (
    check_matrix,
    observable_matrix,
    error_priors,
    train_detectors,
    train_observables,
    validation_detectors,
    validation_observables,
    alpha=None,
    alpha_iteration_scaling_factor=1.0,
    gamma0=0.1,
    c_damp=None,
    random_decimation_candidates=1,
    pre_iter=80,
    num_sets=300,
    set_max_iter=60,
    relay_posteriors=true,
    stop_nconv=5,
    stopping_criterion="nconv".to_string(),
    decimation_pre_iter=0,
    initial_decimation_percentage=0.0,
    r_low=0,
    t_low=3,
    r_high=0,
    t_high=5,
    negative=-0.0687506131581227,
    positive=0.3557972204473916,
    p_positive=0.5451025293701163,
    candidate_count=64,
    elite_count=8,
    generations=20,
    distribution_repeats=1,
    probability_floor=0.001,
    smoothing=0.7,
    train_max_logical_failures=None,
    validation_max_logical_failures=None,
    seed=0,
    resume_state=None,
    progress_callback=None
))]
#[allow(clippy::too_many_arguments)]
pub fn train_relayed_bpgd_static_discrete_memory_py<'py>(
    py: Python<'py>,
    check_matrix: &Bound<'_, PyAny>,
    observable_matrix: &Bound<'_, PyAny>,
    error_priors: PyReadonlyArray1<'_, f64>,
    train_detectors: PyReadonlyArray2<'_, Bit>,
    train_observables: PyReadonlyArray2<'_, Bit>,
    validation_detectors: PyReadonlyArray2<'_, Bit>,
    validation_observables: PyReadonlyArray2<'_, Bit>,
    alpha: Option<f64>,
    alpha_iteration_scaling_factor: f64,
    gamma0: Option<f64>,
    c_damp: Option<f64>,
    random_decimation_candidates: usize,
    pre_iter: usize,
    num_sets: usize,
    set_max_iter: usize,
    relay_posteriors: bool,
    stop_nconv: usize,
    stopping_criterion: String,
    decimation_pre_iter: usize,
    initial_decimation_percentage: f64,
    r_low: usize,
    t_low: usize,
    r_high: usize,
    t_high: usize,
    negative: f64,
    positive: f64,
    p_positive: f64,
    candidate_count: usize,
    elite_count: usize,
    generations: usize,
    distribution_repeats: usize,
    probability_floor: f64,
    smoothing: f64,
    train_max_logical_failures: Option<usize>,
    validation_max_logical_failures: Option<usize>,
    seed: u64,
    resume_state: Option<&Bound<'_, PyDict>>,
    progress_callback: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let problem = BernoulliTrainingProblem {
        check_matrix: Arc::new(get_sprs_bit_matrix_from_python(py, check_matrix)?),
        observable_matrix: Arc::new(get_sprs_bit_matrix_from_python(py, observable_matrix)?),
        error_priors: error_priors.as_array().to_owned(),
        train_detectors: train_detectors.as_array().to_owned(),
        train_observables: train_observables.as_array().to_owned(),
        validation_detectors: validation_detectors.as_array().to_owned(),
        validation_observables: validation_observables.as_array().to_owned(),
    };
    let decoder_config = RelayedBpgdTrainingDecoderConfig {
        alpha,
        alpha_iteration_scaling_factor,
        gamma0,
        c_damp,
        random_decimation_candidates,
        pre_iter,
        num_sets,
        set_max_iter,
        relay_posteriors,
        stopping_criterion: parse_stopping_criterion(&stopping_criterion, stop_nconv),
        stop_nconv,
        decimation_pre_iter,
        initial_decimation_percentage,
        r_low,
        t_low,
        r_high,
        t_high,
    };
    let training_config = StaticDiscreteMemoryTrainingConfig {
        negative,
        positive,
        p_positive,
        candidate_count,
        elite_count,
        generations,
        distribution_repeats,
        probability_floor,
        smoothing,
        train_max_logical_failures,
        validation_max_logical_failures,
        seed,
    };
    let resume_state = match resume_state {
        Some(state) => Some(static_discrete_resume_state_from_dict(state)?),
        None => None,
    };
    let mut callback_holder = progress_callback.map(|callback| {
        move |progress: &StaticDiscreteTrainingProgress| -> Result<(), String> {
            Python::with_gil(|py| {
                let payload = static_discrete_progress_to_dict(py, progress)
                    .map_err(|err| err.to_string())?;
                callback
                    .call1(py, (payload,))
                    .map_err(|err| err.to_string())?;
                Ok(())
            })
        }
    });
    let callback_ref = callback_holder.as_mut().map(|callback| {
        callback as &mut dyn FnMut(&StaticDiscreteTrainingProgress) -> Result<(), String>
    });
    let result = train_relayed_bpgd_static_discrete_memory_with_progress(
        problem,
        decoder_config,
        training_config,
        resume_state,
        callback_ref,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let payload = PyDict::new(py);
    payload.set_item(
        "best_candidate",
        static_discrete_candidate_to_dict(py, &result.best_candidate)?,
    )?;
    payload.set_item(
        "validation_metrics",
        metrics_to_dict(py, &result.validation_metrics)?,
    )?;
    payload.set_item(
        "generation_records",
        static_discrete_generation_records_to_list(py, &result.generation_records)?,
    )?;
    payload.set_item(
        "final_probability_vector",
        result.final_probability_vector.to_vec(),
    )?;
    Ok(payload)
}

#[pyfunction]
#[pyo3(signature = (
    check_matrix,
    observable_matrix,
    error_priors,
    train_detectors,
    train_observables,
    validation_detectors,
    validation_observables,
    alpha=None,
    alpha_iteration_scaling_factor=1.0,
    gamma0=0.1,
    c_damp=None,
    random_decimation_candidates=1,
    pre_iter=80,
    num_sets=300,
    set_max_iter=60,
    relay_posteriors=true,
    stop_nconv=5,
    stopping_criterion="nconv".to_string(),
    decimation_pre_iter=0,
    initial_decimation_percentage=0.0,
    r_low=0,
    t_low=3,
    r_high=0,
    t_high=5,
    negative=-0.0687506131581227,
    positive=0.3557972204473916,
    p_positive=0.5451025293701163,
    memory_min=-0.3,
    memory_max=0.66,
    candidate_count=64,
    elite_count=8,
    generations=20,
    distribution_repeats=1,
    initialization_candidates=64,
    initial_std=0.05,
    std_floor=0.001,
    smoothing=0.7,
    train_max_logical_failures=None,
    validation_max_logical_failures=None,
    seed=0,
    initial_gammas=None,
    resume_state=None,
    progress_callback=None
))]
#[allow(clippy::too_many_arguments)]
pub fn train_relayed_bpgd_static_continuous_memory_py<'py>(
    py: Python<'py>,
    check_matrix: &Bound<'_, PyAny>,
    observable_matrix: &Bound<'_, PyAny>,
    error_priors: PyReadonlyArray1<'_, f64>,
    train_detectors: PyReadonlyArray2<'_, Bit>,
    train_observables: PyReadonlyArray2<'_, Bit>,
    validation_detectors: PyReadonlyArray2<'_, Bit>,
    validation_observables: PyReadonlyArray2<'_, Bit>,
    alpha: Option<f64>,
    alpha_iteration_scaling_factor: f64,
    gamma0: Option<f64>,
    c_damp: Option<f64>,
    random_decimation_candidates: usize,
    pre_iter: usize,
    num_sets: usize,
    set_max_iter: usize,
    relay_posteriors: bool,
    stop_nconv: usize,
    stopping_criterion: String,
    decimation_pre_iter: usize,
    initial_decimation_percentage: f64,
    r_low: usize,
    t_low: usize,
    r_high: usize,
    t_high: usize,
    negative: f64,
    positive: f64,
    p_positive: f64,
    memory_min: f64,
    memory_max: f64,
    candidate_count: usize,
    elite_count: usize,
    generations: usize,
    distribution_repeats: usize,
    initialization_candidates: usize,
    initial_std: f64,
    std_floor: f64,
    smoothing: f64,
    train_max_logical_failures: Option<usize>,
    validation_max_logical_failures: Option<usize>,
    seed: u64,
    initial_gammas: Option<PyReadonlyArray1<'_, f64>>,
    resume_state: Option<&Bound<'_, PyDict>>,
    progress_callback: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let problem = BernoulliTrainingProblem {
        check_matrix: Arc::new(get_sprs_bit_matrix_from_python(py, check_matrix)?),
        observable_matrix: Arc::new(get_sprs_bit_matrix_from_python(py, observable_matrix)?),
        error_priors: error_priors.as_array().to_owned(),
        train_detectors: train_detectors.as_array().to_owned(),
        train_observables: train_observables.as_array().to_owned(),
        validation_detectors: validation_detectors.as_array().to_owned(),
        validation_observables: validation_observables.as_array().to_owned(),
    };
    let decoder_config = RelayedBpgdTrainingDecoderConfig {
        alpha,
        alpha_iteration_scaling_factor,
        gamma0,
        c_damp,
        random_decimation_candidates,
        pre_iter,
        num_sets,
        set_max_iter,
        relay_posteriors,
        stopping_criterion: parse_stopping_criterion(&stopping_criterion, stop_nconv),
        stop_nconv,
        decimation_pre_iter,
        initial_decimation_percentage,
        r_low,
        t_low,
        r_high,
        t_high,
    };
    let training_config = StaticContinuousMemoryTrainingConfig {
        negative,
        positive,
        p_positive,
        memory_min,
        memory_max,
        candidate_count,
        elite_count,
        generations,
        distribution_repeats,
        initialization_candidates,
        initial_std,
        std_floor,
        smoothing,
        train_max_logical_failures,
        validation_max_logical_failures,
        seed,
        initial_gammas: initial_gammas.map(|gammas| gammas.as_array().to_owned()),
    };
    let resume_state = match resume_state {
        Some(state) => Some(static_continuous_resume_state_from_dict(state)?),
        None => None,
    };
    let mut callback_holder = progress_callback.map(|callback| {
        move |progress: &StaticContinuousTrainingProgress| -> Result<(), String> {
            Python::with_gil(|py| {
                let payload = static_continuous_progress_to_dict(py, progress)
                    .map_err(|err| err.to_string())?;
                callback
                    .call1(py, (payload,))
                    .map_err(|err| err.to_string())?;
                Ok(())
            })
        }
    });
    let callback_ref = callback_holder.as_mut().map(|callback| {
        callback as &mut dyn FnMut(&StaticContinuousTrainingProgress) -> Result<(), String>
    });
    let result = train_relayed_bpgd_static_continuous_memory_with_progress(
        problem,
        decoder_config,
        training_config,
        resume_state,
        callback_ref,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let payload = PyDict::new(py);
    payload.set_item(
        "best_candidate",
        static_continuous_candidate_to_dict(py, &result.best_candidate)?,
    )?;
    payload.set_item(
        "validation_metrics",
        metrics_to_dict(py, &result.validation_metrics)?,
    )?;
    payload.set_item(
        "generation_records",
        static_continuous_generation_records_to_list(py, &result.generation_records)?,
    )?;
    payload.set_item("final_mean_gammas", result.final_mean_gammas.to_vec())?;
    payload.set_item("final_std_gammas", result.final_std_gammas.to_vec())?;
    Ok(payload)
}

fn resume_state_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<BernoulliTrainingResumeState> {
    let completed_generations = required_item(dict, "completed_generations")?.extract()?;
    let raw_mean = extract_triplet(&required_item(dict, "final_raw_mean")?, "final_raw_mean")?;
    let raw_std = extract_triplet(&required_item(dict, "final_raw_std")?, "final_raw_std")?;
    let best_candidate = match dict.get_item("best_candidate")? {
        Some(item) => Some(candidate_from_any(&item)?),
        None => None,
    };
    let generation_records = match dict.get_item("generation_records")? {
        Some(item) => generation_records_from_any(&item)?,
        None => Vec::new(),
    };
    Ok(BernoulliTrainingResumeState {
        completed_generations,
        raw_mean,
        raw_std,
        best_candidate,
        generation_records,
    })
}

fn required_item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("resume_state missing {key:?}"))
    })
}

fn extract_triplet(value: &Bound<'_, PyAny>, label: &str) -> PyResult<[f64; 3]> {
    let values: Vec<f64> = value.extract()?;
    if values.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{label} must contain exactly three values"
        )));
    }
    Ok([values[0], values[1], values[2]])
}

fn candidate_from_any(value: &Bound<'_, PyAny>) -> PyResult<BernoulliCandidateResult> {
    let raw_params = extract_triplet(&value.get_item("raw_params")?, "raw_params")?;
    let params_any = value.get_item("params")?;
    let metrics_any = value.get_item("metrics")?;
    Ok(BernoulliCandidateResult {
        raw_params,
        params: BernoulliGammaParams {
            negative: params_any.get_item("negative")?.extract()?,
            positive: params_any.get_item("positive")?.extract()?,
            p_positive: params_any.get_item("p_positive")?.extract()?,
        },
        metrics: metrics_from_any(&metrics_any)?,
    })
}

fn metrics_from_any(value: &Bound<'_, PyAny>) -> PyResult<BernoulliEvaluationMetrics> {
    Ok(BernoulliEvaluationMetrics {
        shots: value.get_item("shots")?.extract()?,
        repeats: value.get_item("repeats")?.extract()?,
        trials: value.get_item("trials")?.extract()?,
        logical_failures: value.get_item("logical_failures")?.extract()?,
        max_logical_failures: optional_usize_item(value, "max_logical_failures")?,
        stopped_early: optional_bool_item(value, "stopped_early")?.unwrap_or(false),
        logical_failure_rate: value.get_item("logical_failure_rate")?.extract()?,
        convergence_rate: value.get_item("convergence_rate")?.extract()?,
        mean_iterations: value.get_item("mean_iterations")?.extract()?,
    })
}

fn optional_usize_item(value: &Bound<'_, PyAny>, key: &str) -> PyResult<Option<usize>> {
    match value.get_item(key) {
        Ok(item) if item.is_none() => Ok(None),
        Ok(item) => item.extract().map(Some),
        Err(_) => Ok(None),
    }
}

fn optional_bool_item(value: &Bound<'_, PyAny>, key: &str) -> PyResult<Option<bool>> {
    match value.get_item(key) {
        Ok(item) if item.is_none() => Ok(None),
        Ok(item) => item.extract().map(Some),
        Err(_) => Ok(None),
    }
}

fn generation_record_from_any(value: &Bound<'_, PyAny>) -> PyResult<BernoulliGenerationRecord> {
    let raw_mean = extract_triplet(&value.get_item("raw_mean")?, "raw_mean")?;
    let raw_std = extract_triplet(&value.get_item("raw_std")?, "raw_std")?;
    let mean_params_any = value.get_item("mean_params")?;
    Ok(BernoulliGenerationRecord {
        generation: value.get_item("generation")?.extract()?,
        raw_mean,
        raw_std,
        mean_params: BernoulliGammaParams {
            negative: mean_params_any.get_item("negative")?.extract()?,
            positive: mean_params_any.get_item("positive")?.extract()?,
            p_positive: mean_params_any.get_item("p_positive")?.extract()?,
        },
        best_candidate: candidate_from_any(&value.get_item("best_candidate")?)?,
    })
}

fn generation_records_from_any(
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<BernoulliGenerationRecord>> {
    let list = value.downcast::<PyList>()?;
    let mut records = Vec::with_capacity(list.len());
    for item in list.iter() {
        records.push(generation_record_from_any(&item)?);
    }
    Ok(records)
}

fn extract_f64_vector(value: &Bound<'_, PyAny>, label: &str) -> PyResult<ndarray::Array1<f64>> {
    let values: Vec<f64> = value.extract()?;
    if values.iter().any(|item| !item.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{label} values must be finite"
        )));
    }
    Ok(ndarray::Array1::from(values))
}

fn extract_bit_vector(value: &Bound<'_, PyAny>, label: &str) -> PyResult<ndarray::Array1<Bit>> {
    let values: Vec<Bit> = value.extract()?;
    if values.iter().any(|item| *item > 1) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{label} values must be 0 or 1"
        )));
    }
    Ok(ndarray::Array1::from(values))
}

fn static_discrete_candidate_from_any(
    value: &Bound<'_, PyAny>,
) -> PyResult<StaticDiscreteCandidateResult> {
    Ok(StaticDiscreteCandidateResult {
        mask: extract_bit_vector(&value.get_item("mask")?, "mask")?,
        gammas: extract_f64_vector(&value.get_item("gammas")?, "gammas")?,
        metrics: metrics_from_any(&value.get_item("metrics")?)?,
    })
}

fn static_discrete_generation_record_from_any(
    value: &Bound<'_, PyAny>,
) -> PyResult<StaticDiscreteGenerationRecord> {
    Ok(StaticDiscreteGenerationRecord {
        generation: value.get_item("generation")?.extract()?,
        probability_vector: extract_f64_vector(
            &value.get_item("probability_vector")?,
            "probability_vector",
        )?,
        best_candidate: static_discrete_candidate_from_any(&value.get_item("best_candidate")?)?,
    })
}

fn static_discrete_generation_records_from_any(
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<StaticDiscreteGenerationRecord>> {
    let list = value.downcast::<PyList>()?;
    let mut records = Vec::with_capacity(list.len());
    for item in list.iter() {
        records.push(static_discrete_generation_record_from_any(&item)?);
    }
    Ok(records)
}

fn static_discrete_resume_state_from_dict(
    dict: &Bound<'_, PyDict>,
) -> PyResult<StaticDiscreteTrainingResumeState> {
    let completed_generations = required_item(dict, "completed_generations")?.extract()?;
    let probability_vector = extract_f64_vector(
        &required_item(dict, "final_probability_vector")?,
        "final_probability_vector",
    )?;
    let best_candidate = match dict.get_item("best_candidate")? {
        Some(item) => Some(static_discrete_candidate_from_any(&item)?),
        None => None,
    };
    let generation_records = match dict.get_item("generation_records")? {
        Some(item) => static_discrete_generation_records_from_any(&item)?,
        None => Vec::new(),
    };
    Ok(StaticDiscreteTrainingResumeState {
        completed_generations,
        probability_vector,
        best_candidate,
        generation_records,
    })
}

fn static_continuous_candidate_from_any(
    value: &Bound<'_, PyAny>,
) -> PyResult<StaticContinuousCandidateResult> {
    Ok(StaticContinuousCandidateResult {
        gammas: extract_f64_vector(&value.get_item("gammas")?, "gammas")?,
        metrics: metrics_from_any(&value.get_item("metrics")?)?,
    })
}

fn static_continuous_generation_record_from_any(
    value: &Bound<'_, PyAny>,
) -> PyResult<StaticContinuousGenerationRecord> {
    Ok(StaticContinuousGenerationRecord {
        generation: value.get_item("generation")?.extract()?,
        mean_gammas: extract_f64_vector(&value.get_item("mean_gammas")?, "mean_gammas")?,
        std_gammas: extract_f64_vector(&value.get_item("std_gammas")?, "std_gammas")?,
        best_candidate: static_continuous_candidate_from_any(&value.get_item("best_candidate")?)?,
    })
}

fn static_continuous_generation_records_from_any(
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<StaticContinuousGenerationRecord>> {
    let list = value.downcast::<PyList>()?;
    let mut records = Vec::with_capacity(list.len());
    for item in list.iter() {
        records.push(static_continuous_generation_record_from_any(&item)?);
    }
    Ok(records)
}

fn static_continuous_resume_state_from_dict(
    dict: &Bound<'_, PyDict>,
) -> PyResult<StaticContinuousTrainingResumeState> {
    let completed_generations = required_item(dict, "completed_generations")?.extract()?;
    let mean_gammas = extract_f64_vector(
        &required_item(dict, "final_mean_gammas")?,
        "final_mean_gammas",
    )?;
    let std_gammas = extract_f64_vector(
        &required_item(dict, "final_std_gammas")?,
        "final_std_gammas",
    )?;
    let best_candidate = match dict.get_item("best_candidate")? {
        Some(item) => Some(static_continuous_candidate_from_any(&item)?),
        None => None,
    };
    let generation_records = match dict.get_item("generation_records")? {
        Some(item) => static_continuous_generation_records_from_any(&item)?,
        None => Vec::new(),
    };
    Ok(StaticContinuousTrainingResumeState {
        completed_generations,
        mean_gammas,
        std_gammas,
        best_candidate,
        generation_records,
    })
}

fn parse_stopping_criterion(value: &str, stop_nconv: usize) -> StoppingCriterion {
    match value {
        "pre_iter" => StoppingCriterion::PreIter,
        "all" => StoppingCriterion::All,
        "nconv" => StoppingCriterion::NConv {
            stop_after: stop_nconv,
        },
        _ => StoppingCriterion::NConv {
            stop_after: stop_nconv,
        },
    }
}

fn params_to_dict<'py>(
    py: Python<'py>,
    params: &BernoulliGammaParams,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("negative", params.negative)?;
    dict.set_item("positive", params.positive)?;
    dict.set_item("p_positive", params.p_positive)?;
    Ok(dict)
}

fn metrics_to_dict<'py>(
    py: Python<'py>,
    metrics: &BernoulliEvaluationMetrics,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("shots", metrics.shots)?;
    dict.set_item("repeats", metrics.repeats)?;
    dict.set_item("trials", metrics.trials)?;
    dict.set_item("logical_failures", metrics.logical_failures)?;
    dict.set_item("max_logical_failures", metrics.max_logical_failures)?;
    dict.set_item("stopped_early", metrics.stopped_early)?;
    dict.set_item("logical_failure_rate", metrics.logical_failure_rate)?;
    dict.set_item("convergence_rate", metrics.convergence_rate)?;
    dict.set_item("mean_iterations", metrics.mean_iterations)?;
    Ok(dict)
}

fn candidate_to_dict<'py>(
    py: Python<'py>,
    candidate: &BernoulliCandidateResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("raw_params", candidate.raw_params.to_vec())?;
    dict.set_item("params", params_to_dict(py, &candidate.params)?)?;
    dict.set_item("metrics", metrics_to_dict(py, &candidate.metrics)?)?;
    Ok(dict)
}

fn generation_record_to_dict<'py>(
    py: Python<'py>,
    record: &BernoulliGenerationRecord,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("generation", record.generation)?;
    dict.set_item("raw_mean", record.raw_mean.to_vec())?;
    dict.set_item("raw_std", record.raw_std.to_vec())?;
    dict.set_item("mean_params", params_to_dict(py, &record.mean_params)?)?;
    dict.set_item(
        "best_candidate",
        candidate_to_dict(py, &record.best_candidate)?,
    )?;
    Ok(dict)
}

fn generation_records_to_list<'py>(
    py: Python<'py>,
    records: &[BernoulliGenerationRecord],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for record in records {
        list.append(generation_record_to_dict(py, record)?)?;
    }
    Ok(list)
}

fn progress_to_dict<'py>(
    py: Python<'py>,
    progress: &BernoulliTrainingProgress,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("completed_generations", progress.completed_generations)?;
    dict.set_item(
        "generation_record",
        generation_record_to_dict(py, &progress.generation_record)?,
    )?;
    dict.set_item(
        "best_candidate",
        candidate_to_dict(py, &progress.best_candidate)?,
    )?;
    dict.set_item(
        "generation_records",
        generation_records_to_list(py, &progress.generation_records)?,
    )?;
    dict.set_item("final_raw_mean", progress.final_raw_mean.to_vec())?;
    dict.set_item("final_raw_std", progress.final_raw_std.to_vec())?;
    Ok(dict)
}

fn static_discrete_candidate_to_dict<'py>(
    py: Python<'py>,
    candidate: &StaticDiscreteCandidateResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("mask", candidate.mask.to_vec())?;
    dict.set_item("gammas", candidate.gammas.to_vec())?;
    dict.set_item("metrics", metrics_to_dict(py, &candidate.metrics)?)?;
    Ok(dict)
}

fn static_discrete_generation_record_to_dict<'py>(
    py: Python<'py>,
    record: &StaticDiscreteGenerationRecord,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("generation", record.generation)?;
    dict.set_item("probability_vector", record.probability_vector.to_vec())?;
    dict.set_item(
        "best_candidate",
        static_discrete_candidate_to_dict(py, &record.best_candidate)?,
    )?;
    Ok(dict)
}

fn static_discrete_generation_records_to_list<'py>(
    py: Python<'py>,
    records: &[StaticDiscreteGenerationRecord],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for record in records {
        list.append(static_discrete_generation_record_to_dict(py, record)?)?;
    }
    Ok(list)
}

fn static_discrete_progress_to_dict<'py>(
    py: Python<'py>,
    progress: &StaticDiscreteTrainingProgress,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("completed_generations", progress.completed_generations)?;
    dict.set_item(
        "generation_record",
        static_discrete_generation_record_to_dict(py, &progress.generation_record)?,
    )?;
    dict.set_item(
        "best_candidate",
        static_discrete_candidate_to_dict(py, &progress.best_candidate)?,
    )?;
    dict.set_item(
        "generation_records",
        static_discrete_generation_records_to_list(py, &progress.generation_records)?,
    )?;
    dict.set_item(
        "final_probability_vector",
        progress.final_probability_vector.to_vec(),
    )?;
    Ok(dict)
}

fn static_continuous_candidate_to_dict<'py>(
    py: Python<'py>,
    candidate: &StaticContinuousCandidateResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("gammas", candidate.gammas.to_vec())?;
    dict.set_item("metrics", metrics_to_dict(py, &candidate.metrics)?)?;
    Ok(dict)
}

fn static_continuous_generation_record_to_dict<'py>(
    py: Python<'py>,
    record: &StaticContinuousGenerationRecord,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("generation", record.generation)?;
    dict.set_item("mean_gammas", record.mean_gammas.to_vec())?;
    dict.set_item("std_gammas", record.std_gammas.to_vec())?;
    dict.set_item(
        "best_candidate",
        static_continuous_candidate_to_dict(py, &record.best_candidate)?,
    )?;
    Ok(dict)
}

fn static_continuous_generation_records_to_list<'py>(
    py: Python<'py>,
    records: &[StaticContinuousGenerationRecord],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for record in records {
        list.append(static_continuous_generation_record_to_dict(py, record)?)?;
    }
    Ok(list)
}

fn static_continuous_progress_to_dict<'py>(
    py: Python<'py>,
    progress: &StaticContinuousTrainingProgress,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("completed_generations", progress.completed_generations)?;
    dict.set_item(
        "generation_record",
        static_continuous_generation_record_to_dict(py, &progress.generation_record)?,
    )?;
    dict.set_item(
        "best_candidate",
        static_continuous_candidate_to_dict(py, &progress.best_candidate)?,
    )?;
    dict.set_item(
        "generation_records",
        static_continuous_generation_records_to_list(py, &progress.generation_records)?,
    )?;
    dict.set_item("final_mean_gammas", progress.final_mean_gammas.to_vec())?;
    dict.set_item("final_std_gammas", progress.final_std_gammas.to_vec())?;
    Ok(dict)
}

#[pymodule]
pub fn _training<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_relayed_bpgd_bernoulli_memory_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        train_relayed_bpgd_static_discrete_memory_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        train_relayed_bpgd_static_continuous_memory_py,
        m
    )?)?;
    Ok(())
}

pub fn init_training<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    let training_module = PyModule::new(_py, "_relay_bp._training")?;
    _training(_py, &training_module)?;
    m.add("_training", &training_module)?;
    training_module.setattr("__name__", "_training")?;
    _py.import("sys")?
        .getattr("modules")?
        .set_item("_relay_bp._training", &training_module)?;
    Ok(())
}
