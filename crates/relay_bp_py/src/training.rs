use std::sync::Arc;

use crate::decoder::get_sprs_bit_matrix_from_python;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use relay_bp::bp::relay::StoppingCriterion;
use relay_bp::decoder::Bit;
use relay_bp::training::{
    train_relayed_bpgd_bernoulli_memory, BernoulliCandidateResult,
    BernoulliCemTrainingConfig, BernoulliEvaluationMetrics, BernoulliGammaParams,
    BernoulliGenerationRecord, BernoulliTrainingProblem, RelayedBpgdTrainingDecoderConfig,
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
    seed=0
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
    seed: u64,
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
        seed,
    };
    let result = train_relayed_bpgd_bernoulli_memory(problem, decoder_config, training_config)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let payload = PyDict::new(py);
    payload.set_item("best_candidate", candidate_to_dict(py, &result.best_candidate)?)?;
    payload.set_item(
        "validation_metrics",
        metrics_to_dict(py, &result.validation_metrics)?,
    )?;
    payload.set_item("generation_records", generation_records_to_list(py, &result.generation_records)?)?;
    payload.set_item("final_raw_mean", result.final_raw_mean.to_vec())?;
    payload.set_item("final_raw_std", result.final_raw_std.to_vec())?;
    Ok(payload)
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
    dict.set_item("best_candidate", candidate_to_dict(py, &record.best_candidate)?)?;
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

#[pymodule]
pub fn _training<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_relayed_bpgd_bernoulli_memory_py, m)?)?;
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
