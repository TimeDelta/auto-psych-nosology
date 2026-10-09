import json

import numpy as np
import polars as pl
import pytest
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit

from run_sparse_rr_baseline import load_baseline_dataset, main, run_baseline
from sparse_rr_logistic import (
    SparseReducedRankLogisticRegression,
    prepare_evidence,
    sparse_low_rank_proximal,
)


def _synthetic_binary_data():
    random_generator = np.random.default_rng(19)
    features = random_generator.normal(size=(80, 5))
    features[:, -1] = 0
    probabilities = expit(1.5 * features[:, 0] - features[:, 1])
    outcome = (random_generator.random(80) < probabilities).astype(float)
    return features, np.column_stack([outcome, outcome])


def _artifact(tmp_path, *, sparse_features=False, **overrides):
    features, targets = _synthetic_binary_data()
    arrays = {
        "targets": targets,
        "observed_mask": np.ones(targets.shape, dtype=bool),
        "evidence_weights": np.ones(targets.shape),
        "perturbation_ids": np.asarray(
            [f"perturbation:{index}" for index in range(80)]
        ),
        "feature_names": np.asarray([f"node:{index}:state" for index in range(5)]),
        "symptom_names": np.asarray(["symptom:a", "symptom:b"]),
        "splits": np.asarray(["train"] * 50 + ["validation"] * 20 + ["test"] * 10),
        "group_ids": np.asarray([f"group:{index}" for index in range(80)]),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "format_version": 1,
                    "graph_snapshot_sha256": "a" * 64,
                    "feature_generator": {
                        "name": "synthetic-test-fixture",
                        "configuration": {},
                    },
                    "feature_fit_perturbation_ids": [],
                }
            )
        ),
    }
    if sparse_features:
        feature_matrix = sparse.csr_matrix(features)
        arrays.update(
            feature_data=feature_matrix.data,
            feature_indices=feature_matrix.indices,
            feature_indptr=feature_matrix.indptr,
            feature_shape=np.asarray(feature_matrix.shape),
        )
    else:
        arrays["features"] = features
    arrays.update(overrides)
    artifact_path = tmp_path / f"dataset_{len(list(tmp_path.iterdir()))}.npz"
    np.savez(artifact_path, **arrays)
    return artifact_path


def test_joint_proximal_matches_sparse_group_lasso_closed_form():
    # With one symptom the nuclear norm is the Euclidean norm and the row
    # penalty is the L1 norm. This has an independent closed-form solution.
    coefficient_column = np.array([[3.0], [0.5], [-2.0]])
    soft_thresholded = np.sign(coefficient_column) * np.maximum(
        np.abs(coefficient_column) - 1.0, 0.0
    )
    expected = soft_thresholded * max(1.0 - 0.7 / np.linalg.norm(soft_thresholded), 0.0)
    actual = sparse_low_rank_proximal(coefficient_column, 1.0, 0.7, tolerance=1e-12)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert actual[1, 0] == 0


def test_joint_proximal_diagonal_singular_values():
    actual = sparse_low_rank_proximal(np.diag([3.0, 1.0]), 0.5, 1.0)
    np.testing.assert_allclose(actual, np.diag([1.5, 0.0]), atol=1e-10)


@pytest.mark.parametrize("use_sparse", [False, True])
def test_masked_weighted_standardized_gradient_matches_finite_differences(use_sparse):
    features = np.array([[2.0, -1.0], [5.0, 3.0], [-2.0, 0.5]])
    targets = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    weights = np.array([[1.0, 0.0], [0.3, 0.7], [0.0, 2.0]])
    weights /= weights.sum()
    model = SparseReducedRankLogisticRegression(ridge_penalty=0.1)
    model.feature_mean_ = features.mean(axis=0)
    model.feature_scale_ = features.std(axis=0)
    if use_sparse:
        features = sparse.csr_matrix(features)
    coefficients = np.array([[0.4, -0.8], [0.1, 0.2]])
    intercepts = np.array([0.3, -0.1])
    _, coefficient_gradient, intercept_gradient = model._smooth_loss_and_gradient(
        features, targets, weights, coefficients, intercepts
    )
    epsilon = 1e-6
    for parameters, gradient in (
        (coefficients, coefficient_gradient),
        (intercepts, intercept_gradient),
    ):
        for index in np.ndindex(parameters.shape):
            original_value = parameters[index]
            parameters[index] = original_value + epsilon
            upper_loss = model._smooth_loss_and_gradient(
                features, targets, weights, coefficients, intercepts, gradient=False
            )[0]
            parameters[index] = original_value - epsilon
            lower_loss = model._smooth_loss_and_gradient(
                features, targets, weights, coefficients, intercepts, gradient=False
            )[0]
            parameters[index] = original_value
            assert gradient[index] == pytest.approx(
                (upper_loss - lower_loss) / (2 * epsilon), abs=1e-8
            )


def test_sparse_standardization_is_stable_for_large_feature_offsets():
    features = np.column_stack([1e8 + np.arange(8), np.arange(8)])
    targets = np.array([[0], [1], [0], [1], [1], [0], [1], [0]], dtype=float)
    dense_model = SparseReducedRankLogisticRegression(row_sparsity=10).fit(
        features, targets
    )
    sparse_model = SparseReducedRankLogisticRegression(row_sparsity=10).fit(
        sparse.csr_matrix(features), targets
    )
    np.testing.assert_allclose(
        dense_model.feature_scale_, sparse_model.feature_scale_, atol=1e-9
    )


def test_iteration_limit_is_reported_as_unconverged():
    features, targets = _synthetic_binary_data()
    model = SparseReducedRankLogisticRegression(max_iter=1).fit(features, targets)
    assert model.model_statistics()["converged"] is False
    assert model.n_iter_ == 1


def test_unpenalized_fit_matches_independent_logistic_optimizer():
    features, targets = _synthetic_binary_data()
    features = features[:, :2]
    model = SparseReducedRankLogisticRegression(
        row_sparsity=0,
        nuclear_penalty=0,
        ridge_penalty=0,
        standardize=False,
        max_iter=2500,
        tolerance=1e-9,
    ).fit(features, targets)
    augmented_features = np.column_stack([features, np.ones(len(features))])

    def objective(parameters):
        logits = augmented_features @ parameters
        losses = np.logaddexp(0.0, logits) - targets[:, 0] * logits
        gradient = (
            augmented_features.T @ (expit(logits) - targets[:, 0]) / len(features)
        )
        return losses.mean(), gradient

    reference = minimize(
        objective, np.zeros(3), jac=True, method="BFGS", options={"gtol": 1e-9}
    )
    np.testing.assert_allclose(model.coef_[:, 0], reference.x[:2], atol=2e-6)
    np.testing.assert_allclose(model.intercept_[0], reference.x[2], atol=2e-6)
    assert model.converged_


def test_fit_reduces_loss_with_sparse_low_rank_coefficients():
    features, targets = _synthetic_binary_data()
    model = SparseReducedRankLogisticRegression(max_iter=1000).fit(features, targets)
    history = np.asarray(model.objective_history_)
    assert history[-1] < history[0] - 0.05
    assert np.all(np.diff(history) <= 1e-12)
    assert model.model_statistics()["coefficient_rank"] == 1
    assert np.all(model.coef_[-1] == 0)
    assert model.converged_


def test_strong_row_penalty_returns_weighted_intercept_only_model():
    targets = np.array([[0.0], [1.0], [1.0], [1.0]])
    weights = np.array([[10.0], [1.0], [1.0], [1.0]])
    features = np.arange(8).reshape(4, 2)
    model = SparseReducedRankLogisticRegression(row_sparsity=10).fit(
        features, targets, evidence_weights=weights
    )
    assert np.all(model.coef_ == 0)
    np.testing.assert_allclose(model.predict_proba(features), 3.0 / 13.0)
    assert model.model_statistics()["coefficient_rank"] == 0


def test_masked_targets_weights_and_unlabelled_rows_do_not_affect_fit():
    features, targets = _synthetic_binary_data()
    mask = np.ones(targets.shape, dtype=bool)
    mask[::3, 0] = False
    mask[-1] = False
    first_model = SparseReducedRankLogisticRegression(max_iter=1000).fit(
        features, targets, observed_mask=mask
    )
    changed_targets = np.where(mask, targets, np.nan)
    changed_weights = np.where(mask, 1.0, np.nan)
    changed_features = features.copy()
    changed_features[-1] = 1e9
    second_model = SparseReducedRankLogisticRegression(max_iter=1000).fit(
        changed_features,
        changed_targets,
        observed_mask=mask,
        evidence_weights=changed_weights,
    )
    np.testing.assert_array_equal(first_model.coef_, second_model.coef_)
    np.testing.assert_array_equal(first_model.feature_mean_, second_model.feature_mean_)
    assert first_model.training_row_count_ == 79


def test_dense_and_csr_fit_agree_without_densifying_feature_matrix(monkeypatch):
    features, targets = _synthetic_binary_data()
    dense_model = SparseReducedRankLogisticRegression(max_iter=1000).fit(
        features, targets
    )

    def forbid_densification(*args, **kwargs):
        raise AssertionError("The sparse feature matrix must not be densified")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbid_densification)
    sparse_model = SparseReducedRankLogisticRegression(max_iter=1000).fit(
        sparse.csr_matrix(features), targets
    )
    np.testing.assert_allclose(dense_model.coef_, sparse_model.coef_, atol=1e-9)
    np.testing.assert_allclose(
        dense_model.predict_proba(features),
        sparse_model.predict_proba(sparse.csr_matrix(features)),
        atol=1e-10,
    )


def test_saved_model_predicts_new_perturbations_and_checks_feature_order(tmp_path):
    features, targets = _synthetic_binary_data()
    feature_names = [f"node:{index}" for index in range(features.shape[1])]
    model = SparseReducedRankLogisticRegression(max_iter=1000).fit(
        features, targets, feature_names=feature_names
    )
    model_path = tmp_path / "model.npz"
    model.save(model_path)
    loaded_model = SparseReducedRankLogisticRegression.load(model_path)
    new_features = np.array([[2.0, -1.0, 0.5, 1.0, 0.0]])
    np.testing.assert_array_equal(
        model.predict_proba(new_features),
        loaded_model.predict_proba(new_features, feature_names=feature_names),
    )
    with pytest.raises(ValueError, match="names/order"):
        loaded_model.predict_proba(new_features, feature_names=feature_names[::-1])


def test_log_loss_stays_finite_and_weighted_loss_is_not_labelled_bits():
    features, targets = _synthetic_binary_data()
    model = SparseReducedRankLogisticRegression(max_iter=1000).fit(features, targets)
    report = model.evaluation_report(
        features * 1e4, targets, evidence_weights=np.full(targets.shape, 0.25)
    )
    assert np.isfinite(report["unweighted_nll_nats"])
    assert report["conditional_data_cost_bits"] == pytest.approx(
        report["unweighted_nll_nats"] / np.log(2)
    )
    assert report["weighted_mean_log_loss"] == pytest.approx(
        report["unweighted_nll_nats"] / targets.size
    )


@pytest.mark.parametrize(
    "argument",
    [
        {"row_sparsity": -1},
        {"nuclear_penalty": np.nan},
        {"ridge_penalty": -1},
        {"initial_step_size": 0},
        {"max_iter": 0},
    ],
)
def test_invalid_optimizer_configuration_fails(argument):
    with pytest.raises(ValueError):
        SparseReducedRankLogisticRegression(**argument)


def test_missing_targets_require_explicit_mask_and_training_evidence():
    with pytest.raises(ValueError, match="binary"):
        prepare_evidence([[1.0, np.nan]])
    with pytest.raises(ValueError, match="boolean"):
        prepare_evidence([[1.0]], [[1]])
    with pytest.raises(ValueError, match="Every fitted symptom"):
        SparseReducedRankLogisticRegression().fit(
            np.ones((2, 1)),
            np.ones((2, 2)),
            observed_mask=np.array([[True, False], [True, False]]),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        prepare_evidence([[1]], evidence_weights=[[-1]])


@pytest.mark.parametrize("sparse_features", [False, True])
def test_cli_artifact_end_to_end_and_test_labels_hidden(tmp_path, sparse_features):
    input_path = _artifact(tmp_path, sparse_features=sparse_features)
    output_path = tmp_path / "result"
    assert (
        main(
            [
                str(input_path),
                str(output_path),
                "--max-iter",
                "1000",
                "--expected-graph-snapshot",
                "a" * 64,
            ]
        )
        == 0
    )
    report = json.loads((output_path / "report.json").read_text())
    manifest = json.loads((output_path / "manifest.json").read_text())
    predictions = pl.read_parquet(output_path / "predictions.parquet")
    assert report["test_evaluated"] is False and "test" not in report
    assert report["statistics"]["training_row_count"] == 50
    test_predictions = predictions.filter(pl.col("split") == "test")
    assert test_predictions["target"].is_null().all()
    assert not test_predictions["included_in_scoring"].any()
    assert len(manifest["input_sha256"]) == 64
    assert set(manifest["outputs"]) == {
        "model.npz",
        "report.json",
        "predictions.parquet",
    }
    with pytest.raises(FileExistsError):
        run_baseline(input_path, output_path)


def test_held_out_features_and_targets_cannot_change_fitted_parameters(tmp_path):
    features, targets = _synthetic_binary_data()
    original_artifact = _artifact(tmp_path)
    modified_targets = targets.copy()
    modified_targets[50:] = 1 - modified_targets[50:]
    modified_features = features.copy()
    modified_features[50:] *= 20
    modified_artifact = _artifact(
        tmp_path, features=modified_features, targets=modified_targets
    )
    for name, input_path in (
        ("original", original_artifact),
        ("changed", modified_artifact),
    ):
        run_baseline(
            input_path, tmp_path / name, model_configuration={"max_iter": 1000}
        )
    original_model = SparseReducedRankLogisticRegression.load(
        tmp_path / "original" / "model.npz"
    )
    changed_model = SparseReducedRankLogisticRegression.load(
        tmp_path / "changed" / "model.npz"
    )
    np.testing.assert_array_equal(original_model.coef_, changed_model.coef_)
    np.testing.assert_array_equal(original_model.intercept_, changed_model.intercept_)
    np.testing.assert_array_equal(
        original_model.feature_mean_, changed_model.feature_mean_
    )


def test_explicit_test_evaluation_and_masked_entries(tmp_path):
    features, targets = _synthetic_binary_data()
    mask = np.ones(targets.shape, dtype=bool)
    mask[-1, 0] = False
    targets[-1, 0] = np.nan
    artifact = _artifact(tmp_path, observed_mask=mask, targets=targets)
    report = run_baseline(
        artifact,
        tmp_path / "scored",
        evaluate_test=True,
        minimum_positives=1,
        model_configuration={"max_iter": 1000},
    )
    assert report["test_evaluated"]
    assert report["test"]["included_pair_count"] == 19


def test_cross_split_group_and_feature_fit_leakage_rejected(tmp_path):
    groups = np.asarray([f"group:{index}" for index in range(80)])
    groups[50] = groups[0]
    with pytest.raises(ValueError, match="crosses predefined splits"):
        load_baseline_dataset(_artifact(tmp_path, group_ids=groups))
    metadata = {
        "format_version": 1,
        "graph_snapshot_sha256": "a" * 64,
        "feature_generator": {"name": "fixture", "configuration": {}},
        "feature_fit_perturbation_ids": ["perturbation:50"],
    }
    with pytest.raises(ValueError, match="outside the training"):
        load_baseline_dataset(
            _artifact(tmp_path, metadata_json=np.asarray(json.dumps(metadata)))
        )


def test_snapshot_mismatch_rejected_before_fit_or_output(tmp_path, monkeypatch):
    artifact = _artifact(tmp_path)

    def forbid_fit(*args, **kwargs):
        raise AssertionError("A mismatched snapshot must not train")

    monkeypatch.setattr(SparseReducedRankLogisticRegression, "fit", forbid_fit)
    with pytest.raises(ValueError, match="expected pin"):
        run_baseline(artifact, tmp_path / "result", expected_graph_snapshot="b" * 64)
    assert not (tmp_path / "result").exists()


def test_duplicate_feature_names_and_invalid_csr_fail(tmp_path):
    with pytest.raises(ValueError, match="feature names"):
        load_baseline_dataset(
            _artifact(tmp_path, feature_names=np.asarray(["same"] * 5))
        )
    with pytest.raises(ValueError, match="indices"):
        load_baseline_dataset(
            _artifact(
                tmp_path,
                sparse_features=True,
                feature_indices=np.ones(320, dtype=int) * 100,
            )
        )
