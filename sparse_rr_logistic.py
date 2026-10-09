"""Sparse reduced-rank logistic regression for masked, multivariate outcomes.

The convex objective combines mean binary log loss, a row-group penalty and a
nuclear-norm penalty. Dykstra's algorithm computes the joint proximal operator;
sequentially thresholding rows and singular values once is not that operator.
No physiology operator or psychiatric label inference is defined here.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.utils.sparsefuncs import mean_variance_axis


def _validated_features(features: Any) -> np.ndarray | sparse.csr_matrix:
    if sparse.issparse(features):
        feature_matrix = sparse.csr_matrix(features, dtype=np.float64, copy=True)
        feature_matrix.sum_duplicates()
        finite_values = feature_matrix.data
    else:
        feature_matrix = np.asarray(features, dtype=np.float64)
        finite_values = feature_matrix
    if (
        feature_matrix.ndim != 2
        or feature_matrix.shape[1] == 0
        or not np.isfinite(finite_values).all()
    ):
        raise ValueError("Features must be a finite matrix with at least one column")
    return feature_matrix


def _validated_names(
    names: Sequence[str] | None, size: int, prefix: str
) -> tuple[str, ...]:
    if names is None:
        return tuple(f"{prefix}_{index}" for index in range(size))
    names = tuple(names)
    if (
        len(names) != size
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != size
    ):
        raise ValueError(
            f"{prefix} names must be unique nonempty strings of length {size}"
        )
    return names


def prepare_evidence(
    targets: Any,
    observed_mask: Any | None = None,
    evidence_weights: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate included binary targets; never turn missing entries into negatives.

    A mask is mandatory when any target is missing. Weights outside the mask are
    ignored, as are target placeholders outside the mask. A zero weight excludes
    an entry even when its observation mask is true.
    """
    target_matrix = np.asarray(targets, dtype=np.float64)
    if target_matrix.ndim != 2 or target_matrix.shape[1] == 0:
        raise ValueError("Targets must be a matrix with at least one symptom column")
    if observed_mask is None:
        observation_mask = np.ones(target_matrix.shape, dtype=bool)
    else:
        observation_mask = np.asarray(observed_mask)
        if observation_mask.dtype != np.bool_:
            raise ValueError("observed_mask must have boolean dtype")
    if observation_mask.shape != target_matrix.shape:
        raise ValueError("Observation mask and targets must have identical shapes")
    if evidence_weights is None:
        weight_matrix = np.ones(target_matrix.shape, dtype=np.float64)
    else:
        weight_matrix = np.asarray(evidence_weights, dtype=np.float64)
    if weight_matrix.shape != target_matrix.shape:
        raise ValueError("Evidence weights and targets must have identical shapes")
    if (
        not np.isfinite(weight_matrix[observation_mask]).all()
        or (weight_matrix[observation_mask] < 0).any()
    ):
        raise ValueError("Included evidence weights must be finite and nonnegative")
    effective_weights = np.where(observation_mask, weight_matrix, 0.0)
    included_mask = effective_weights > 0
    included_targets = target_matrix[included_mask]
    if not np.isin(included_targets, (0.0, 1.0)).all():
        raise ValueError("Included targets must be binary; missing targets need a mask")
    cleaned_targets = np.where(included_mask, target_matrix, 0.0)
    return cleaned_targets, included_mask, effective_weights


def _nuclear_threshold(matrix: np.ndarray, threshold: float) -> np.ndarray:
    if threshold == 0:
        return matrix.copy()
    left_vectors, singular_values, right_vectors = np.linalg.svd(
        matrix, full_matrices=False
    )
    retained_values = np.maximum(singular_values - threshold, 0.0)
    return (left_vectors * retained_values) @ right_vectors


def _row_threshold(matrix: np.ndarray, threshold: float) -> np.ndarray:
    row_norms = np.linalg.norm(matrix, axis=1)
    multipliers = np.zeros_like(row_norms)
    retained_rows = row_norms > threshold
    multipliers[retained_rows] = 1.0 - threshold / row_norms[retained_rows]
    return matrix * multipliers[:, None]


def sparse_low_rank_proximal(
    matrix: np.ndarray,
    row_threshold: float,
    nuclear_threshold: float,
    *,
    tolerance: float = 1e-9,
    max_iterations: int = 300,
) -> np.ndarray:
    """Prox of row-group plus nuclear penalties, solved with Dykstra corrections."""
    if row_threshold == 0:
        return _nuclear_threshold(matrix, nuclear_threshold)
    if nuclear_threshold == 0:
        return _row_threshold(matrix, row_threshold)
    current_matrix = matrix.copy()
    nuclear_correction = np.zeros_like(matrix)
    row_correction = np.zeros_like(matrix)
    for _ in range(max_iterations):
        nuclear_input = current_matrix + nuclear_correction
        low_rank_matrix = _nuclear_threshold(nuclear_input, nuclear_threshold)
        nuclear_correction = nuclear_input - low_rank_matrix
        row_input = low_rank_matrix + row_correction
        updated_matrix = _row_threshold(row_input, row_threshold)
        row_correction = row_input - updated_matrix
        difference = np.linalg.norm(updated_matrix - current_matrix)
        current_matrix = updated_matrix
        if difference <= tolerance * max(1.0, np.linalg.norm(current_matrix)):
            return current_matrix
    raise RuntimeError(
        "Joint proximal operator did not converge; increase proximal_max_iter"
    )


class SparseReducedRankLogisticRegression:
    """Inductive multi-symptom predictor with sparse rows and a low-rank coefficient matrix.

    Rank is selected by the nuclear penalty, not a fixed latent dimension. This is
    a convex estimator, not a reproduction of the SCAD estimator in Park et al.
    (2024). Intercepts are unpenalized. Standardization uses fitting rows only.
    """

    def __init__(
        self,
        *,
        row_sparsity: float = 0.01,
        nuclear_penalty: float = 0.01,
        ridge_penalty: float = 1e-4,
        standardize: bool = True,
        max_iter: int = 500,
        tolerance: float = 1e-6,
        initial_step_size: float = 1.0,
        proximal_tolerance: float = 1e-9,
        proximal_max_iter: int = 300,
    ) -> None:
        for name, value in (
            ("row_sparsity", row_sparsity),
            ("nuclear_penalty", nuclear_penalty),
            ("ridge_penalty", ridge_penalty),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (
            ("tolerance", tolerance),
            ("initial_step_size", initial_step_size),
            ("proximal_tolerance", proximal_tolerance),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("max_iter", max_iter),
            ("proximal_max_iter", proximal_max_iter),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.configuration = {
            "row_sparsity": float(row_sparsity),
            "nuclear_penalty": float(nuclear_penalty),
            "ridge_penalty": float(ridge_penalty),
            "standardize": bool(standardize),
            "max_iter": max_iter,
            "tolerance": float(tolerance),
            "initial_step_size": float(initial_step_size),
            "proximal_tolerance": float(proximal_tolerance),
            "proximal_max_iter": proximal_max_iter,
        }

    def _logits(
        self,
        features: np.ndarray | sparse.csr_matrix,
        coefficients: np.ndarray,
        intercepts: np.ndarray,
    ) -> np.ndarray:
        original_scale_coefficients = coefficients / self.feature_scale_[:, None]
        return np.asarray(features @ original_scale_coefficients) + (
            intercepts - self.feature_mean_ @ original_scale_coefficients
        )

    def _smooth_loss_and_gradient(
        self,
        features: np.ndarray | sparse.csr_matrix,
        targets: np.ndarray,
        normalized_weights: np.ndarray,
        coefficients: np.ndarray,
        intercepts: np.ndarray,
        *,
        gradient: bool = True,
    ) -> tuple[float, np.ndarray | None, np.ndarray | None]:
        logits = self._logits(features, coefficients, intercepts)
        ridge_penalty = self.configuration["ridge_penalty"]
        smooth_loss = float(
            np.sum(normalized_weights * (np.logaddexp(0.0, logits) - targets * logits))
            + 0.5 * ridge_penalty * np.sum(coefficients**2)
        )
        if not gradient:
            return smooth_loss, None, None
        residuals = normalized_weights * (expit(logits) - targets)
        intercept_gradient = residuals.sum(axis=0)
        coefficient_gradient = (
            np.asarray(features.T @ residuals)
            - self.feature_mean_[:, None] * intercept_gradient
        ) / self.feature_scale_[:, None]
        coefficient_gradient += ridge_penalty * coefficients
        return smooth_loss, coefficient_gradient, intercept_gradient

    def _penalty(self, coefficients: np.ndarray) -> float:
        return float(
            self.configuration["row_sparsity"]
            * np.linalg.norm(coefficients, axis=1).sum()
            + self.configuration["nuclear_penalty"]
            * np.linalg.svd(coefficients, compute_uv=False).sum()
        )

    def fit(
        self,
        features: Any,
        targets: Any,
        *,
        observed_mask: Any | None = None,
        evidence_weights: Any | None = None,
        feature_names: Sequence[str] | None = None,
        symptom_names: Sequence[str] | None = None,
    ) -> "SparseReducedRankLogisticRegression":
        feature_matrix = _validated_features(features)
        target_matrix, _, weight_matrix = prepare_evidence(
            targets, observed_mask, evidence_weights
        )
        if feature_matrix.shape[0] != target_matrix.shape[0]:
            raise ValueError("Features and targets must have identical row counts")
        total_weight = float(weight_matrix.sum())
        if not math.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("Fitting requires positive, finite total evidence weight")
        if (weight_matrix.sum(axis=0) == 0).any():
            raise ValueError(
                "Every fitted symptom needs positive-weight training evidence"
            )
        # Entirely unlabelled rows cannot influence fitted preprocessing.
        fitting_rows = weight_matrix.sum(axis=1) > 0
        feature_matrix = feature_matrix[fitting_rows]
        target_matrix = target_matrix[fitting_rows]
        weight_matrix = weight_matrix[fitting_rows]
        self.feature_names_ = _validated_names(
            feature_names, feature_matrix.shape[1], "feature"
        )
        self.symptom_names_ = _validated_names(
            symptom_names, target_matrix.shape[1], "symptom"
        )
        self.feature_mean_ = np.zeros(feature_matrix.shape[1])
        self.feature_scale_ = np.ones(feature_matrix.shape[1])
        if self.configuration["standardize"]:
            if sparse.issparse(feature_matrix):
                self.feature_mean_, feature_variance = mean_variance_axis(
                    feature_matrix, axis=0
                )
            else:
                self.feature_mean_ = feature_matrix.mean(axis=0)
                feature_variance = feature_matrix.var(axis=0)
            self.feature_scale_ = np.sqrt(np.maximum(feature_variance, 0.0))
            self.feature_scale_[self.feature_scale_ < 1e-12] = 1.0
        if (
            not np.isfinite(self.feature_mean_).all()
            or not np.isfinite(self.feature_scale_).all()
        ):
            raise ValueError("Feature standardization overflowed")
        coefficients = np.zeros((feature_matrix.shape[1], target_matrix.shape[1]))
        base_rates = np.clip(
            (weight_matrix * target_matrix).sum(axis=0) / weight_matrix.sum(axis=0),
            1e-6,
            1.0 - 1e-6,
        )
        intercepts = np.log(base_rates) - np.log1p(-base_rates)
        normalized_weights = weight_matrix / total_weight
        step_size = self.configuration["initial_step_size"]
        self.objective_history_ = []
        self.converged_ = False
        for iteration in range(self.configuration["max_iter"]):
            (
                smooth_loss,
                coefficient_gradient,
                intercept_gradient,
            ) = self._smooth_loss_and_gradient(
                feature_matrix,
                target_matrix,
                normalized_weights,
                coefficients,
                intercepts,
            )
            objective = smooth_loss + self._penalty(coefficients)
            if not self.objective_history_:
                self.objective_history_.append(objective)
            # A majorization check chooses a stable step; an objective check also
            # protects against error in the numerically solved proximal operator.
            for _ in range(50):
                updated_coefficients = sparse_low_rank_proximal(
                    coefficients - step_size * coefficient_gradient,
                    step_size * self.configuration["row_sparsity"],
                    step_size * self.configuration["nuclear_penalty"],
                    tolerance=self.configuration["proximal_tolerance"],
                    max_iterations=self.configuration["proximal_max_iter"],
                )
                updated_intercepts = intercepts - step_size * intercept_gradient
                updated_smooth_loss, _, _ = self._smooth_loss_and_gradient(
                    feature_matrix,
                    target_matrix,
                    normalized_weights,
                    updated_coefficients,
                    updated_intercepts,
                    gradient=False,
                )
                coefficient_change = updated_coefficients - coefficients
                intercept_change = updated_intercepts - intercepts
                squared_change = float(
                    np.sum(coefficient_change**2) + np.sum(intercept_change**2)
                )
                majorizer = (
                    smooth_loss
                    + np.sum(coefficient_gradient * coefficient_change)
                    + np.sum(intercept_gradient * intercept_change)
                    + squared_change / (2.0 * step_size)
                )
                updated_objective = updated_smooth_loss + self._penalty(
                    updated_coefficients
                )
                if (
                    math.isfinite(updated_objective)
                    and updated_smooth_loss <= majorizer + 1e-12
                    and updated_objective <= objective + 1e-12
                ):
                    break
                step_size *= 0.5
            else:
                raise RuntimeError(
                    "Logistic optimizer could not find a descending step"
                )
            coefficients, intercepts = updated_coefficients, updated_intercepts
            self.objective_history_.append(updated_objective)
            self.n_iter_ = iteration + 1
            parameter_scale = max(
                1.0, math.sqrt(np.sum(coefficients**2) + np.sum(intercepts**2))
            )
            # The proximal-gradient mapping includes the step size; a tiny
            # backtracking step must not be mistaken for convergence.
            gradient_mapping_norm = math.sqrt(squared_change) / step_size
            if (
                gradient_mapping_norm
                <= self.configuration["tolerance"] * parameter_scale
            ):
                self.converged_ = True
                break
            step_size = min(step_size * 1.2, self.configuration["initial_step_size"])
        self.coef_ = coefficients
        self.intercept_ = intercepts
        self.training_row_count_ = int(fitting_rows.sum())
        return self

    def decision_function(
        self, features: Any, *, feature_names: Sequence[str] | None = None
    ) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("Fit or load the model before prediction")
        feature_matrix = _validated_features(features)
        if feature_matrix.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                "Prediction features do not match the fitted feature count"
            )
        if feature_names is not None and tuple(feature_names) != self.feature_names_:
            raise ValueError(
                "Prediction feature names/order do not match the fitted vocabulary"
            )
        return self._logits(feature_matrix, self.coef_, self.intercept_)

    def predict_proba(
        self, features: Any, *, feature_names: Sequence[str] | None = None
    ) -> np.ndarray:
        return expit(self.decision_function(features, feature_names=feature_names))

    def evaluation_report(
        self,
        features: Any,
        targets: Any,
        *,
        observed_mask: Any | None = None,
        evidence_weights: Any | None = None,
    ) -> dict[str, float | int | None]:
        target_matrix, included_mask, weight_matrix = prepare_evidence(
            targets, observed_mask, evidence_weights
        )
        logits = self.decision_function(features)
        if logits.shape != target_matrix.shape:
            raise ValueError("Evaluation targets do not match prediction shape")
        log_losses = np.logaddexp(0.0, logits) - target_matrix * logits
        included_count = int(included_mask.sum())
        total_weight = float(weight_matrix.sum())
        nll_nats = float(log_losses[included_mask].sum())
        return {
            "included_pair_count": included_count,
            "total_evidence_weight": total_weight,
            "unweighted_nll_nats": nll_nats,
            "conditional_data_cost_bits": nll_nats / math.log(2.0),
            "weighted_mean_log_loss": (
                float(np.sum(weight_matrix * log_losses) / total_weight)
                if total_weight > 0
                else None
            ),
        }

    def model_statistics(self) -> dict[str, Any]:
        if not hasattr(self, "coef_"):
            raise RuntimeError("Fit or load the model before inspecting coefficients")
        singular_values = np.linalg.svd(self.coef_, compute_uv=False)
        rank_tolerance = max(1e-8, float(singular_values.max(initial=0.0)) * 1e-8)
        active_rows = np.any(self.coef_ != 0, axis=1)
        return {
            "coefficient_rank": int(np.sum(singular_values > rank_tolerance)),
            "rank_tolerance": rank_tolerance,
            "singular_values": singular_values.tolist(),
            "active_feature_count": int(active_rows.sum()),
            "active_feature_names": [
                name for name, active in zip(self.feature_names_, active_rows) if active
            ],
            "stored_scalar_parameter_count": int(
                self.coef_.size
                + self.intercept_.size
                + self.feature_mean_.size
                + self.feature_scale_.size
            ),
            "converged": self.converged_,
            "iterations": self.n_iter_,
            "training_row_count": self.training_row_count_,
        }

    def save(self, output_path: str | Path) -> None:
        self.model_statistics()  # Require a fitted model before writing anything.
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": 1,
            "configuration": self.configuration,
            "converged": self.converged_,
            "iterations": self.n_iter_,
            "training_row_count": self.training_row_count_,
        }
        with NamedTemporaryFile(
            dir=output_path.parent, suffix=".npz", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        try:
            np.savez_compressed(
                temporary_path,
                coefficients=self.coef_,
                intercepts=self.intercept_,
                feature_mean=self.feature_mean_,
                feature_scale=self.feature_scale_,
                feature_names=np.asarray(self.feature_names_),
                symptom_names=np.asarray(self.symptom_names_),
                objective_history=np.asarray(self.objective_history_),
                metadata_json=np.asarray(json.dumps(metadata)),
            )
            os.replace(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, input_path: str | Path) -> "SparseReducedRankLogisticRegression":
        with np.load(input_path, allow_pickle=False) as model_archive:
            metadata = json.loads(str(model_archive["metadata_json"].item()))
            if metadata["format_version"] != 1:
                raise ValueError("Unsupported sparse logistic model format")
            model = cls(**metadata["configuration"])
            model.coef_ = _validated_features(model_archive["coefficients"])
            model.feature_names_ = _validated_names(
                model_archive["feature_names"].tolist(), model.coef_.shape[0], "feature"
            )
            model.symptom_names_ = _validated_names(
                model_archive["symptom_names"].tolist(), model.coef_.shape[1], "symptom"
            )
            for name, archive_name, size in (
                ("intercept_", "intercepts", model.coef_.shape[1]),
                ("feature_mean_", "feature_mean", model.coef_.shape[0]),
                ("feature_scale_", "feature_scale", model.coef_.shape[0]),
            ):
                values = np.asarray(model_archive[archive_name], dtype=np.float64)
                if values.shape != (size,) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid saved {archive_name}")
                setattr(model, name, values)
            if (model.feature_scale_ <= 0).any():
                raise ValueError("Saved feature scales must be positive")
            model.objective_history_ = model_archive["objective_history"].tolist()
            model.converged_ = bool(metadata["converged"])
            model.n_iter_ = int(metadata["iterations"])
            model.training_row_count_ = int(metadata["training_row_count"])
        return model
