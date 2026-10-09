"""Fit the sparse logistic baseline on a frozen, explicitly split feature artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score

from sparse_rr_logistic import (
    SparseReducedRankLogisticRegression,
    _validated_features,
    _validated_names,
    prepare_evidence,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_baseline_dataset(input_path: Path) -> dict[str, Any]:
    """Load dense or CSR features without pickle; require a fixed observation policy."""
    with np.load(input_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if not isinstance(metadata, dict) or metadata.get("format_version") != 1:
            raise ValueError("Dataset metadata must specify format_version=1")
        graph_snapshot = metadata.get("graph_snapshot_sha256", "")
        if (
            not isinstance(graph_snapshot, str)
            or len(graph_snapshot) != 64
            or any(character not in "0123456789abcdef" for character in graph_snapshot)
        ):
            raise ValueError("Dataset needs a lowercase graph_snapshot_sha256")
        generator = metadata.get("feature_generator")
        if (
            not isinstance(generator, dict)
            or not isinstance(generator.get("name"), str)
            or not generator["name"].strip()
            or not isinstance(generator.get("configuration"), dict)
        ):
            raise ValueError("Record the feature generator name and configuration")
        feature_fit_ids = metadata.get("feature_fit_perturbation_ids")
        if (
            not isinstance(feature_fit_ids, list)
            or any(
                not isinstance(identifier, str) or not identifier.strip()
                for identifier in feature_fit_ids
            )
            or len(set(feature_fit_ids)) != len(feature_fit_ids)
        ):
            raise ValueError(
                "Record feature_fit_perturbation_ids (empty for a fixed operator)"
            )
        has_dense = "features" in archive
        csr_keys = {
            "feature_data",
            "feature_indices",
            "feature_indptr",
            "feature_shape",
        }
        has_csr = bool(csr_keys.intersection(archive.files))
        if has_dense == has_csr:
            raise ValueError("Supply exactly one dense or CSR feature representation")
        if has_dense:
            features = _validated_features(archive["features"])
        else:
            if not csr_keys.issubset(archive.files):
                raise ValueError("Incomplete CSR feature representation")
            for key in ("feature_indices", "feature_indptr", "feature_shape"):
                if not np.issubdtype(archive[key].dtype, np.integer):
                    raise ValueError(f"{key} must have integer dtype")
            feature_shape = archive["feature_shape"]
            if feature_shape.shape != (2,) or (feature_shape < 0).any():
                raise ValueError(
                    "feature_shape must contain two nonnegative dimensions"
                )
            features = sparse.csr_matrix(
                (
                    archive["feature_data"],
                    archive["feature_indices"],
                    archive["feature_indptr"],
                ),
                shape=tuple(feature_shape),
            )
            features.check_format(full_check=True)
            features = _validated_features(features)
        targets, observed_mask, weights = prepare_evidence(
            archive["targets"], archive["observed_mask"], archive["evidence_weights"]
        )
        if targets.shape[0] != features.shape[0]:
            raise ValueError("Targets and features have different row counts")
        perturbation_ids = _validated_names(
            archive["perturbation_ids"].tolist(), features.shape[0], "perturbation"
        )
        feature_names = _validated_names(
            archive["feature_names"].tolist(), features.shape[1], "feature"
        )
        symptom_names = _validated_names(
            archive["symptom_names"].tolist(), targets.shape[1], "symptom"
        )
        splits = archive["splits"].copy()
        group_ids = archive["group_ids"].tolist()
    if (
        splits.shape != (features.shape[0],)
        or not np.isin(splits, ("train", "validation", "test")).all()
    ):
        raise ValueError("Every row needs a train, validation or test split")
    if not np.any(splits == "train") or not np.any(splits == "validation"):
        raise ValueError("The artifact needs training and validation rows")
    if len(group_ids) != features.shape[0] or any(
        not isinstance(group, str) or not group.strip() for group in group_ids
    ):
        raise ValueError("Every perturbation needs an explicit nonempty leakage group")
    group_splits: dict[str, str] = {}
    for group, split in zip(group_ids, splits):
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"Leakage group {group!r} crosses predefined splits")
        group_splits[group] = str(split)
    training_ids = {
        identifier
        for identifier, split in zip(perturbation_ids, splits)
        if split == "train"
    }
    if not set(feature_fit_ids).issubset(training_ids):
        raise ValueError("Feature generator was fitted outside the training split")
    return {
        "features": features,
        "targets": targets,
        "observed_mask": observed_mask,
        "evidence_weights": weights,
        "perturbation_ids": perturbation_ids,
        "feature_names": feature_names,
        "symptom_names": symptom_names,
        "splits": splits,
        "group_ids": group_ids,
        "metadata": metadata,
    }


def _split_report(
    model: SparseReducedRankLogisticRegression,
    dataset: dict[str, Any],
    split: str,
    minimum_positives: int,
) -> dict[str, Any]:
    selected_rows = dataset["splits"] == split
    features = dataset["features"][selected_rows]
    targets = dataset["targets"][selected_rows]
    mask = dataset["observed_mask"][selected_rows]
    probabilities = model.predict_proba(features)
    report = model.evaluation_report(
        features,
        targets,
        observed_mask=mask,
        evidence_weights=dataset["evidence_weights"][selected_rows],
    )
    per_symptom = {}
    for symptom_index, symptom_name in enumerate(dataset["symptom_names"]):
        symptom_mask = mask[:, symptom_index]
        symptom_targets = targets[symptom_mask, symptom_index]
        symptom_probabilities = probabilities[symptom_mask, symptom_index]
        positive_count = int(symptom_targets.sum())
        negative_count = int(len(symptom_targets) - positive_count)
        per_symptom[symptom_name] = {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "average_precision": (
                float(average_precision_score(symptom_targets, symptom_probabilities))
                if positive_count >= minimum_positives and negative_count
                else None
            ),
            "auroc": (
                float(roc_auc_score(symptom_targets, symptom_probabilities))
                if positive_count >= minimum_positives and negative_count
                else None
            ),
        }
    macro_values = [
        record["average_precision"]
        for record in per_symptom.values()
        if record["average_precision"] is not None
    ]
    flat_targets = targets[mask]
    report.update(
        macro_average_precision=float(np.mean(macro_values)) if macro_values else None,
        micro_average_precision=(
            float(average_precision_score(flat_targets, probabilities[mask]))
            if flat_targets.size
            and np.any(flat_targets == 1)
            and np.any(flat_targets == 0)
            else None
        ),
        per_symptom=per_symptom,
    )
    return report


def run_baseline(
    input_path: Path,
    output_directory: Path,
    *,
    model_configuration: dict[str, Any] | None = None,
    expected_graph_snapshot: str | None = None,
    evaluate_test: bool = False,
    minimum_positives: int = 5,
) -> dict[str, Any]:
    """Fit training rows only; test labels are not scored/exported unless requested."""
    input_path, output_directory = Path(input_path), Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(f"Output already exists: {output_directory}")
    if minimum_positives < 1:
        raise ValueError("minimum_positives must be positive")
    input_checksum = _sha256(input_path)
    dataset = load_baseline_dataset(input_path)
    metadata = dataset["metadata"]
    if (
        expected_graph_snapshot is not None
        and expected_graph_snapshot != metadata["graph_snapshot_sha256"]
    ):
        raise ValueError("Graph snapshot does not match the expected pin")
    model = SparseReducedRankLogisticRegression(**(model_configuration or {}))
    training_rows = dataset["splits"] == "train"
    model.fit(
        dataset["features"][training_rows],
        dataset["targets"][training_rows],
        observed_mask=dataset["observed_mask"][training_rows],
        evidence_weights=dataset["evidence_weights"][training_rows],
        feature_names=dataset["feature_names"],
        symptom_names=dataset["symptom_names"],
    )
    report = {
        "model": "sparse_reduced_rank_logistic",
        "configuration": model.configuration,
        "statistics": model.model_statistics(),
        "training": _split_report(model, dataset, "train", minimum_positives),
        "validation": _split_report(model, dataset, "validation", minimum_positives),
        "minimum_positives_to_score": minimum_positives,
        "test_evaluated": bool(evaluate_test and np.any(dataset["splits"] == "test")),
    }
    if report["test_evaluated"]:
        report["test"] = _split_report(model, dataset, "test", minimum_positives)
    probabilities = model.predict_proba(
        dataset["features"], feature_names=dataset["feature_names"]
    )
    predictions = []
    for row_index, (perturbation_id, split) in enumerate(
        zip(dataset["perturbation_ids"], dataset["splits"])
    ):
        if split == "train":
            continue
        for symptom_index, symptom_name in enumerate(dataset["symptom_names"]):
            expose_target = split != "test" or evaluate_test
            included = (
                bool(dataset["observed_mask"][row_index, symptom_index])
                and expose_target
            )
            predictions.append(
                {
                    "perturbation_id": perturbation_id,
                    "split": str(split),
                    "symptom": symptom_name,
                    "probability": float(probabilities[row_index, symptom_index]),
                    "included_in_scoring": included,
                    "target": float(dataset["targets"][row_index, symptom_index])
                    if included
                    else None,
                    "evidence_weight": float(
                        dataset["evidence_weights"][row_index, symptom_index]
                    )
                    if included
                    else None,
                }
            )
    if _sha256(input_path) != input_checksum:
        raise ValueError("Input artifact changed during fitting")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=output_directory.parent, prefix="sparse_rr_stage_"
    ) as stage:
        staged_directory = Path(stage) / "result"
        staged_directory.mkdir()
        model.save(staged_directory / "model.npz")
        pl.DataFrame(predictions).write_parquet(
            staged_directory / "predictions.parquet"
        )
        (staged_directory / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        manifest = {
            "format_version": 1,
            "input_sha256": input_checksum,
            "feature_provenance": metadata,
            "split_counts": {
                split: int(np.sum(dataset["splits"] == split))
                for split in ("train", "validation", "test")
            },
            "outputs": {
                path.name: _sha256(path) for path in sorted(staged_directory.iterdir())
            },
        }
        (staged_directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        staged_directory.rename(output_directory)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", type=Path, help="Frozen feature/evidence NPZ artifact"
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--row-sparsity", type=float, default=0.01)
    parser.add_argument("--nuclear-penalty", type=float, default=0.01)
    parser.add_argument("--ridge-penalty", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--expected-graph-snapshot", type=str)
    parser.add_argument("--minimum-positives", type=int, default=5)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Explicitly score/export test targets; leave off during model selection",
    )
    arguments = parser.parse_args(argv)
    report = run_baseline(
        arguments.dataset,
        arguments.output_directory,
        model_configuration={
            "row_sparsity": arguments.row_sparsity,
            "nuclear_penalty": arguments.nuclear_penalty,
            "ridge_penalty": arguments.ridge_penalty,
            "max_iter": arguments.max_iter,
            "tolerance": arguments.tolerance,
            "standardize": not arguments.no_standardize,
        },
        expected_graph_snapshot=arguments.expected_graph_snapshot,
        evaluate_test=arguments.evaluate_test,
        minimum_positives=arguments.minimum_positives,
    )
    print(
        json.dumps(
            {
                "output_directory": str(arguments.output_directory),
                "statistics": report["statistics"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
