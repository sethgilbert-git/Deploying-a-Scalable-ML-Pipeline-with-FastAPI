"""Behavioral tests for preprocessing, model artifacts and slice metrics."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.data import CATEGORICAL_FEATURES, process_data
from ml.model import (compute_model_metrics, inference, load_model,
                      performance_on_categorical_slice, save_model, train_model)


@pytest.fixture(scope="module")
def trained():
    data = pd.read_csv("data/census.csv").iloc[:500]
    X, y, encoder, lb = process_data(data, CATEGORICAL_FEATURES, "salary")
    return data, X, y, encoder, lb, train_model(X, y)


def test_training_returns_fitted_classifier(trained):
    _, X, _, _, _, model = trained
    assert isinstance(model, RandomForestClassifier)
    assert model.n_features_in_ == X.shape[1]
    assert set(inference(model, X)) <= {0, 1}


def test_metrics_known_confusion_matrix():
    scores = compute_model_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert scores == pytest.approx((0.5, 0.5, 0.5))


def test_undefined_metrics_are_zero():
    assert compute_model_metrics([0, 0], [0, 0]) == (0, 0, 0)


def test_saved_artifacts_preserve_predictions(trained, tmp_path):
    _, X, _, encoder, lb, model = trained
    for name, artifact in [("model", model), ("encoder", encoder), ("lb", lb)]:
        save_model(artifact, tmp_path / f"{name}.pkl")
    restored = load_model(tmp_path / "model.pkl")
    np.testing.assert_array_equal(inference(model, X), inference(restored, X))
    assert load_model(tmp_path / "encoder.pkl").categories_[0].tolist() == (
        encoder.categories_[0].tolist())
    assert load_model(tmp_path / "lb.pkl").classes_.tolist() == lb.classes_.tolist()


def test_unseen_category_does_not_refit_encoder(trained):
    data, X, _, encoder, lb, _ = trained
    sample = data.iloc[:1].drop(columns="salary").copy()
    sample["workclass"] = "Previously-unseen"
    before = [category.copy() for category in encoder.categories_]
    result, y, _, _ = process_data(
        sample, CATEGORICAL_FEATURES, training=False, encoder=encoder, lb=lb,
    )
    assert result.shape == (1, X.shape[1])
    assert y.size == 0
    for old, new in zip(before, encoder.categories_):
        np.testing.assert_array_equal(old, new)


def test_slice_metrics_match_direct_evaluation(trained):
    data, _, _, encoder, lb, model = trained
    subset = data.loc[data["sex"] == "Female"]
    X, y, _, _ = process_data(
        subset, CATEGORICAL_FEATURES, "salary", False, encoder, lb,
    )
    actual = performance_on_categorical_slice(
        data, "sex", "Female", CATEGORICAL_FEATURES, "salary", encoder, lb, model,
    )
    assert actual == pytest.approx(compute_model_metrics(y, inference(model, X)))


def test_empty_slice_raises(trained):
    data, _, _, encoder, lb, model = trained
    with pytest.raises(ValueError, match="empty"):
        performance_on_categorical_slice(
            data, "sex", "missing", CATEGORICAL_FEATURES, "salary",
            encoder, lb, model,
        )
