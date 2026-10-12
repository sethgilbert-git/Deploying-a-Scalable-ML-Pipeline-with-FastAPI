"""Model training, persistence, inference and held-out slice evaluation."""
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from ml.data import process_data


def train_model(X_train, y_train):
    """Fit a reproducible random forest on encoded training records."""
    model = RandomForestClassifier(
        n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=2,
    )
    return model.fit(X_train, y_train)


def compute_model_metrics(y, preds):
    """Return positive-class precision, recall and F1; undefined scores are 0."""
    return (precision_score(y, preds, zero_division=0),
            recall_score(y, preds, zero_division=0),
            f1_score(y, preds, zero_division=0))


def inference(model, X):
    """Return binary predictions for encoded records."""
    return model.predict(X)


def save_model(model, path):
    """Save a model or encoder to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(model, stream)


def load_model(path):
    """Load a trusted local artifact; never load untrusted pickle files."""
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder,
    lb, model,
):
    """Evaluate one nonempty held-out category using training-fitted encoders."""
    subset = data.loc[data[column_name] == slice_value]
    if subset.empty:
        raise ValueError("The requested slice is empty.")
    X, y, _, _ = process_data(
        subset, categorical_features, label, False, encoder, lb,
    )
    return compute_model_metrics(y, inference(model, X))
