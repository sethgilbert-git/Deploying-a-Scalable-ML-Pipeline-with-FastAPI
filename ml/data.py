"""Training-only encoding and shared Census feature schema."""
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

CATEGORICAL_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
FEATURES = [
    "age", "workclass", "fnlgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
]


def process_data(X, categorical_features=None, label=None, training=True,
                 encoder=None, lb=None):
    """Encode categories without fitting encoders on evaluation data."""
    categorical_features = categorical_features or []
    X = X.copy()
    X.columns = X.columns.str.strip()
    for column in X.select_dtypes(include="object"):
        X[column] = X[column].str.strip()
    y = X.pop(label) if label else None
    continuous = X.drop(columns=categorical_features).to_numpy(dtype=float)
    categorical = X[categorical_features]
    if training:
        if y is None:
            raise ValueError("Training requires labels.")
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        encoded = encoder.fit_transform(categorical)
        labels = lb.fit_transform(y).ravel()
    else:
        if encoder is None or lb is None:
            raise ValueError("Inference requires fitted encoders.")
        encoded = encoder.transform(categorical)
        labels = lb.transform(y).ravel() if y is not None else np.array([])
    return np.concatenate([continuous, encoded], axis=1), labels, encoder, lb


def apply_label(predictions):
    """Decode one binary prediction."""
    return ">50K" if predictions[0] == 1 else "<=50K"
