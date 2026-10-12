"""Train Census classifier and write reproducible held-out evaluations."""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import CATEGORICAL_FEATURES, process_data
from ml.model import (compute_model_metrics, inference,
                      performance_on_categorical_slice, save_model, train_model)

ROOT = Path(__file__).resolve().parent


def main():
    """Fit preprocessing on training data only and evaluate held-out slices."""
    data = pd.read_csv(ROOT / "data/census.csv", skipinitialspace=True)
    data.columns = data.columns.str.strip()
    for column in data.select_dtypes(include="object"):
        data[column] = data[column].str.strip()
    train, test = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["salary"],
    )
    X_train, y_train, encoder, lb = process_data(
        train, CATEGORICAL_FEATURES, "salary",
    )
    X_test, y_test, _, _ = process_data(
        test, CATEGORICAL_FEATURES, "salary", False, encoder, lb,
    )
    model = train_model(X_train, y_train)
    for name, artifact in [("model", model), ("encoder", encoder), ("lb", lb)]:
        save_model(artifact, ROOT / "model" / f"{name}.pkl")
    precision, recall, f1 = compute_model_metrics(
        y_test, inference(model, X_test),
    )
    metrics = dict(precision=float(precision), recall=float(recall),
                   f1=float(f1), train_count=len(train), test_count=len(test),
                   random_state=42)
    (ROOT / "model/metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    lines = ["Held-out slices; positive class >50K; undefined metrics = 0."]
    for feature in CATEGORICAL_FEATURES:
        for value in sorted(test[feature].unique()):
            scores = performance_on_categorical_slice(
                test, feature, value, CATEGORICAL_FEATURES, "salary",
                encoder, lb, model,
            )
            count = int((test[feature] == value).sum())
            positives = int(((test[feature] == value)
                             & (test["salary"] == ">50K")).sum())
            line = (f"{feature}: {value} | Count: {count} | "
                    f"Positives: {positives} | Precision: {scores[0]:.4f} | "
                    f"Recall: {scores[1]:.4f} | F1: {scores[2]:.4f}")
            lines.append(line)
            print(line)
    (ROOT / "slice_output.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
