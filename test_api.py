"""HTTP contract and real-artifact inference integration tests."""
from fastapi.testclient import TestClient
import pytest

from local_api import EXAMPLE
from main import app


@pytest.fixture
def client():
    with TestClient(app) as instance:
        yield instance


def test_root_greeting(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Hello" in response.json()["message"]


@pytest.mark.parametrize("index", [0, 7])
def test_prediction_matches_trained_artifacts(client, index):
    import pandas as pd
    from ml.data import CATEGORICAL_FEATURES, FEATURES, process_data
    from ml.model import inference
    row = pd.read_csv("data/census.csv").iloc[index]
    payload = row.drop("salary").to_dict()
    response = client.post("/predict", json=payload)
    X, _, _, _ = process_data(
        pd.DataFrame([payload])[FEATURES], CATEGORICAL_FEATURES,
        training=False, encoder=app.state.encoder, lb=app.state.lb,
    )
    expected = app.state.lb.inverse_transform(inference(app.state.model, X))[0]
    assert response.status_code == 200
    assert response.json() == {"prediction": expected}


def test_invalid_record_rejected(client):
    response = client.post("/predict", json={**EXAMPLE, "age": -1})
    assert response.status_code == 422


def test_missing_features_rejected(client):
    assert client.post("/predict", json={"age": 39}).status_code == 422
