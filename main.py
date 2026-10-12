"""FastAPI interface to the trained Census income classifier."""
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from ml.data import CATEGORICAL_FEATURES, FEATURES, process_data
from ml.model import inference, load_model

ROOT = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app):
    """Load all artifacts once at startup."""
    app.state.model = load_model(ROOT / "model/model.pkl")
    app.state.encoder = load_model(ROOT / "model/encoder.pkl")
    app.state.lb = load_model(ROOT / "model/lb.pkl")
    yield


app = FastAPI(title="Census Income API", lifespan=lifespan)


class CensusRecord(BaseModel):
    """One record; aliases preserve Census hyphenated column names."""
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    age: int = Field(ge=0, le=120)
    workclass: str
    fnlgt: int = Field(ge=0)
    education: str
    education_num: int = Field(alias="education-num", ge=1, le=16)
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain", ge=0)
    capital_loss: int = Field(alias="capital-loss", ge=0)
    hours_per_week: int = Field(alias="hours-per-week", ge=0, le=168)
    native_country: str = Field(alias="native-country")


@app.get("/")
def greeting():
    """Return the root welcome message."""
    return {"message": "Hello from the Census Income API!"}


@app.post("/predict")
def predict(record: CensusRecord):
    """Return the income label for one validated Census record."""
    frame = pd.DataFrame([record.model_dump(by_alias=True)])[FEATURES]
    X, _, _, _ = process_data(
        frame, CATEGORICAL_FEATURES, training=False,
        encoder=app.state.encoder, lb=app.state.lb,
    )
    prediction = inference(app.state.model, X)
    label = app.state.lb.inverse_transform(prediction)[0]
    return {"prediction": str(label)}
