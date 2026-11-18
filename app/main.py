from typing import Literal 
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
from catboost import CatBoostRegressor

import numpy as np
import pandas as pd

from src.features import prepare_features

app = FastAPI(title="Calories Prediction API")
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "catboost_model.cbm"

model = CatBoostRegressor()
model.load_model(MODEL_PATH)


class CaloriesInput(BaseModel):
    Age: int = Field(ge=18, le=81)
    Height: float = Field(gt=124, lt=228)
    Weight: float = Field(gt=34, lt=134)
    Duration: float = Field(gt=0.9, lt=32.0)
    Heart_Rate: float = Field(gt=65.0, lt=130)
    Body_Temp: float = Field(gt=36.0, lt=43.5)
    Sex: Literal["male", "female"]


@app.get('/health')
def health():
    return {'status' : 'ok'}

@app.post('/predict')
def predict(data: CaloriesInput):
    data = data.model_dump()
    df = pd.DataFrame([data])
    df = prepare_features(df)
    pred = float(np.expm1(model.predict(df)[0]))
    return {'calories' : pred}

