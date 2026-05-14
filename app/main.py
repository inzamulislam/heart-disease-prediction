from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# model loading
model = joblib.load("model/heart_model.joblib")

# input data format (Pydantic Model)
class HeartData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/info")
def info():
    return {
        "model_type": "Random Forest Classifier",
        "features": list(HeartData.model_fields.keys())
    }

@app.post("/predict")
def predict(data: HeartData):
    # create dataframe from disctionary
    input_df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_df)[0]
    return {"heart_disease": bool(prediction)}