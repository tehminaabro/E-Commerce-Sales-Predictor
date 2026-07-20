
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


app = FastAPI()


model = joblib.load('saved_model/data/1_e-commerse_clean_model.joblib')


class InputData(BaseModel):
    UnitPrice: float
    Quantity: int


@app.post("/predict")
def predict_sales(data: InputData):
    
    input_array = np.array([[data.UnitPrice, data.Quantity]])
    
    
    prediction = model.predict(input_array)
    
    
    return {"Predicted_Sales": float(prediction)}
