
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

MODEL_PATH = "models/best_purchase_prediction_model.pkl"
PREPROCESSOR_PATH = "models/deployment_preprocessor.pkl"

# Load trained model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load preprocessing objects
with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

brand_encoder = preprocessor["brand_encoder"]
category_encoder = preprocessor["category_encoder"]
scaler = preprocessor["scaler"]
features = preprocessor["features"]


app = FastAPI(
    title="Customer Purchase Prediction API",
    description="API for predicting customer purchase intent.",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    view_count: int
    cart_count: int
    remove_count: int
    price: float
    brand: str
    category_code: str


@app.get("/")
def root():
    return {
        "message": "Customer Purchase Prediction API is running",
        "endpoint": "/predict"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    # Check whether brand exists in training data
    if request.brand not in brand_encoder.classes_:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown brand: {request.brand}"
        )

    # Check whether category exists in training data
    if request.category_code not in category_encoder.classes_:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category_code: {request.category_code}"
        )

    # Create input DataFrame
    input_df = pd.DataFrame([{
        "view_count": request.view_count,
        "cart_count": request.cart_count,
        "remove_count": request.remove_count,
        "price": request.price,
        "brand": request.brand,
        "category_code": request.category_code
    }])

    # Encode categorical features
    input_df["brand"] = brand_encoder.transform(
        input_df["brand"].astype(str)
    )

    input_df["category_code"] = category_encoder.transform(
        input_df["category_code"].astype(str)
    )

    # Apply the same scaler used during Experiment 4
    input_scaled = scaler.transform(input_df[features])

    # Make prediction
    prediction = int(model.predict(input_scaled)[0])

    # Get probability of purchase
    probability = float(
        model.predict_proba(input_scaled)[0][1]
    )

    if prediction == 1:
        result = "Likely to purchase"
    else:
        result = "Unlikely to purchase"

    return {
        "prediction": prediction,
        "purchase_probability": round(probability, 4),
        "result": result
    }
