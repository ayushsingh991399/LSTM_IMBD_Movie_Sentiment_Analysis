# predict.py
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


MODEL_PATH = "model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 200


model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)


app = FastAPI(
    title="IMDB Sentiment Analysis API",
    description="Predict sentiment (Positive/Negative) of IMDB movie reviews using LSTM",
    version="1.0"
)


class ReviewRequest(BaseModel):
    review: str

class PredictionResponse(BaseModel):
    sentiment: str
    probability: float


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(data: ReviewRequest):
    
    sequence = tokenizer.texts_to_sequences([data.review])

    # Padding
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)

    # Prediction
    probability = float(model.predict(padded_sequence)[0][0])

    sentiment = "Positive" if probability >= 0.5 else "Negative"

    return {
        "sentiment": sentiment,
        "probability": round(probability, 4)
    }

@app.get("/")
def health_check():
    return {"status": "API is running"}
