"""FastAPI entry point for the ML service."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from config import config  # your Config class

app = FastAPI(title="Portfolio Optimizer ML Service")


# ---------------------------
# Request Schemas (Validation)
# ---------------------------

class PredictRequest(BaseModel):
    features: List[float]


class TrainRequest(BaseModel):
    epochs: Optional[int] = config.EPOCHS
    batch_size: Optional[int] = config.BATCH_SIZE
    learning_rate: Optional[float] = config.LEARNING_RATE


# ---------------------------
# Routes
# ---------------------------

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "ML Service is running",
        "port": config.PORT
    }


@app.post("/predict")
async def predict(request: PredictRequest):
    """Prediction endpoint."""
    try:
        # 👉 Replace with actual model inference
        prediction = sum(request.features)  # dummy logic

        return {
            "prediction": prediction,
            "model_used": "dummy-model-v1"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train(request: TrainRequest):
    """Training endpoint."""
    try:
        # 👉 Replace with actual training logic
        return {
            "status": "Training started",
            "config": {
                "epochs": request.epochs,
                "batch_size": request.batch_size,
                "learning_rate": request.learning_rate
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# Run Server
# ---------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        reload=True  # auto-reload for dev
    )
