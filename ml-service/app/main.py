"""FastAPI entry point for the ML service."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="Portfolio Optimizer ML Service")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "ML Service is running"}


@app.post("/predict")
async def predict(data: dict):
    """Prediction endpoint."""
    return {"prediction": None}


@app.post("/train")
async def train(config: dict):
    """Training endpoint."""
    return {"status": "Training started"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
