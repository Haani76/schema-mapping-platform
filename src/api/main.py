import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
from src.inference.predictor import SchemaPredictor
from configs.config import config

# Initialize app
app = FastAPI(
    title="Schema Mapping Platform",
    description="BERT-based semantic column inference API with confidence scoring",
    version="1.0.0",
)

# Load predictor once at startup
predictor = None


@app.on_event("startup")
async def load_model():
    global predictor
    print("Loading model...")
    predictor = SchemaPredictor()
    print("Model ready.")


# --- Request/Response Models ---

class ColumnRequest(BaseModel):
    column_name: str
    sample_value: Optional[str] = ""


class BatchRequest(BaseModel):
    columns: List[ColumnRequest]


class PredictionResponse(BaseModel):
    column_name: str
    sample_value: str
    predicted_label: str
    confidence: float
    routing: str
    top3_predictions: list
    threshold_used: float
    latency_ms: float


class BatchResponse(BaseModel):
    total_columns: int
    auto_mapped_count: int
    needs_review_count: int
    auto_map_rate: float
    auto_mapped: list
    needs_review: list
    total_latency_ms: float


# --- Endpoints ---

@app.get("/")
def root():
    return {
        "service": "Schema Mapping Platform",
        "version": "1.0.0",
        "status": "running",
        "model": config.MODEL_NAME,
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_single(request: ColumnRequest):
    """Predict semantic type for a single column."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    result = predictor.predict_column(request.column_name, request.sample_value)
    latency_ms = round((time.time() - start) * 1000, 2)

    return {**result, "latency_ms": latency_ms}


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    """Predict semantic types for a batch of columns."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.columns) > 100:
        raise HTTPException(status_code=400, detail="Max 100 columns per batch")

    start = time.time()
    columns = [{"column_name": c.column_name, "sample_value": c.sample_value}
               for c in request.columns]
    result = predictor.predict_dataframe_columns(columns)
    latency_ms = round((time.time() - start) * 1000, 2)

    return {**result, "total_latency_ms": latency_ms}


@app.get("/labels")
def get_labels():
    """Return all supported semantic label types."""
    return {
        "labels": [l.replace("B-", "") for l in config.LABELS if l != "O"],
        "total": config.NUM_LABELS - 1,
    }