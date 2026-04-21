"""
FastAPI inference server for the Self-Pruning Neural Network.

Endpoints:
    POST /predict         — Classify a CIFAR-10 image
    GET  /sparsity-report — View per-layer pruning statistics
    GET  /health          — Health check
    GET  /docs            — Auto-generated OpenAPI documentation

The model is loaded once at startup and cached in memory.
"""

import os
import sys
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.prunable_net import PrunableNet
from src.api.schemas import (
    CIFAR10_CLASSES,
    HealthResponse,
    ImageRequest,
    PredictionResponse,
    SparsityReportResponse,
)
from src.api.cache import PredictionCache

# --- App setup ---
app = FastAPI(
    title="Self-Pruning Net Inference API",
    description=(
        "Inference API for a self-pruning neural network trained on CIFAR-10. "
        "The model uses differentiable gate parameters to learn which weights "
        "to prune during training, achieving significant sparsity with minimal "
        "accuracy loss."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Globals ---
_cache = PredictionCache()

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(PROJECT_ROOT / "outputs" / "best_model.pth"),
)


@lru_cache(maxsize=1)
def load_model() -> PrunableNet:
    """Load the trained model (cached singleton).

    Returns:
        PrunableNet in eval mode on CPU.
    """
    model = PrunableNet()
    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. "
            f"Run training first: python experiments/run_all.py"
        )

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Model loaded from {model_path}")
    return model


# --- Endpoints ---


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: ImageRequest):
    """Classify a CIFAR-10 image.

    Accepts a 3×32×32 normalized image and returns the predicted class,
    confidence score, and optionally the active gate fraction.
    """
    # Check cache
    cached = _cache.get(req.pixels)
    if cached:
        return PredictionResponse(**cached)

    try:
        model = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Validate input shape
    try:
        x = torch.tensor(req.pixels).unsqueeze(0).float()  # (1, 3, 32, 32)
        if x.shape != (1, 3, 32, 32):
            raise ValueError(f"Expected shape (1, 3, 32, 32), got {x.shape}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid input: {e}")

    # Inference
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(1).item()
        conf = probs[0, pred].item()

    response_data = {
        "predicted_class": pred,
        "class_name": CIFAR10_CLASSES[pred],
        "confidence": round(conf, 4),
        "active_gate_fraction": None,
    }

    if req.return_gates:
        sparsity = model.get_sparsity_report().get("overall", 0.0)
        response_data["active_gate_fraction"] = round(1 - sparsity, 4)

    # Cache result
    _cache.set(req.pixels, response_data)

    return PredictionResponse(**response_data)


@app.get("/sparsity-report", response_model=SparsityReportResponse)
async def sparsity_report():
    """Get the current sparsity report for all prunable layers."""
    try:
        model = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    report = model.get_sparsity_report()
    total = sum(p.numel() for p in model.parameters())
    active = model.get_active_params()

    return SparsityReportResponse(
        report=report,
        layer_count=sum(1 for _ in model.modules() if hasattr(_, "gate_scores")),
        total_params=total,
        active_params=active,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    model_loaded = False
    try:
        load_model()
        model_loaded = True
    except Exception:
        pass

    return HealthResponse(status="ok", model_loaded=model_loaded)
