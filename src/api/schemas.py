"""
Pydantic schemas for the FastAPI inference server.
"""

from pydantic import BaseModel, Field


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class ImageRequest(BaseModel):
    """Request body for /predict endpoint.

    Attributes:
        pixels: Normalized image as a 3D list [3][32][32] (CHW format).
            Values should be normalized (e.g., mean-subtracted and scaled).
        return_gates: If True, include gate activity fraction in response.
    """

    pixels: list[list[list[float]]] = Field(
        ...,
        description="3×32×32 normalized image in CHW format",
    )
    return_gates: bool = Field(
        default=False,
        description="Include active gate fraction in response",
    )


class PredictionResponse(BaseModel):
    """Response body for /predict endpoint."""

    predicted_class: int = Field(..., description="Predicted class index (0-9)")
    class_name: str = Field(..., description="Human-readable class name")
    confidence: float = Field(..., description="Softmax confidence for predicted class")
    active_gate_fraction: float | None = Field(
        default=None,
        description="Fraction of gates that are active (non-pruned)",
    )


class SparsityReportResponse(BaseModel):
    """Response body for /sparsity-report endpoint."""

    report: dict = Field(..., description="Per-layer and overall sparsity percentages")
    layer_count: int = Field(..., description="Number of prunable layers")
    total_params: int = Field(..., description="Total model parameters")
    active_params: int = Field(..., description="Active (non-pruned) parameters")


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str = "ok"
    model_loaded: bool = False
