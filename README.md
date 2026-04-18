# Self-Pruning Neural Network

> A production-grade neural network that **learns to prune itself** during training. The network uses differentiable gate parameters on every weight connection, and an L1 sparsity penalty drives unnecessary gates toward zero — removing those weights entirely. The result: a sparse, efficient network that maintains accuracy while eliminating redundant connections.

## What Makes This Interesting

This isn't just a homework exercise in sparsity. Five specific features map directly to production ML engineering:

| Feature | Why It Matters |
|---------|---------------|
| **FastAPI Inference Server** | Demonstrates deployment-ready model serving with caching and health checks |
| **Animated Gate Evolution GIF** | Visualizes the pruning process epoch-by-epoch — not just final results |
| **FLOPs Counting** | Measures actual compute reduction, not just parameter count |
| **Gradient Flow Tests** | Verifies the custom layer's differentiability with pytest |
| **Lottery Ticket Hypothesis Connection** | Shows awareness of the theoretical foundations (Frankle & Carlin, 2019) |

## Architecture

```
CIFAR-10 Input (3×32×32 = 3072)
         │
    ┌────┼────────────────────────────┐
    │    ▼                            │
    │  PrunableLinear(3072→1024) + BN + GELU
    │    │                            │
    │  PrunableLinear(1024→512)  + BN + GELU
    │    │                            │
    │  PrunableLinear(512→256)   + BN + GELU
    │    │                            │
    │    ▼                            ▼
    │  (add) ◄────── skip_proj(3072→256)
    │    │
    │  PrunableLinear(256→128) + GELU
    │    │
    │  Linear(128→10) [classifier, not prunable]
    │    │
    │    ▼
    │  10-class output
    └─────────────────────────────────┘
```

Each `PrunableLinear` layer has a learnable gate per weight: `output = x @ (W ⊙ σ(gate_scores))ᵀ + b`

## Quick Start

```bash
# 1. Setup
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt

# 2. Run all experiments (6 configs × 50 epochs)
python experiments/run_all.py

# 3. Quick smoke test (2 configs × 5 epochs)
python experiments/run_all.py --quick

# 4. Run tests
pytest tests/ -v

# 5. Start API server
uvicorn src.api.server:app --port 8000

# 6. Docker deployment
docker-compose up --build
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify CIFAR-10 image, return class + confidence |
| `/sparsity-report` | GET | Per-layer and overall pruning statistics |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive OpenAPI documentation |

## Experiment Configurations

| Config | λ_max | Schedule | Purpose |
|--------|-------|----------|---------|
| `ultra_low` | 1e-5 | static | Minimal pruning baseline |
| `low` | 1e-4 | static | Light pruning |
| `mid` | 1e-3 | static | Moderate pruning |
| `high` | 1e-2 | static | Aggressive pruning |
| `annealed_mid` | 1e-3 | cosine warmup | Gradual pruning onset |
| `annealed_high` | 1e-2 | linear warmup | Aggressive with grace period |

## Project Structure

```
self-pruning-nn/
├── src/
│   ├── layers/          ← PrunableLinear + gate strategies
│   ├── models/          ← PrunableNet + BaselineNet
│   ├── losses/          ← SparsityLoss (L1/L2/Hoyer)
│   ├── training/        ← Trainer + LambdaScheduler
│   ├── evaluation/      ← Metrics + gradient diagnostics
│   ├── visualization/   ← Plots + animated GIF
│   └── api/             ← FastAPI server + caching
├── experiments/         ← Run scripts + YAML configs
├── tests/               ← pytest test suite
├── outputs/             ← Checkpoints, plots, results
├── report.md            ← Technical write-up
├── Dockerfile           ← Production container
└── docker-compose.yml   ← App + Redis deployment
```

## Key Design Decisions

1. **Gate initialization at 0.5**: `sigmoid(0.5) ≈ 0.62` — most gates start open so the network can learn before pruning kicks in.
2. **L1 over L2**: L1 has constant gradient `d|g|/dg = 1`, pushing gates to *exactly* zero. L2's gradient `2g → 0` only makes gates small, never truly zero.
3. **Skip connection**: Enables the network to maintain accuracy even when middle layers are heavily pruned.
4. **Lambda warmup**: λ=0 for the first N epochs, then ramps up. Prevents premature pruning of important connections.

## License

MIT
