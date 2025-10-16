# Project Overview & Development Notes
### Modular AI Trading & Simulation System

---

## Project Vision

This project aims to create a **modular, extensible AI-driven trading environment** that allows experimentation with machine learning and reinforcement learning (RL) strategies for financial decision-making.  

The design emphasizes **separation of concerns**, scalability, and maintainability — each component operates independently but communicates through clearly defined APIs or data exchange formats (e.g., JSON).  

Ultimately, this system will:
- **Train and test predictive trading models**
- **Simulate trading strategies over historical data**
- **Display results and insights through a modern web dashboard**
- **Enable flexible experimentation and easy future expansion**

---

## High-Level Architecture

The project consists of three primary modules, each encapsulated as a microservice or standalone component:

### 1. **Dashboard (Frontend)**
- A **web-based control center** for viewing models, launching simulations, and visualizing results.  
- Built using **React (or Vite + React)** with an API-driven architecture.
- Displays model metadata, simulation stats, trade histories, and performance charts.
- All backend communication occurs through JSON REST APIs (and later possibly WebSockets).

### 2. **AI Trainer**
- A backend module written in **Python (FastAPI)** responsible for training models.
- The AI layer will support:
  - Traditional ML algorithms (`scikit-learn`, `xgboost`)
  - Reinforcement Learning agents (`stable-baselines3`)
- Outputs:
  - Trained model artifact (e.g., `.pkl`, `.pt`, or `.safetensors`)
  - `metadata.json` describing the model, training configuration, and preprocessing pipeline
- Initially, this will remain **generic** — a placeholder for model training that can be customized once a trading approach or dataset is defined.

### 3. **Simulator**
- Another **Python FastAPI microservice** that:
  - Loads a trained model from disk
  - Fetches historical OHLC data (via `yfinance` or cached sources)
  - Simulates trades based on model predictions
  - Tracks portfolio metrics and trade-level details
- Outputs:
  - JSON reports with all trades, profits/losses, statistics, and final portfolio summary
- Designed to support **multi-asset simulations**, realistic commissions, slippage, and position sizing.

---

## Data Flow Overview

```text
                  
                        ┌──────────────┐        ┌───────────────────┐        ┌──────────────┐
                        │   Dashboard  │ ─────> │  FastAPI Gateway  │ ─────> │    Trainer   │
                        │  (React UI)  │ <───── │  (Router Layer)   │ <───── │  & Simulator │
                        └──────────────┘        └───────────────────┘        └──────────────┘
                                                          │
                                                          ▼
                                                  Redis / SQLite / FS

```

1. **Frontend (React)** sends JSON requests to the backend API.  
2. **FastAPI services** (Trainer & Simulator) handle logic and return results as JSON.  
3. **Data and model artifacts** are cached or persisted via SQLite/Redis/local FS.  
4. The dashboard visualizes this data as charts, tables, and reports.

---

## Core Functional Goals

| Module | Core Responsibilities | Output Format |
|---------|----------------------|----------------|
| **AI Trainer** | Train, test, and save ML/RL models | `metadata.json`, `model.pkl`, metrics JSON |
| **Simulator** | Replay model decisions on historical data | Portfolio stats JSON, trades JSON |
| **Dashboard** | Display, control, and interact with AI & simulation results | REST/JSON interface |

---

## Design Principles

### Modularity
Each module (Trainer, Simulator, Dashboard) functions independently and can be developed, deployed, or replaced without affecting others.  
APIs act as the “contract” between components — JSON in/out ensures flexibility.

### Scalability
All heavy operations (training, simulation) are structured as asynchronous background tasks using **Celery** + **Redis**.  
This will enable distributed task execution in the future and allow live progress tracking through job polling or WebSockets.

### Extensibility
The architecture is designed to grow. You could later:
- Integrate **real broker APIs** (e.g., Alpaca, Interactive Brokers)
- Add **multiple AI models** for ensemble predictions
- Support **multi-user accounts**
- Transition to **cloud-hosted microservices**

### Transparency
Every model and simulation run generates detailed metadata and logs, allowing full reproducibility of results.

---

## Development Philosophy

Rather than rushing to implement everything at once, this project prioritizes:
1. **Building modular components individually**
2. **Testing each part in isolation**
3. **Integrating only after stability and reliability are verified**

This ensures:
- No circular dependencies
- Clear debugging paths
- Easier unit/integration testing
- A strong foundation for iterative development

---

## Tech Stack Overview

| Layer | Tools & Frameworks |
|--------|--------------------|
| **Frontend** | React, TypeScript, Recharts/ApexCharts, Axios |
| **Backend APIs** | FastAPI (Python 3.11+) |
| **Task Queue** | Celery + Redis |
| **AI/ML** | scikit-learn, pandas, numpy, torch, stable-baselines3 |
| **Data Handling** | yfinance, pandas_ta |
| **Storage** | SQLite (dev), Postgres (future), Redis cache |
| **Infrastructure** | Docker Compose (with optional GPU support) |
| **Monitoring** | Prometheus + Grafana |

---

## Expected Outputs

### AI Trainer
```json
{
  "model_name": "ppo_agent_v1",
  "version": "1.0.0",
  "checksum": "7a3d9f8b...",
  "framework": "stable-baselines3",
  "trained_on": "2025-10-16",
  "features": ["rsi", "ema_20", "close_price"],
  "metrics": {
    "train_accuracy": 0.82,
    "validation_accuracy": 0.79
  }
}
```

### Simulator

```json
{
  "simulation_id": "sim_001",
  "model_used": "ppo_agent_v1",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "portfolio_start": 100000,
  "portfolio_end": 126000,
  "total_trades": 234,
  "max_drawdown_pct": 12.4,
  "avg_trade_return_pct": 0.8,
  "best_trade_return_pct": 5.6,
  "worst_trade_return_pct": -3.1
}
```

---

## Testing & Validation

| Type                  | Tool                 | Description                                        |
| --------------------- | -------------------- | -------------------------------------------------- |
| **Unit Tests**        | Pytest               | Validate Trainer and Simulator logic independently |
| **Integration Tests** | Docker Compose E2E   | Launch stack and simulate real API interactions    |
| **Load Testing**      | Locust               | Verify async queue scalability                     |
| **Monitoring**        | Prometheus / Grafana | Track job time, API latency, and resource usage    |

---

## Future Expansion Ideas

* **Reinforcement Learning Research Lab:** Add a sandbox for comparing RL algorithms under different hyperparameters.
* **Live Paper Trading:** Integrate with Alpaca API for simulation-to-live transition.
* **Monte Carlo Stress Tests:** Assess model robustness under synthetic volatility.
* **Model Governance Dashboard:** Central UI for reviewing all past runs and metrics.
* **Mobile Client:** Build a lightweight native app via Capacitor or React Native.

---

## Considerations and Caveats

1. **Security:** Serialized models (`.pkl`, `.joblib`) are not safe to load from untrusted sources — restrict access and consider safer formats like `safetensors`.
2. **Realism:** Backtesting results can be misleading without realistic trading conditions (slippage, latency, fees). Simulations will account for these progressively.
3. **Data Rate Limits:** `yfinance` has API throttling; caching with Redis mitigates repetitive requests.
4. **Scalability:** For high-frequency or multi-asset simulations, distributed processing with Ray or Dask may be necessary.
5. **Compliance:** If extended to live trading, broker API compliance and paper-trading validation will be mandatory.

---

## Summary

This project will evolve into a **flexible experimentation and decision-support system for trading strategies**, starting from a controlled local prototype and growing into a scalable, server-hosted platform.

The guiding principle:

> *“Design everything modular, integrate nothing prematurely.”*

This ensures a stable foundation for reliable AI experimentation, future scalability, and maintainable code evolution.

---

**Author:** Brogan O’Connor

**Last Updated:** *(YYYY-MM-DD)*

**Version:** 1.0.0 — Project Overview Release
