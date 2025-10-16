# AetherTrade

**Modular AI Trading & Simulation Framework**

AetherTrade is a modular, extensible platform for experimenting with AI-driven trading strategies.  
It provides tools for **training predictive models**, **simulating trades on historical data**, and **visualizing results** via a lightweight web dashboard.  

The project is designed to be **modular**, **scalable**, and **future-proof**, allowing easy integration of new AI models, multi-asset simulations, and eventually live paper trading.

---

## Features

- Train and test machine learning or reinforcement learning models.
- Simulate trades on historical financial data with detailed portfolio tracking.
- Modular dashboard for monitoring models and simulation results.
- Async task execution with Celery + Redis for long-running tasks.
- JSON-based APIs for interoperability and frontend integration.
- Extensible architecture ready for multiple assets, broker APIs, and mobile clients.

---

## Architecture

AetherTrade consists of three main modules:

1. **AI Trainer** – Python module to train, test, and save models with metadata.
2. **Simulator** – Python module to run backtests and output portfolio stats and trade histories.
3. **Dashboard** – Web-based React frontend for controlling modules and visualizing results.

All modules communicate via **JSON REST APIs** and can run independently or together.

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **Frontend** | React, TypeScript, Recharts/ApexCharts |
| **Backend** | FastAPI (Python 3.11+) |
| **Task Queue** | Celery + Redis |
| **AI/ML** | scikit-learn, pandas, numpy, PyTorch, stable-baselines3 |
| **Data** | yfinance, pandas_ta |
| **Storage** | SQLite (dev), Redis cache, Postgres (future) |
| **Infrastructure** | Docker Compose, optional GPU support |
| **Monitoring** | Prometheus + Grafana |

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/AetherTrade.git
cd AetherTrade
```

### 2. Set up Docker environment
```bash
docker-compose up --build
```
This will launch:

- FastAPI backend services (Trainer & Simulator)
- Redis for task queue
- React dashboard on default port 3000

### 3. Access the dashboard

Open your browser and navigate to:

**http://localhost:3000**

### 4. Start using

- Launch a new model training
- Run simulations on historical data
- Visualize results in the dashboard

---

## Testing

| Test Type | Tools | Purpose |
|-----------|--------|---------|
| **Unit Tests** | pytest | Trainer and Simulator logic |
| **Integration Tests** | Docker Compose | E2E tests |
| **Load Testing** | locust | Verify queue scalability |
| **Monitoring** | Prometheus + Grafana | Performance tracking |

---

## Future Plans

- Multi-asset simulations
- Integration with live paper trading brokers (Alpaca, Interactive Brokers)
- Reinforcement learning research sandbox
- Monte Carlo and stress testing
- Mobile client via React Native or Capacitor
- Model governance dashboard for reproducibility

---

## Notes & Considerations

- Serialized models (`.pkl`, `.joblib`) are unsafe from untrusted sources — use caution.
- Backtests are not fully realistic without considering slippage, fees, and latency.
- For high-frequency or large-scale simulations, consider distributed execution (Ray/Dask).
- Compliance and broker API limitations must be respected if evolving to live trading.

---

## 📄 License

MIT License © 2025 Brogan O'Connor

---
*Built with ❤️ for AI trading experimentation*
