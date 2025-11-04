# AetherTrade

**Reinforcement Learning–Driven AI Trading Framework**

AetherTrade is a modular research and simulation platform for developing, training, and evaluating AI trading agents.  
It uses reinforcement learning (RL) to train agents to trade options and equities using historical market data and provides a real-time web dashboard to visualize training progress, performance, and simulations.

---

## Key Features

- Train PPO (Proximal Policy Optimization) agents using `stable-baselines3`
- Resume training from saved checkpoints (supports multi-symbol models)
- Run simulations (backtests) on historical data with detailed trade-level logs
- Real-time dashboard with:
  - Live metrics via Server-Sent Events (SSE)
  - Progress tracking, reward and loss visualization, validation stats
  - Trade log filtering, alternating row colors, and P&L highlighting
- Atomic model saving with `VecNormalize` statistics
- Redis-powered live status updates for asynchronous monitoring
- Fully containerized via Docker for reproducibility and deployment

---

## Architecture

AetherTrade is composed of three core services:

| Module | Description |
|---------|--------------|
| **Trainer** | Trains PPO agents using a custom `OptionTradingEnv` and streams metrics via SSE |
| **Simulator** | Loads trained models, runs backtests on historical data, and returns trade logs and portfolio performance |
| **Dashboard** | Vue 3 + Tailwind web interface for launching, resuming, and visualizing model performance |

All modules communicate through **FastAPI JSON APIs** and **Server-Sent Events**, sharing persistent storage under `/app/models` and `/app/logs`.

---

## Tech Stack

| Category | Technologies |
|-----------|--------------|
| **Frontend** | Vue 3, Vite, Tailwind CSS |
| **Backend** | FastAPI (Python 3.11+), stable-baselines3 |
| **Environment** | Custom `OptionTradingEnv` (Gymnasium), yfinance, pandas_ta |
| **State / Cache** | Redis |
| **Machine Learning** | PyTorch, NumPy |
| **Infrastructure** | Docker Compose, persistent volumes |
| **Visualization** | TensorBoard, live SSE dashboard |

---

## Quick Start

### 1. Clone and Launch
```bash
git clone https://github.com/<your-username>/AetherTrade.git
cd AetherTrade
docker-compose up --build
```

### 2. Set up Docker environment
```bash
docker-compose up --build
```
This will launch:

- FastAPI backend services (Trainer & Simulator)
- Redis for task queue
- Vite dashboard on default port 5173

### 3. Access the dashboard

Open your browser and navigate to Vite/Vue:

**http://localhost:5173**

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
