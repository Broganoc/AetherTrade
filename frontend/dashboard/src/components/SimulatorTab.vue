<template>
  <div>
    <h2 class="text-xl font-semibold mb-4">Simulator Data</h2>
    <pre class="bg-gray-100 p-4 rounded overflow-x-auto">{{ simulatorData }}</pre>
  </div>
</template>

<script>
export default {
  data() {
    return {
      simulatorData: {
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
    }
  }
}
</script>
