<template>
  <div>
    <h2 class="text-xl font-semibold mb-4">Training Data</h2>
    <pre class="bg-gray-100 p-4 rounded overflow-x-auto">{{ trainingData }}</pre>
  </div>
</template>

<script>
export default {
  data() {
    return {
      trainingData: {
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
    }
  }
}
</script>
