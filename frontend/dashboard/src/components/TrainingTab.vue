<template>
  <div class="p-6 min-h-screen">
    <h2 class="text-xl font-semibold mb-4 dark:text-gray-100">Training Data</h2>

    <div class="mb-4 flex items-center gap-2">
      <input
        v-model="selectedSymbol"
        list="symbols-list"
        placeholder="Enter stock symbol"
        class="px-3 py-2 border rounded dark:bg-gray-700 dark:text-gray-100"
      />
      <datalist id="symbols-list">
        <option v-for="symbol in symbols" :key="symbol" :value="symbol">{{ symbol }}</option>
      </datalist>

      <button
        @click="runTraining"
        class="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded shadow"
      >
        Run Training
      </button>
    </div>

    <!-- Training progress bar -->
    <div v-if="trainingStatus" class="mb-4">
      <div class="w-full bg-gray-200 rounded h-4 dark:bg-gray-700">
        <div
          class="h-4 bg-blue-500 rounded transition-all"
          :style="{ width: trainingProgress + '%' }"
        ></div>
      </div>
      <div class="text-sm text-blue-400 mt-1">{{ trainingStatus }}</div>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      <div
        v-for="(model, index) in models"
        :key="model.model_name"
        class="bg-white dark:bg-gray-800 p-4 rounded-lg shadow hover:shadow-lg transition cursor-pointer"
        @click="toggleExpanded(index)"
      >
        <div class="flex justify-between items-center">
          <h3 class="font-semibold text-gray-800 dark:text-gray-100">{{ model.model_name + " " + model.symbol }}</h3>
          <span class="text-sm text-gray-500 dark:text-gray-400">{{ model.trained_on }}</span>
        </div>

        <p class="text-sm text-gray-600 dark:text-gray-300">Framework: {{ model.framework }}</p>
        <p v-if="model.metrics.mean_reward !== null" class="text-sm text-gray-600 dark:text-gray-300">
          Mean Reward: {{ model.metrics.mean_reward.toFixed(4) }}
        </p>
        <p v-if="model.metrics.mean_episode_length !== null" class="text-sm text-gray-600 dark:text-gray-300">
          Mean Episode Length: {{ model.metrics.mean_episode_length.toFixed(2) }}
        </p>

        <div v-if="model.expanded" class="mt-3 border-t border-gray-200 dark:border-gray-700 pt-2 text-sm text-gray-700 dark:text-gray-300">
          <p><strong>Path:</strong> {{ model.path }}</p>
          <p><strong>Metrics:</strong></p>
          <ul>
            <li>Mean Reward: {{ model.metrics.mean_reward !== null ? model.metrics.mean_reward.toFixed(4) : 'N/A' }}</li>
            <li>Mean Episode Length: {{ model.metrics.mean_episode_length !== null ? model.metrics.mean_episode_length.toFixed(2) : 'N/A' }}</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      models: [],
      trainingStatus: "",
      trainingProgress: 0,
      selectedSymbol: "",
      symbols: ["AAPL", "TSLA", "MSFT", "GOOG", "AMZN"], // add more symbols
      statusInterval: null,
    };
  },
  created() {
    this.fetchModels();
    this.checkTrainingStatus(); // Check initial status to start polling if training is ongoing
  },
  beforeUnmount() {
    if (this.statusInterval) {
      clearInterval(this.statusInterval);
    }
  },
  methods: {
    async fetchModels() {
      try {
        const res = await fetch("http://localhost:8001/models");
        this.models = await res.json();
        this.models.forEach(m => (m.expanded = false));
      } catch (err) {
        console.error("Failed to fetch models:", err);
      }
    },
    toggleExpanded(index) {
      this.models[index].expanded = !this.models[index].expanded;
    },
    async runTraining() {
      if (!this.selectedSymbol) {
        this.trainingStatus = "Please enter a stock symbol!";
        return;
      }
      this.trainingStatus = "Training started...";
      this.trainingProgress = 0;

      try {
        await fetch(
          `http://localhost:8001/train?symbol=${this.selectedSymbol}`,
          { method: "POST" }
        );
        this.checkTrainingStatus(); // Check status immediately after starting training
      } catch (err) {
        console.error(err);
        this.trainingStatus = "Failed to start training.";
      }
    },
    async checkTrainingStatus() {
      try {
        const res = await fetch("http://localhost:8001/training_status");
        const data = await res.json();

        if (!data || data.status === "idle") {
          this.trainingStatus = "";
          this.trainingProgress = 0;
        } else if (data.status === "started" || data.status === "training") {
          const totalChunks = data.chunks || 1;
          const currentChunk = data.current_chunk || 0;
          const progress = Math.min((currentChunk / totalChunks) * 100, 100);
          this.trainingProgress = progress;
          this.trainingStatus = `Training ${data.symbol}: chunk ${currentChunk}/${totalChunks} (${progress.toFixed(1)}%)`;
        } else if (data.status === "completed") {
          this.trainingProgress = 100;
          this.trainingStatus = `Training completed for ${data.symbol}. Model saved at ${data.model_path}`;
          this.fetchModels(); // refresh model list
        } else if (data.status === "error") {
          this.trainingProgress = 0;
          this.trainingStatus = `Error during training: ${data.message}`;
        }

        // Manage polling based on status
        if (data && (data.status === "completed" || data.status === "error" || data.status === "idle")) {
          if (this.statusInterval) {
            clearInterval(this.statusInterval);
            this.statusInterval = null;
          }
        } else {
          if (!this.statusInterval) {
            this.startStatusPolling();
          }
        }
      } catch (err) {
        console.error("Failed to fetch training status:", err);
      }
    },
    startStatusPolling() {
      this.statusInterval = setInterval(this.checkTrainingStatus, 3000); // poll every 3s
    },
  },
};
</script>