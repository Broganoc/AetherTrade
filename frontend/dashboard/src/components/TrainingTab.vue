<template>
  <div class="p-6 min-h-screen">
    <h2 class="text-xl font-semibold mb-4 dark:text-gray-100">Training Data</h2>

    <!-- Symbol input & buttons -->
    <div class="mb-4 flex items-center gap-2">
      <input
        v-model="selectedSymbol"
        list="symbols-list"
        placeholder="Enter stock symbol"
        class="px-3 py-2 border rounded dark:bg-gray-700 dark:text-gray-100"
      />
      <datalist id="symbols-list">
        <option
          v-for="symbol in symbols"
          :key="symbol"
          :value="symbol"
        >
          {{ symbol }}
        </option>
      </datalist>

      <button
        @click="runTraining"
        :disabled="isTraining"
        class="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded shadow disabled:opacity-50"
      >
        {{ isTraining ? "Training..." : "Run Training" }}
      </button>

      <button
        v-if="isTraining"
        @click="cancelTraining"
        class="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded shadow"
      >
        Cancel
      </button>
    </div>

    <!-- Progress bar -->
    <div v-if="trainingStatus" class="mb-4">
      <div class="w-full bg-gray-200 rounded h-4 dark:bg-gray-700 overflow-hidden">
        <div
          class="h-4 bg-blue-500 text-xs text-white flex items-center justify-center transition-all duration-300"
          :style="{ width: trainingProgress + '%' }"
        >
          {{ trainingProgress }}%
        </div>
      </div>
      <div class="text-sm text-blue-400 mt-1">{{ trainingStatus }}</div>
    </div>

    <!-- Models Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      <div
        v-for="(model, index) in models"
        :key="model.model_name"
        class="bg-white dark:bg-gray-800 p-4 rounded-lg shadow hover:shadow-lg transition cursor-pointer"
        @click="toggleExpanded(index)"
      >
        <div class="flex justify-between items-center">
          <h3 class="font-semibold text-gray-800 dark:text-gray-100">
            {{ model.model_name + " " + model.symbol }}
          </h3>
          <span class="text-sm text-gray-500 dark:text-gray-400">{{ model.trained_on }}</span>
        </div>

        <p class="text-sm text-gray-600 dark:text-gray-300">Framework: {{ model.framework }}</p>
        <p v-if="model.metrics.mean_reward !== null" class="text-sm text-gray-600 dark:text-gray-300">
          Mean Reward: {{ model.metrics.mean_reward.toFixed(4) }}
        </p>
        <p v-if="model.metrics.mean_episode_length !== null" class="text-sm text-gray-600 dark:text-gray-300">
          Mean Episode Length: {{ model.metrics.mean_episode_length.toFixed(2) }}
        </p>

        <div
          v-if="model.expanded"
          class="mt-3 border-t border-gray-200 dark:border-gray-700 pt-2 text-sm text-gray-700 dark:text-gray-300"
        >
          <p><strong>Path:</strong> {{ model.path }}</p>
          <p><strong>Metrics:</strong></p>
          <ul>
            <li>
              Mean Reward:
              {{
                model.metrics.mean_reward !== null
                  ? model.metrics.mean_reward.toFixed(4)
                  : "N/A"
              }}
            </li>
            <li>
              Mean Episode Length:
              {{
                model.metrics.mean_episode_length !== null
                  ? model.metrics.mean_episode_length.toFixed(2)
                  : "N/A"
              }}
            </li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "TrainingTab",
  data() {
    return {
      models: [],
      trainingStatus: "",
      trainingProgress: 0,
      selectedSymbol: "",
      symbols: ["AAPL", "TSLA", "MSFT", "GOOG", "AMZN"],
      eventSource: null,
      isTraining: false,
    };
  },
  created() {
    this.fetchModels();
  },
  beforeUnmount() {
    if (this.eventSource) {
      this.eventSource.close();
    }
  },
  methods: {
    async fetchModels() {
      try {
        const res = await fetch("http://localhost:8001/models");
        const data = await res.json();
        this.models = data.map((m) => ({ ...m, expanded: false }));
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

      this.isTraining = true;
      this.trainingProgress = 0;
      this.trainingStatus = "Training started...";

      // Create SSE connection
      const url = `http://localhost:8001/train_stream?symbol=${this.selectedSymbol}`;
      this.eventSource = new EventSource(url);

      this.eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.status === "started") {
          this.trainingProgress = 0;
          this.trainingStatus = `Training started for ${data.symbol}`;
        } else if (data.status === "training") {
          this.trainingProgress = data.progress;
          this.trainingStatus = `Training ${data.symbol}: ${data.progress}% complete`;
        } else if (data.status === "completed") {
          this.trainingProgress = 100;
          this.trainingStatus = `✅ Training completed for ${data.symbol}`;
          this.isTraining = false;

          setTimeout(() => {
            this.trainingStatus = "";
            this.trainingProgress = 0;
            this.fetchModels(); // refresh models
          }, 1500);

          this.eventSource.close();
          this.eventSource = null;
        }
      };

      this.eventSource.onerror = (err) => {
        console.error("SSE Error:", err);
        this.trainingStatus = "⚠️ Connection lost.";
        this.isTraining = false;
        if (this.eventSource) {
          this.eventSource.close();
          this.eventSource = null;
        }
      };
    },

    async cancelTraining() {
      if (this.eventSource) {
        this.eventSource.close();
        this.eventSource = null;
      }

      this.isTraining = false;
      this.trainingStatus = "❌ Training cancelled.";

      try {
        await fetch("http://localhost:8001/training_status", { method: "DELETE" });
      } catch (err) {
        console.error("Failed to cancel training:", err);
      }

      setTimeout(() => {
        this.trainingProgress = 0;
        this.trainingStatus = "";
      }, 1000);
    },
  },
};
</script>

<style scoped>
.disabled\:opacity-50 {
  opacity: 0.5;
}
</style>
