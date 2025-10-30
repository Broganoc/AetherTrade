<template>
  <div class="p-6 min-h-screen">
    <h2 class="text-xl font-semibold mb-4 dark:text-gray-100">Training Dashboard</h2>

    <!-- Symbol input & buttons -->
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
      <div class="text-sm mt-1 text-blue-400 dark:text-blue-300">
        {{ trainingStatus }}
        <span
          v-if="meanReward !== null && meanEpLength !== null"
          class="ml-2 text-xs text-gray-500 dark:text-gray-400"
        >
          (Mean Reward: {{ meanReward.toFixed(2) }}, Ep Len: {{ meanEpLength.toFixed(2) }})
        </span>
      </div>
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
            {{ model.model_name }} — {{ model.symbol }}
          </h3>
          <span class="text-sm text-gray-500 dark:text-gray-400">{{ model.trained_on }}</span>
        </div>

        <p class="text-sm text-gray-600 dark:text-gray-300">
          Framework: {{ model.framework }}
        </p>

        <!-- Mean Reward -->
        <p class="text-sm text-gray-600 dark:text-gray-300">
          Mean Reward:
          {{
            model.metrics?.mean_reward !== null && model.metrics?.mean_reward !== undefined
              ? model.metrics.mean_reward.toFixed(4)
              : model.validation_mean_reward !== undefined
                ? model.validation_mean_reward.toFixed(4)
                : model.mean_reward !== undefined
                  ? model.mean_reward.toFixed(4)
                  : "N/A"
          }}
        </p>

        <!-- Validation Mean Reward (if available) -->
        <p
          v-if="model.validation_mean_reward !== undefined"
          class="text-sm text-gray-600 dark:text-gray-300"
        >
          Validation Mean Reward: {{ model.validation_mean_reward.toFixed(4) }}
        </p>

        <!-- Mean Episode Length -->
        <p class="text-sm text-gray-600 dark:text-gray-300">
          Mean Episode Length:
          {{
            model.metrics?.mean_episode_length !== null && model.metrics?.mean_episode_length !== undefined
              ? model.metrics.mean_episode_length.toFixed(2)
              : model.mean_episode_length !== undefined
                ? model.mean_episode_length.toFixed(2)
                : "N/A"
          }}
        </p>

        <!-- Expanded view -->
        <transition name="fade">
          <div
            v-if="model.expanded"
            class="mt-3 border-t border-gray-200 dark:border-gray-700 pt-2 text-sm text-gray-700 dark:text-gray-300"
          >
            <p><strong>Path:</strong> {{ model.path }}</p>
            <p><strong>Metrics:</strong></p>
            <ul class="mt-2 space-y-1">
              <li>
                Mean Reward:
                {{
                  model.metrics?.mean_reward !== null && model.metrics?.mean_reward !== undefined
                    ? model.metrics.mean_reward.toFixed(4)
                    : model.validation_mean_reward !== undefined
                      ? model.validation_mean_reward.toFixed(4)
                      : model.mean_reward !== undefined
                        ? model.mean_reward.toFixed(4)
                        : "N/A"
                }}
              </li>
              <li v-if="model.validation_mean_reward !== undefined">
                Validation Mean Reward: {{ model.validation_mean_reward.toFixed(4) }}
              </li>
              <li>
                Mean Episode Length:
                {{
                  model.metrics?.mean_episode_length !== null && model.metrics?.mean_episode_length !== undefined
                    ? model.metrics.mean_episode_length.toFixed(2)
                    : model.mean_episode_length !== undefined
                      ? model.mean_episode_length.toFixed(2)
                      : "N/A"
                }}
              </li>
            </ul>
          </div>
        </transition>
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
      meanReward: null,
      meanEpLength: null,
    };
  },
  created() {
    this.fetchModels();
  },
  beforeUnmount() {
    if (this.eventSource) this.eventSource.close();
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
      this.trainingStatus = `Training started for ${this.selectedSymbol}...`;
      this.meanReward = null;
      this.meanEpLength = null;

      const url = `http://localhost:8001/train_stream?symbol=${this.selectedSymbol}`;
      this.eventSource = new EventSource(url);

      this.eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.status === "started") {
          this.trainingProgress = 0;
          this.trainingStatus = `Training started for ${data.symbols?.join(", ") || this.selectedSymbol}`;
        } else if (data.status === "training") {
          this.trainingProgress = data.progress || 0;
          this.meanReward = data.mean_reward ?? null;
          this.meanEpLength = data.mean_episode_length ?? null;
          this.trainingStatus = `Training ${data.symbols?.join(", ") || this.selectedSymbol}: ${data.progress}%`;
        } else if (data.status === "completed") {
          this.trainingProgress = 100;
          this.trainingStatus = `✅ Training completed for ${data.symbols?.join(", ") || this.selectedSymbol}`;
          this.meanReward = data.val_mean ?? this.meanReward;
          this.isTraining = false;

          setTimeout(() => {
            this.trainingStatus = "";
            this.trainingProgress = 0;
            this.fetchModels();
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
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
