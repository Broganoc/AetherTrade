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

      <button @click="runTraining"
              class="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded shadow">
        Run Training
      </button>
    </div>

    <div v-if="trainingStatus" class="text-sm mb-4 text-blue-400">{{ trainingStatus }}</div>

    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      <div v-for="(model, index) in models" :key="model.model_name"
           class="bg-white dark:bg-gray-800 p-4 rounded-lg shadow hover:shadow-lg transition cursor-pointer"
           @click="toggleExpanded(index)">
        <div class="flex justify-between items-center">
          <h3 class="font-semibold text-gray-800 dark:text-gray-100">{{ model.model_name }}</h3>
          <span class="text-sm text-gray-500 dark:text-gray-400">{{ model.version }}</span>
        </div>

        <p class="text-sm text-gray-600 dark:text-gray-300">Framework: {{ model.framework }}</p>
        <p class="text-sm text-gray-600 dark:text-gray-300">Trained on: {{ model.trained_on }}</p>

        <div v-if="model.expanded" class="mt-3 border-t border-gray-200 dark:border-gray-700 pt-2 text-sm text-gray-700 dark:text-gray-300">
          <p><strong>Checksum:</strong> {{ model.checksum }}</p>
          <p><strong>Features:</strong> {{ model.features.join(', ') }}</p>
          <p><strong>Train Accuracy:</strong> {{ model.metrics.train_accuracy*100 }}%</p>
          <p><strong>Validation Accuracy:</strong> {{ model.metrics.validation_accuracy*100 }}%</p>
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
      selectedSymbol: "",
      symbols: ["AAPL", "TSLA", "MSFT", "GOOG", "AMZN"] // add more symbols as needed
    };
  },
  created() {
    this.fetchModels();
  },
  methods: {
    async fetchModels() {
      const res = await fetch("http://localhost:8001/models");
      this.models = await res.json();
      this.models.forEach(m => m.expanded = false);
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
      try {
        await fetch(`http://localhost:8001/train?symbol=${this.selectedSymbol}`, { method: "POST" });
        this.trainingStatus = `Training for ${this.selectedSymbol} running in background. Refresh models in a few moments.`;
      } catch (err) {
        console.error(err);
        this.trainingStatus = "Failed to start training.";
      }
    }
  }
}
</script>
