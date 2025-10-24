<template>
  <div class="p-6 relative min-h-screen">
    <h2 class="text-xl font-semibold mb-4 dark:text-gray-100">Simulation Dashboard</h2>

    <!-- Simulation Form -->
    <div class="bg-gray-100 dark:bg-gray-800 p-6 rounded-lg shadow mb-6 relative z-10">
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div>
          <label class="block text-sm font-medium mb-1 dark:text-gray-200">Model</label>
          <select v-model="selectedModel" class="w-full p-2 border rounded bg-white dark:bg-gray-700 dark:text-gray-200">
            <option value="">Select Model</option>
            <option v-for="model in models" :key="model.full_name" :value="model.full_name">
              {{ model.full_name }}
            </option>
          </select>
        </div>

        <div>
          <label class="block text-sm font-medium mb-1 dark:text-gray-200">Start Date</label>
          <input type="date" v-model="startDate" class="w-full p-2 border rounded bg-white dark:bg-gray-700 dark:text-gray-200" />
        </div>

        <div>
          <label class="block text-sm font-medium mb-1 dark:text-gray-200">End Date</label>
          <input type="date" v-model="endDate" class="w-full p-2 border rounded bg-white dark:bg-gray-700 dark:text-gray-200" />
        </div>

        <div>
          <label class="block text-sm font-medium mb-1 dark:text-gray-200">Starting Portfolio ($)</label>
          <input type="number" v-model.number="portfolioStart" class="w-full p-2 border rounded bg-white dark:bg-gray-700 dark:text-gray-200" />
        </div>
      </div>

      <div v-if="validationError" class="text-red-600 text-sm mt-3">{{ validationError }}</div>

      <button
        @click="runSimulation"
        :disabled="isRunning"
        class="mt-4 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded disabled:opacity-50"
      >
        Run Simulation
      </button>
    </div>

    <!-- Simulation Summary -->
    <div
      v-if="simulatorData"
      class="bg-white dark:bg-gray-800 p-6 rounded-lg shadow mb-6 relative z-10"
    >
      <h3 class="text-lg font-semibold mb-2 dark:text-gray-100">Simulation Summary</h3>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-2 text-sm dark:text-gray-200">
        <p><strong>Model Used:</strong> {{ simulatorData.model_used }}</p>
        <p><strong>Date Range:</strong> {{ simulatorData.start_date }} → {{ simulatorData.end_date }}</p>
        <p><strong>Portfolio Start:</strong> ${{ simulatorData.portfolio_start.toLocaleString() }}</p>
        <p><strong>Portfolio End:</strong> ${{ simulatorData.portfolio_end.toLocaleString() }}</p>
        <p><strong>Total Trades:</strong> {{ simulatorData.total_trades }}</p>
        <p><strong>Max Drawdown:</strong> {{ simulatorData.max_drawdown_pct }}%</p>
        <p><strong>Avg Trade Return:</strong> {{ simulatorData.avg_trade_return_pct }}%</p>
        <p><strong>Best Trade:</strong> {{ simulatorData.best_trade_return_pct }}%</p>
        <p><strong>Worst Trade:</strong> {{ simulatorData.worst_trade_return_pct }}%</p>
      </div>
    </div>

    <!-- Trade Log Table -->
    <div
      v-if="trades.length"
      class="bg-white dark:bg-gray-800 p-6 rounded-lg shadow relative z-10"
    >
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-lg font-semibold dark:text-gray-100">Trade Log</h3>
        <button
          @click="downloadCSV"
          class="bg-green-600 hover:bg-green-700 text-white text-sm px-3 py-1 rounded"
        >
          Download CSV
        </button>
      </div>

      <div class="overflow-x-auto">
        <table class="min-w-full text-sm border-collapse border border-gray-300 dark:border-gray-700">
          <thead class="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-100">
            <tr>
              <th class="px-3 py-2 border border-gray-300 dark:border-gray-700">Step</th>
              <th class="px-3 py-2 border border-gray-300 dark:border-gray-700">Date</th>
              <th class="px-3 py-2 border border-gray-300 dark:border-gray-700">Action</th>
              <th class="px-3 py-2 border border-gray-300 dark:border-gray-700">Price Δ (%)</th>
              <th class="px-3 py-2 border border-gray-300 dark:border-gray-700">Option Return (%)</th>
              <th class="px-3 py-2 border border-gray-300 dark:border-gray-700">Portfolio ($)</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="trade in trades"
              :key="trade.step"
              class="border-t border-gray-300 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700"
            >
              <td class="px-3 py-1 text-center">{{ trade.step }}</td>
              <td class="px-3 py-1 text-center">{{ trade.date }}</td>
              <td class="px-3 py-1 text-center">{{ trade.action }}</td>
              <td class="px-3 py-1 text-center">{{ trade.price_change_pct }}</td>
              <td class="px-3 py-1 text-center">{{ trade.option_return_pct }}</td>
              <td class="px-3 py-1 text-center">
                ${{ trade.portfolio_value.toLocaleString() }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Running Simulation Overlay -->
    <transition name="fade">
      <div
        v-if="isRunning"
        class="fixed inset-0 bg-black/40 flex items-center justify-center z-50"
      >
        <div
          class="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg flex items-center space-x-4"
        >
          <svg
            class="animate-spin h-6 w-6 text-blue-600"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              class="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              stroke-width="4"
            ></circle>
            <path
              class="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8v8H4z"
            ></path>
          </svg>
          <span class="font-semibold text-gray-700 dark:text-gray-200">
            Running Simulation...
          </span>
        </div>
      </div>
    </transition>
  </div>
</template>

<script>
export default {
  data() {
    return {
      models: [],
      selectedModel: "",
      startDate: "",
      endDate: "",
      portfolioStart: 100000,
      simulatorData: null,
      trades: [], // ✅ declare here
      validationError: "",
      isRunning: false,
    };
  },
  created() {
    this.fetchModels();
  },
  methods: {
    async fetchModels() {
      try {
        const res = await fetch("http://localhost:8001/models");
        this.models = await res.json();
      } catch (err) {
        console.error("Failed to fetch models:", err);
      }
    },
    async runSimulation() {
      this.validationError = "";
      if (!this.selectedModel || !this.startDate || !this.endDate) {
        this.validationError = "Please fill out all fields.";
        return;
      }
      const start = new Date(this.startDate);
      const end = new Date(this.endDate);
      if (end <= start) {
        this.validationError = "End date must be after start date.";
        return;
      }

      this.isRunning = true;
      try {
        const response = await fetch("http://localhost:8002/run-sim", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_used: this.selectedModel,
            start_date: this.startDate,
            end_date: this.endDate,
            portfolio_start: this.portfolioStart,
          }),
        });

        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || "Simulation failed");
        }

        const data = await response.json();
        this.simulatorData = data.summary;
        this.trades = data.trades || [];
      } catch (err) {
        console.error("Simulation failed:", err);
        this.validationError = err.message || "Simulation failed. Check backend logs.";
      } finally {
        this.isRunning = false;
      }
    },

    // ✅ Download CSV handler
    downloadCSV() {
      if (!this.trades.length) return;

      const headers = Object.keys(this.trades[0]);
      const csvRows = [
        headers.join(","), // header row
        ...this.trades.map(trade =>
          headers.map(h => JSON.stringify(trade[h] ?? "")).join(",")
        ),
      ];
      const blob = new Blob([csvRows.join("\n")], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${this.selectedModel}_trade_log.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    },
  },
};
</script>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
