<template>
  <div class="p-6 relative min-h-screen">
    <h2 class="text-xl font-semibold mb-4 dark:text-gray-100">Simulation Dashboard</h2>

    <!-- Simulation Form -->
    <div class="bg-gray-100 dark:bg-gray-800 p-6 rounded-lg shadow mb-6 relative z-10">
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
        <!-- Model -->
        <div>
          <label class="block text-sm font-medium mb-1 dark:text-gray-200">Model</label>
          <select
            v-model="selectedModel"
            class="w-full p-2 border rounded bg-white dark:bg-gray-700 dark:text-gray-200"
          >
            <option value="">Select Model</option>
            <option v-for="model in models" :key="model.full_name" :value="model.full_name">
              {{ model.full_name }}
            </option>
          </select>
        </div>

        <!-- Symbol -->
        <div>
          <label class="block text-sm font-medium mb-1 dark:text-gray-200">Symbol</label>
          <input
            v-model="symbol"
            placeholder="e.g. AAPL"
            class="w-full p-2 border rounded bg-white dark:bg-gray-700 dark:text-gray-200 uppercase"
          />
        </div>

        <!-- Dates -->
        <div>
          <label class="block text-sm font-medium mb-1 dark:text-gray-200">Start Date</label>
          <input
            type="date"
            v-model="startDate"
            class="w-full p-2 border rounded bg-white dark:bg-gray-700 dark:text-gray-200"
          />
        </div>
        <div>
          <label class="block text-sm font-medium mb-1 dark:text-gray-200">End Date</label>
          <input
            type="date"
            v-model="endDate"
            class="w-full p-2 border rounded bg-white dark:bg-gray-700 dark:text-gray-200"
          />
        </div>

        <!-- Starting Balance -->
        <div>
          <label class="block text-sm font-medium mb-1 dark:text-gray-200">Starting Balance ($)</label>
          <input
            type="number"
            v-model.number="startingBalance"
            min="1"
            step="1"
            class="w-full p-2 border rounded bg-white dark:bg-gray-700 dark:text-gray-200"
          />
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
      v-if="simResult"
      class="bg-white dark:bg-gray-800 p-6 rounded-lg shadow relative z-10"
    >
      <h3 class="text-lg font-semibold mb-4 dark:text-gray-100">Simulation Summary</h3>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-2 text-sm dark:text-gray-200">
        <p><strong>Symbol:</strong> {{ simResult.symbol }}</p>
        <p><strong>Model:</strong> {{ selectedModel }}</p>
        <p><strong>Start:</strong> {{ simResult.start }}</p>
        <p><strong>End:</strong> {{ simResult.end }}</p>
        <p><strong>Starting Balance:</strong> ${{ numberFmt(simResult.starting_balance) }}</p>
        <p><strong>Final Balance:</strong> ${{ numberFmt(simResult.final_balance) }}</p>
        <p>
          <strong>P&L:</strong>
          <span :class="pnl >= 0 ? 'text-green-600' : 'text-red-600'">
            ${{ numberFmt(pnl) }} ({{ simResult.pnl_pct.toFixed(2) }}%)
          </span>
        </p>
      </div>

      <!-- Portfolio Chart -->
      <div class="mt-6">
        <h4 class="font-semibold text-sm dark:text-gray-200 mb-2">Portfolio Value Over Time</h4>
        <div class="w-full overflow-hidden border border-gray-200 dark:border-gray-700 rounded">
          <svg :viewBox="`0 0 ${chartWidth} ${chartHeight}`" class="w-full h-48">
            <path
              v-if="portfolioPath"
              :d="portfolioPath"
              fill="none"
              stroke="currentColor"
              class="text-blue-600 dark:text-blue-400"
              stroke-width="2"
            />
          </svg>
        </div>
        <div class="text-xs text-gray-500 dark:text-gray-400 mt-1">
          Min: ${{ numberFmt(minPortfolio) }} &nbsp;|&nbsp; Max: ${{ numberFmt(maxPortfolio) }}
        </div>
      </div>
    </div>

    <!-- Trade Log Table -->
    <div
      v-if="flattenedTrades.length"
      class="bg-white dark:bg-gray-800 p-6 rounded-lg shadow relative z-10 mt-6"
    >
      <div class="flex flex-col md:flex-row md:items-center md:justify-between mb-4 gap-2">
        <h3 class="text-lg font-semibold dark:text-gray-100">Trade Log</h3>
        <div class="flex flex-wrap items-center gap-3">
          <select
            v-model="filterAction"
            class="p-2 border rounded bg-white dark:bg-gray-700 dark:text-gray-200"
          >
            <option value="">All Actions</option>
            <option value="CALL">CALL</option>
            <option value="PUT">PUT</option>
            <option value="HOLD">HOLD</option>
          </select>

          <input
            v-model="searchText"
            type="text"
            placeholder="Search date or strike..."
            class="p-2 border rounded bg-white dark:bg-gray-700 dark:text-gray-200"
          />

          <button
            @click="downloadCSV"
            class="bg-green-600 hover:bg-green-700 text-white text-sm px-3 py-1 rounded"
          >
            Download CSV
          </button>
        </div>
      </div>

      <div class="overflow-x-auto">
        <table class="min-w-full text-xs border-collapse border border-gray-300 dark:border-gray-700">
          <thead class="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-100">
            <tr>
              <th class="px-3 py-2 border">Date</th>
              <th class="px-3 py-2 border">Action</th>
              <th class="px-3 py-2 border">Open ($)</th>
              <th class="px-3 py-2 border">Close ($)</th>
              <th class="px-3 py-2 border">Strike ($)</th>
              <th class="px-3 py-2 border">Opt Open ($)</th>
              <th class="px-3 py-2 border">Opt Close ($)</th>
              <th class="px-3 py-2 border">Contracts</th>
              <th class="px-3 py-2 border">Cost ($)</th>
              <th class="px-3 py-2 border">Proceeds ($)</th>
              <th class="px-3 py-2 border">P&L ($)</th>
              <th class="px-3 py-2 border">P&L (%)</th>
              <th class="px-3 py-2 border">σ</th>
              <th class="px-3 py-2 border">DTE</th>
              <th class="px-3 py-2 border">Portfolio ($)</th>
            </tr>
          </thead>
          <tbody>
            <tr
                v-for="(row, i) in filteredTrades"
                :key="row._key"
                class="border-t border-gray-300 dark:border-gray-700 hover:opacity-90 transition-colors"
                :class="i % 2 === 0 ? 'bg-gray-50 dark:bg-gray-800/40' : 'bg-gray-100 dark:bg-gray-700/40'"
              >
              <td class="px-3 py-1 text-center">{{ row.date }}</td>
              <td class="px-3 py-1 text-center font-semibold">{{ row.action }}</td>
              <td class="px-3 py-1 text-center">{{ numberFmt(row.underlying_open) }}</td>
              <td class="px-3 py-1 text-center">{{ numberFmt(row.underlying_close) }}</td>
              <td class="px-3 py-1 text-center">{{ row.strike ?? '-' }}</td>
              <td class="px-3 py-1 text-center">{{ row.option_open ?? '-' }}</td>
              <td class="px-3 py-1 text-center">{{ row.option_close ?? '-' }}</td>
              <td class="px-3 py-1 text-center">{{ row.contracts }}</td>
              <td class="px-3 py-1 text-center">{{ numberFmt(row.total_cost) }}</td>
              <td class="px-3 py-1 text-center">{{ numberFmt(row.total_proceeds) }}</td>
              <td
                class="px-3 py-1 text-center font-semibold"
                :class="row.pnl >= 0 ? 'text-green-600' : 'text-red-600'"
              >
                {{ numberFmt(row.pnl) }}
              </td>
              <td class="px-3 py-1 text-center">{{ row.pnl_pct?.toFixed?.(2) ?? '-' }}</td>
              <td class="px-3 py-1 text-center">{{ row.volatility?.toFixed?.(3) ?? '-' }}</td>
              <td class="px-3 py-1 text-center">{{ row.dte ?? '-' }}</td>
              <td class="px-3 py-1 text-center font-semibold">{{ numberFmt(row.portfolio_value) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Loading -->
    <transition name="fade">
      <div v-if="isRunning" class="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
        <div class="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg flex items-center space-x-4">
          <svg
            class="animate-spin h-6 w-6 text-blue-600"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
          </svg>
          <span class="font-semibold text-gray-700 dark:text-gray-200">Running Simulation...</span>
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
      symbol: "",
      startDate: "2020-01-01",
      endDate: "2025-01-01",
      startingBalance: 10000,
      simResult: null,
      validationError: "",
      isRunning: false,
      filterAction: "",
      searchText: "",
      chartWidth: 800,
      chartHeight: 200,
      padding: 24,
    };
  },
  created() {
    this.fetchModels();
    this.$nextTick(() => {
      const resize = () => {
        const el = this.$el.querySelector("svg");
        if (el && el.clientWidth) this.chartWidth = Math.max(400, el.clientWidth);
      };
      window.addEventListener("resize", resize);
      resize();
      this._resize = resize;
    });
  },
  beforeUnmount() {
    if (this._resize) window.removeEventListener("resize", this._resize);
  },
  computed: {
    pnl() {
      if (!this.simResult) return 0;
      return this.simResult.final_balance - this.simResult.starting_balance;
    },
    flattenedTrades() {
      return (this.simResult?.trades || []).map((t, i) => ({
        _key: `${t.date}-${i}`,
        ...t,
      }));
    },
    filteredTrades() {
      const text = this.searchText.trim().toLowerCase();
      return this.flattenedTrades.filter((r) => {
        const a = !this.filterAction || r.action === this.filterAction;
        const s =
          !text ||
          String(r.date).toLowerCase().includes(text) ||
          String(r.strike ?? "").toLowerCase().includes(text);
        return a && s;
      });
    },
    minPortfolio() {
      const vals = this.simResult?.portfolio_values || [];
      return vals.length ? Math.min(...vals) : this.startingBalance;
    },
    maxPortfolio() {
      const vals = this.simResult?.portfolio_values || [];
      return vals.length ? Math.max(...vals) : this.startingBalance;
    },
    portfolioPath() {
      const vals = this.simResult?.portfolio_values || [];
      if (!vals.length) return "";
      const w = this.chartWidth,
        h = this.chartHeight,
        pad = this.padding;
      const xScale = (i) => pad + (i * (w - 2 * pad)) / (vals.length - 1);
      const minV = this.minPortfolio,
        maxV = this.maxPortfolio,
        range = maxV - minV || 1;
      const yScale = (v) => h - pad - ((v - minV) / range) * (h - 2 * pad);
      let d = `M ${xScale(0)} ${yScale(vals[0])}`;
      for (let i = 1; i < vals.length; i++) d += ` L ${xScale(i)} ${yScale(vals[i])}`;
      return d;
    },
  },
  methods: {
    numberFmt(n) {
      return Number(n ?? 0).toLocaleString(undefined, { maximumFractionDigits: 2 });
    },
    async fetchModels() {
      const res = await fetch("http://localhost:8001/models");
      this.models = await res.json();
    },
    async runSimulation() {
      this.validationError = "";
      if (!this.selectedModel || !this.symbol) {
        this.validationError = "Please select a model and symbol.";
        return;
      }
      this.isRunning = true;
      this.simResult = null;
      try {
        const url = new URL("http://localhost:8002/simulate");
        url.searchParams.set("symbol", this.symbol.toUpperCase());
        url.searchParams.set("model_name", `${this.selectedModel}.zip`);
        url.searchParams.set("start", this.startDate);
        url.searchParams.set("end", this.endDate);
        url.searchParams.set("starting_balance", this.startingBalance);
        const res = await fetch(url, { method: "POST" });
        if (!res.ok) throw new Error((await res.json()).error || "Simulation failed");
        this.simResult = await res.json();
      } catch (err) {
        this.validationError = err.message;
      } finally {
        this.isRunning = false;
      }
    },
    downloadCSV() {
      if (!this.flattenedTrades.length) return;
      const headers = Object.keys(this.flattenedTrades[0]).filter((h) => !h.startsWith("_"));
      const csv = [
        headers.join(","),
        ...this.flattenedTrades.map((r) => headers.map((h) => r[h] ?? "").join(",")),
      ].join("\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `${this.selectedModel}_${this.symbol}_trades.csv`;
      link.click();
      URL.revokeObjectURL(link.href);
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
tr:hover {
  filter: brightness(0.97);
}
</style>
