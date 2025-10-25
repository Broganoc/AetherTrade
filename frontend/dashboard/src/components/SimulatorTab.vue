<template>
  <div class="p-6 relative min-h-screen">
    <h2 class="text-xl font-semibold mb-4 dark:text-gray-100">Simulation Dashboard</h2>

    <!-- Simulation Form -->
    <div class="bg-gray-100 dark:bg-gray-800 p-6 rounded-lg shadow mb-6 relative z-10">
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
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

        <!-- Symbol Search -->
        <div class="relative">
          <label class="block text-sm font-medium mb-1 dark:text-gray-200">Symbol</label>
          <input
            type="text"
            v-model="symbol"
            @input="filterSymbols"
            @focus="showSuggestions = true"
            @blur="hideSuggestions"
            placeholder="Search or type symbol (e.g. AAPL, SPY, NASDAQ)"
            class="w-full p-2 border rounded bg-white dark:bg-gray-700 dark:text-gray-200 uppercase"
            autocomplete="off"
          />
          <ul
            v-if="showSuggestions && filteredSymbolList.length"
            class="absolute z-50 w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded mt-1 max-h-48 overflow-y-auto shadow-lg"
          >
            <li
              v-for="item in filteredSymbolList"
              :key="item.symbol"
              @mousedown.prevent="selectSymbol(item.symbol)"
              class="px-3 py-2 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 flex justify-between"
            >
              <span class="font-medium">{{ item.symbol }}</span>
              <span class="text-xs text-gray-500 dark:text-gray-400">{{ item.name }}</span>
            </li>
          </ul>
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

        <!-- Portfolio -->
        <div>
          <label class="block text-sm font-medium mb-1 dark:text-gray-200">Starting Portfolio ($)</label>
          <input
            type="number"
            v-model.number="portfolioStart"
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

    <!-- Summary -->
    <div
      v-if="simulatorData"
      class="bg-white dark:bg-gray-800 p-6 rounded-lg shadow mb-6 relative z-10"
    >
      <h3 class="text-lg font-semibold mb-2 dark:text-gray-100">Simulation Summary</h3>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-2 text-sm dark:text-gray-200">
        <p><strong>Model Used:</strong> {{ simulatorData.model_used }}</p>
        <p><strong>Symbol:</strong> {{ simulatorData.symbol }}</p>
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

      <!-- Table -->
      <div class="overflow-x-auto">
        <table class="min-w-full text-sm border-collapse border border-gray-300 dark:border-gray-700">
          <thead class="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-100">
            <tr>
              <th v-for="header in tableHeaders" :key="header" class="px-3 py-2 border border-gray-300 dark:border-gray-700">
                {{ header }}
              </th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="trade in filteredTrades"
              :key="trade.step"
              class="border-t border-gray-300 dark:border-gray-700 hover:opacity-90 transition-colors"
              :class="{
                'bg-green-100 dark:bg-green-900/25': trade.pnl_pct > 0,
                'bg-red-100 dark:bg-red-900/25': trade.pnl_pct < 0
              }"
            >
              <td class="px-3 py-1 text-center">{{ trade.step }}</td>
              <td class="px-3 py-1 text-center">{{ formatDate(trade.date) }}</td>
              <td class="px-3 py-1 text-center">{{ trade.action }}</td>
              <td class="px-3 py-1 text-center">{{ trade.underlying_open ? `$${trade.underlying_open.toFixed(2)}` : "-" }}</td>
              <td class="px-3 py-1 text-center font-semibold">{{ trade.underlying_close ? `$${trade.underlying_close.toFixed(2)}` : "-" }}</td>
              <td class="px-3 py-1 text-center">{{ trade.strike_price ? `$${trade.strike_price.toFixed(2)}` : "-" }}</td>
              <td class="px-3 py-1 text-center">{{ trade.expiration_date || "-" }}</td>
              <td class="px-3 py-1 text-center">{{ trade.purchase_cost ? `$${trade.purchase_cost.toFixed(2)}` : "-" }}</td>
              <td class="px-3 py-1 text-center">{{ trade.contracts || "-" }}</td>
              <td class="px-3 py-1 text-center">{{ trade.total_cost ? `$${trade.total_cost.toLocaleString()}` : "-" }}</td>
              <td class="px-3 py-1 text-center">{{ trade.sale_price_per_contract ? `$${trade.sale_price_per_contract.toFixed(2)}` : "-" }}</td>
              <td class="px-3 py-1 text-center">{{ trade.total_proceeds ? `$${trade.total_proceeds.toLocaleString()}` : "-" }}</td>
              <td class="px-3 py-1 text-center font-semibold">{{ trade.pnl ? `$${trade.pnl.toFixed(2)}` : "-" }}</td>
              <td class="px-3 py-1 text-center font-semibold">{{ trade.pnl_pct ? `${trade.pnl_pct.toFixed(2)}%` : "-" }}</td>
              <td class="px-3 py-1 text-center">${{ trade.portfolio_value.toLocaleString() }}</td>
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
      startDate: "",
      endDate: "",
      portfolioStart: 100000,
      simulatorData: null,
      trades: [],
      validationError: "",
      isRunning: false,
      filterAction: "",
      searchText: "",
      showSuggestions: false,
      filteredSymbolList: [],
      symbolList: [],
      tableHeaders: [
        "Step", "Date", "Action", "Underlying Open ($)", "Underlying Close ($)",
        "Strike ($)", "Expiration", "Cost ($)", "Contracts", "Total Cost ($)",
        "Sale Price ($)", "Total Proceeds ($)", "P&L ($)", "P&L (%)", "Portfolio ($)"
      ]
    };
  },
  computed: {
    filteredTrades() {
      return this.trades.filter((t) => {
        const matchesAction = !this.filterAction || t.action === this.filterAction;
        const text = this.searchText.toLowerCase();
        const matchesSearch =
          !text ||
          t.date?.toLowerCase().includes(text) ||
          String(t.strike_price || "").toLowerCase().includes(text);
        return matchesAction && matchesSearch;
      });
    },
  },
  created() {
    this.fetchModels();
    this.loadSymbols();
  },
  methods: {
    async loadSymbols() {
      try {
        const res = await fetch("/src/data/symbols.json");
        if (!res.ok) throw new Error("Failed to load symbols.json");
        this.symbolList = await res.json();
        this.filteredSymbolList = this.symbolList.slice(0, 8);
      } catch (err) {
        console.warn("Falling back to default symbols:", err);
        this.symbolList = [
          { symbol: "AAPL", name: "Apple Inc." },
          { symbol: "MSFT", name: "Microsoft Corp." },
          { symbol: "SPY", name: "S&P 500 ETF" },
          { symbol: "^IXIC", name: "NASDAQ Composite Index" },
          { symbol: "^DJI", name: "Dow Jones Industrial Average" }
        ];
      }
    },
    filterSymbols() {
      const q = this.symbol.trim().toUpperCase();
      this.filteredSymbolList = this.symbolList.filter(
        (s) => s.symbol.includes(q) || s.name.toUpperCase().includes(q)
      ).slice(0, 10);
    },
    selectSymbol(sym) {
      this.symbol = sym.toUpperCase();
      this.showSuggestions = false;
    },
    hideSuggestions() {
      setTimeout(() => (this.showSuggestions = false), 150);
    },
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
      this.isRunning = true;
      try {
        const res = await fetch("http://localhost:8002/run-sim", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_used: this.selectedModel,
            start_date: this.startDate,
            end_date: this.endDate,
            portfolio_start: this.portfolioStart,
            symbol: this.symbol
          })
        });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || "Simulation failed");
        }
        const data = await res.json();
        this.simulatorData = data.summary;
        this.trades = data.trades || [];
      } catch (err) {
        console.error("Simulation failed:", err);
        this.validationError = err.message || "Simulation failed. Check backend logs.";
      } finally {
        this.isRunning = false;
      }
    },
    formatDate(d) {
      const date = new Date(d);
      if (isNaN(date)) return d;
      return date.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
    },
    downloadCSV() {
      if (!this.trades.length) return;
      const headers = Object.keys(this.trades[0]);
      const csvRows = [
        headers.join(","),
        ...this.trades.map((t) => headers.map((h) => JSON.stringify(t[h] ?? "")).join(","))
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
    }
  }
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
  filter: brightness(0.95);
}
</style>
