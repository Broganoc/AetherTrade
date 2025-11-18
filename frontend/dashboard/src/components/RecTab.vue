<script setup>
import { ref, onMounted, computed } from "vue";

const models = ref([]);
const selectedModel = ref("");
const predictions = ref([]);
const enhanced = ref([]);
const loading = ref(false);

const stats = ref({});
const backtestInfo = ref({ processedCount: 0, skippedCount: 0 });

// ------------------------------------------
// Helpers
// ------------------------------------------
function formatModelLabel(model) {
  return model.endsWith(".zip") ? model.slice(0, -4) : model;
}

function shortModelLabel(m) {
  // remove .zip
  const base = m.replace(".zip", "");

  // Format: "ppo_agent_v1_TSLA" → ["ppo","agent","v1","TSLA"]
  const parts = base.split("_");
  let ticker = parts[parts.length - 1];
  if(ticker == 'best'){
    ticker = parts[parts.length - 2] + " " + parts[parts.length - 1];
  }
  else{
    ticker = parts[parts.length - 1];
  }
  return `PPO / ${ticker}`;
}

const smartSortColumn = ref("combined_score");
const smartSortDir = ref(-1);

const enhancedSorted = computed(() => {
  return [...enhanced.value].sort((a, b) => {
    const x = a[smartSortColumn.value];
    const y = b[smartSortColumn.value];
    if (x === y) return 0;
    return (x > y ? 1 : -1) * smartSortDir.value;
  });
});


// ------------------------------------------
// Fetch available models
// ------------------------------------------
async function loadModels() {
  const res = await fetch("http://localhost:8003/models");
  models.value = await res.json();
}

// ------------------------------------------
// Load stats.json
// ------------------------------------------
async function loadStats() {
  try {
    const res = await fetch("http://localhost:8003/stats");
    const data = await res.json();
    stats.value = data || {};
  } catch (e) {
    stats.value = {};
  }
}

// ------------------------------------------
// Run full backtest over all prediction files or update backtest on only new files
// ------------------------------------------
async function runFullBacktest() {
  loading.value = true;
  try {
    const res = await fetch("http://localhost:8003/backtest-all", {
      method: "POST",
    });
    const data = await res.json();

    if (data && data.stats) {
      stats.value = data.stats;
      backtestInfo.value = {
        processedCount: (data.processed && data.processed.length) || 0,
        skippedCount: (data.skipped && data.skipped.length) || 0,
      };
    }
  } catch (e) {
    console.error("Full backtest failed:", e);
  } finally {
    loading.value = false;
  }
}

async function runBacktestUpdate() {
  loading.value = true;
  try {
    const res = await fetch("http://localhost:8003/backtest-update", {
      method: "POST",
    });
    const data = await res.json();

    if (data && data.stats) {
      stats.value = data.stats;
      backtestInfo.value = {
        processedCount: (data.processed && data.processed.length) || 0,
        skippedCount: (data.skipped && data.skipped.length) || 0,
      };
    }
  } catch (e) {
    console.error("Incremental backtest failed:", e);
  } finally {
    loading.value = false;
  }
}


// ------------------------------------------
// Run predictions with selected model
// ------------------------------------------
async function runPredictions() {
  if (!selectedModel.value) {
    alert("Please select a model first.");
    return;
  }
  loading.value = true;

  try {
    const url = new URL("http://localhost:8003/predict");
    url.searchParams.set("model_name", selectedModel.value);

    const res = await fetch(url, { method: "POST" });
    const data = await res.json();

    predictions.value = data.predictions || [];
    enhanced.value = data.enhanced_rankings || [];
  } catch (e) {
    console.error("Prediction error:", e);
    predictions.value = [];
    enhanced.value = [];
  } finally {
    loading.value = false;
  }
}

// ------------------------------------------
// Computed: All models from stats
// ------------------------------------------
const allModels = computed(() => {
  const s = stats.value || {};
  if (Array.isArray(s._all_models)) return s._all_models;

  const set = new Set();
  Object.entries(s).forEach(([key, val]) => {
    if (key === "_all_models" || !val || !val.model_accuracy) return;
    Object.keys(val.model_accuracy).forEach((m) => set.add(m));
  });
  return Array.from(set);
});

// ------------------------------------------
// Computed: Model summary cards
// ------------------------------------------
const modelCards = computed(() => {
  const s = stats.value || {};
  const modelsList = allModels.value;
  if (!modelsList.length) return [];

  const cards = [];

  modelsList.forEach((model) => {
    let accSum = 0;
    let accCount = 0;
    let pnlSum = 0;
    let pnlCount = 0;

    let bestAcc = -1;
    let bestAccSymbol = null;

    let bestPnl = -1e9;
    let bestPnlSymbol = null;

    Object.entries(s).forEach(([symbol, data]) => {
      if (symbol === "_all_models") return;
      if (!data || !data.model_accuracy) return;

      const acc = data.model_accuracy[model];
      const pnl = data.model_pnl ? data.model_pnl[model] : null;

      if (acc != null) {
        accSum += acc;
        accCount += 1;
        if (acc > bestAcc) {
          bestAcc = acc;
          bestAccSymbol = symbol;
        }
      }

      if (pnl != null) {
        pnlSum += pnl;
        pnlCount += 1;
        if (pnl > bestPnl) {
          bestPnl = pnl;
          bestPnlSymbol = symbol;
        }
      }
    });

    if (accCount === 0 && pnlCount === 0) return;

    cards.push({
      model,
      displayName: formatModelLabel(model),
      overallAccuracy: accCount ? accSum / accCount : null,
      overallPnl: pnlCount ? pnlSum / pnlCount : null,
      bestAccSymbol,
      bestAcc,
      bestPnlSymbol,
      bestPnl,
    });
  });

  return cards;
});

// ------------------------------------------
// Computed: Table rows for stock-by-stock stats
// ------------------------------------------
const sortColumn = ref(null);
const sortDir = ref(1); // 1 = asc, -1 = desc

const tableRows = computed(() => {
  const s = stats.value || {};
  const modelsList = allModels.value;
  const rows = [];

  Object.entries(s).forEach(([symbol, data]) => {
    if (symbol === "_all_models") return;
    if (!data) return;

    const row = {
      symbol,
      overall_acc: data.overall_accuracy ?? null,
      overall_pnl: data.overall_pnl ?? null,
    };

    modelsList.forEach((m) => {
      const acc = data.model_accuracy ? data.model_accuracy[m] : null;
      const pnl = data.model_pnl ? data.model_pnl[m] : null;
      row[`acc_${m}`] = acc ?? null;
      row[`pnl_${m}`] = pnl ?? null;
    });

    rows.push(row);
  });

  if (!sortColumn.value) return rows;

  const key = sortColumn.value;
  const dir = sortDir.value;

  return [...rows].sort((a, b) => {
    const x = a[key];
    const y = b[key];

    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;

    if (x === y) return 0;
    return (x > y ? 1 : -1) * dir;
  });
});

function sortBy(colKey) {
  if (sortColumn.value === colKey) {
    sortDir.value = -sortDir.value;
  } else {
    sortColumn.value = colKey;
    sortDir.value = 1;
  }
}

// Column definitions for table headers
const accuracyColumns = computed(() =>
  allModels.value.map((m) => ({
    key: `acc_${m}`,
    label: `${shortModelLabel(m)} Acc`,
  }))
);

const pnlColumns = computed(() =>
  allModels.value.map((m) => ({
    key: `pnl_${m}`,
    label: `${shortModelLabel(m)} P&L %`,
  }))
);

// ------------------------------------------
// Init
// ------------------------------------------
onMounted(() => {
  loadModels();
  (async () => {
    await loadStats();
  })();
});
</script>

<template>
  <div class="p-6 pb-24">
    <!-- Page Title -->
    <h2 class="text-2xl font-semibold text-center mb-6 dark:text-gray-100">
      Prediction Data
    </h2>

    <!-- Model Summary Cards -->
    <div
      v-if="modelCards.length"
      class="grid gap-4 mb-8 md:grid-cols-2 xl:grid-cols-3"
    >
      <div
        v-for="card in modelCards"
        :key="card.model"
        class="bg-gray-800 rounded-xl p-4 shadow border border-gray-700"
      >
        <div class="flex items-center justify-between mb-2">
          <h3 class="font-semibold text-lg text-gray-100">
            {{ card.displayName }}
          </h3>
          <span class="text-xs text-gray-400 break-all">
            {{ card.model }}
          </span>
        </div>

        <p class="text-sm text-gray-300">
          Overall Accuracy:
          <span v-if="card.overallAccuracy != null" class="font-semibold">
            {{ (card.overallAccuracy * 100).toFixed(2) }}%
          </span>
          <span v-else>—</span>
        </p>

        <p class="text-sm text-gray-300">
          Overall PnL:
          <span v-if="card.overallPnl != null" class="font-semibold">
            {{ (card.overallPnl * 100).toFixed(2) }}%
          </span>
          <span v-else>—</span>
        </p>

        <p class="text-xs text-gray-400 mt-3">
          Best Stock (Accuracy):
          <span v-if="card.bestAccSymbol">
            {{ card.bestAccSymbol }}
            <span v-if="card.bestAcc != null">
              ({{ (card.bestAcc * 100).toFixed(2) }}%)
            </span>
          </span>
          <span v-else>—</span>
        </p>

        <p class="text-xs text-gray-400">
          Best Stock (PnL):
          <span v-if="card.bestPnlSymbol">
            {{ card.bestPnlSymbol }}
            <span v-if="card.bestPnl != null">
              ({{ (card.bestPnl * 100).toFixed(2) }}%)
            </span>
          </span>
          <span v-else>—</span>
        </p>
      </div>
    </div>

    <!-- Stock Prediction Backtest Section -->
    <div class="flex items-center justify-between mb-2">
      <h3 class="text-lg font-semibold dark:text-gray-100">
        Stock Prediction Backtest
      </h3>

      <div class="flex gap-2">
        <button
          @click="runBacktestUpdate"
          :disabled="loading"
          class="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 disabled:opacity-50 text-sm"
        >
          {{ loading ? "Updating..." : "Update Backtest" }}
        </button>

        <button
          @click="runFullBacktest"
          :disabled="loading"
          class="bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700 disabled:opacity-50 text-sm"
        >
          {{ loading ? "Recomputing..." : "Recompute Backtest" }}
        </button>
      </div>
    </div>


    <p class="text-xs text-gray-400 mb-4">
      Processed {{ backtestInfo.processedCount }} file(s), skipped
      {{ backtestInfo.skippedCount }} (future or invalid dates).
    </p>

    <!-- Stock-by-Stock Accuracy & PnL Table -->
    <div v-if="tableRows.length" class="overflow-x-auto mb-10">
      <table class="min-w-full text-sm text-gray-200 bg-gray-800 rounded">
        <thead>
          <tr class="border-b border-gray-700">
            <!-- Symbol -->
            <th class="text-left py-2 px-2 cursor-pointer" @click="sortBy('symbol')">
              Symbol
              <span v-if="sortColumn === 'symbol'">{{ sortDir === 1 ? '▲' : '▼' }}</span>
            </th>

            <!-- Overall Accuracy -->
            <th class="text-left py-2 px-2 cursor-pointer whitespace-nowrap" @click="sortBy('overall_acc')">
              Overall Acc %
              <span v-if="sortColumn === 'overall_acc'">{{ sortDir === 1 ? '▲' : '▼' }}</span>
            </th>

            <!-- Overall P&L -->
            <th class="text-left py-2 px-2 cursor-pointer whitespace-nowrap" @click="sortBy('overall_pnl')">
              Overall P&L %
              <span v-if="sortColumn === 'overall_pnl'">{{ sortDir === 1 ? '▲' : '▼' }}</span>
            </th>

            <!-- Per-model Accuracy -->
            <th
              v-for="col in accuracyColumns"
              :key="col.key"
              class="text-left py-2 px-2 cursor-pointer whitespace-nowrap"
              @click="sortBy(col.key)"
            >
              {{ col.label }}
              <span v-if="sortColumn === col.key">{{ sortDir === 1 ? '▲' : '▼' }}</span>
            </th>

            <!-- Per-model P&L -->
            <th
              v-for="col in pnlColumns"
              :key="col.key"
              class="text-left py-2 px-2 cursor-pointer whitespace-nowrap"
              @click="sortBy(col.key)"
            >
              {{ col.label }}
              <span v-if="sortColumn === col.key">{{ sortDir === 1 ? '▲' : '▼' }}</span>
            </th>

          </tr>
        </thead>

        <tbody>
          <tr
            v-for="row in tableRows"
            :key="row.symbol"
            class="border-b border-gray-700"
          >
            <!-- Symbol -->
            <td class="py-2 px-2 font-semibold">{{ row.symbol }}</td>

            <!-- Overall Accuracy -->
            <td class="py-2 px-2">
              <span v-if="row.overall_acc != null">
                {{ (row.overall_acc * 100).toFixed(2) }}%
              </span>
              <span v-else>—</span>
            </td>

            <!-- Overall P&L -->
            <td class="py-2 px-2">
              <span v-if="row.overall_pnl != null">
                {{ (row.overall_pnl * 100).toFixed(2) }}%
              </span>
              <span v-else>—</span>
            </td>

            <!-- Per-model Accuracies -->
            <td
              v-for="col in accuracyColumns"
              :key="col.key"
              class="py-2 px-2"
            >
              <span v-if="row[col.key] != null">
                {{ (row[col.key] * 100).toFixed(2) }}%
              </span>
              <span v-else>—</span>
            </td>

            <!-- Per-model P&L -->
            <td
              v-for="col in pnlColumns"
              :key="col.key"
              class="py-2 px-2"
            >
              <span v-if="row[col.key] != null">
                {{ (row[col.key] * 100).toFixed(2) }}%
              </span>
              <span v-else>—</span>
            </td>
          </tr>
        </tbody>

      </table>
    </div>

    <h3 class="text-lg font-semibold mt-6 dark:text-gray-100">
      Top 25 SmartRank Predictions
    </h3>

    <table v-if="enhanced.length" class="w-full text-sm mt-3 bg-gray-800 rounded">
      <thead>
        <tr class="border-b border-gray-700">
          <th class="py-2 px-2">Symbol</th>
          <th class="py-2 px-2">Action</th>
          <th class="py-2 px-2">Confidence</th>
          <th class="py-2 px-2">Hist. Acc</th>
          <th class="py-2 px-2">Hist. PnL</th>
          <th class="py-2 px-2">Recent Acc</th>
          <th class="py-2 px-2">Recent PnL</th>
          <th class="py-2 px-2 font-semibold">SmartRank</th>
        </tr>
      </thead>

      <tbody>
        <tr
          v-for="r in enhanced"
          :key="r.symbol"
          class="border-b border-gray-700"
        >
          <td class="py-1 px-2">{{ r.symbol }}</td>
          <td class="py-1 px-2">{{ r.action }}</td>
          <td class="py-1 px-2">{{ r.confidence.toFixed(2) }}%</td>

          <!-- Historical accuracy -->
          <td class="py-1 px-2">
            {{ r.historical_accuracy?.toFixed(2) ?? '—' }}%
          </td>

          <!-- Historical PnL -->
          <td class="py-1 px-2">
            {{ r.historical_pnl?.toFixed(2) ?? '—' }}%
          </td>

          <!-- Recent accuracy -->
          <td class="py-1 px-2">
            {{ r.recent_accuracy?.toFixed(2) ?? '—' }}%
          </td>

          <!-- Recent PnL -->
          <td class="py-1 px-2">
            {{ r.recent_pnl?.toFixed(2) ?? '—' }}%
          </td>

          <!-- Combined score -->
          <td class="py-1 px-2 font-semibold">
            {{ r.combined_score.toFixed(4) }}
          </td>
        </tr>
      </tbody>
    </table>



        <!-- Model Recommendations Results -->
    <h3 class="text-lg font-semibold mb-2 dark:text-gray-100">
      Model Recommendations
    </h3>

    <div>
      <table
        v-if="predictions.length"
        class="w-full text-sm text-gray-200 mt-4 bg-gray-800 rounded"
      >
        <thead>
          <tr class="border-b border-gray-600">
            <th class="text-left py-2 px-2">Symbol</th>
            <th class="text-left py-2 px-2">Action</th>
            <th class="text-left py-2 px-2">Confidence</th>
            <th class="text-left py-2 px-2">Price</th>
            <th class="text-left py-2 px-2">Date</th>
          </tr>
        </thead>

        <tbody>
          <tr
            v-for="(p, idx) in predictions"
            :key="idx"
            class="border-b border-gray-700"
          >
            <td class="py-1 px-2">{{ p.symbol }}</td>
            <td class="py-1 px-2">{{ p.action }}</td>
            <td class="py-1 px-2">
              {{ p.confidence != null ? p.confidence.toFixed(2) + '%' : '—' }}
            </td>
            <td class="py-1 px-2">${{ p.price?.toFixed(2) }}</td>
            <td class="py-1 px-2">{{ p.date }}</td>
          </tr>
        </tbody>
      </table>

      <p
        v-else
        class="text-center text-gray-400 mt-4"
      >
        No predictions yet. Run a prediction above.
      </p>
    </div>

    <!-- Sticky Bottom Overlay -->
    <div
      class="fixed left-0 right-0 bottom-0 bg-gray-900 border-t border-gray-700 p-4 flex flex-col md:flex-row items-center gap-4 z-30"
    >
      <div class="flex-1 flex items-center gap-3 w-full md:w-auto">
        <select
          v-model="selectedModel"
          class="bg-gray-800 border border-gray-600 rounded p-2 text-gray-100 w-full md:w-64"
        >
          <option value="">Select Model</option>
          <option
            v-for="m in models"
            :key="m.full_name"
            :value="m.full_name"
          >
            {{ m.full_name }}
          </option>
        </select>
      </div>

      <button
        @click="runPredictions"
        :disabled="loading"
        class="bg-blue-600 text-white px-6 py-2 rounded shadow hover:bg-blue-700 disabled:opacity-50 w-full md:w-auto"
      >
        {{ loading ? 'Running...' : 'Run Predictions' }}
      </button>
    </div>

  </div>
</template>


<style scoped>
table {
  table-layout: auto;
}

table th,
table td {
  white-space: nowrap;
  padding-right: 12px;
}

@media (max-width: 1200px) {
  table {
    font-size: 0.85rem;
  }
}
</style>


