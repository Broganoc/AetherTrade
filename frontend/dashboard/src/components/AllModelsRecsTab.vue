<script setup>
import { ref, onMounted, computed } from "vue";

/* ----------------------------------------------------
   Sorting State
---------------------------------------------------- */
const sortColumn = ref("smartRank");
const sortDir = ref(-1);

function sortBy(col) {
  if (sortColumn.value === col) {
    sortDir.value = -sortDir.value;
  } else {
    sortColumn.value = col;
    sortDir.value = -1;
  }
}

/* ----------------------------------------------------
   Modal State
---------------------------------------------------- */
const selectedRow = ref(null);
const showModal = ref(false);

function openDetails(row) {
  selectedRow.value = row;
  showModal.value = true;
}

function closeModal() {
  showModal.value = false;
}

/* ----------------------------------------------------
   Helpers
---------------------------------------------------- */

// Strike step logic
function calcStrike(price, action) {
  if (!price) return null;
  let step = 1;
  if (price < 25) step = 0.5;
  else if (price < 200) step = 2.5;
  else step = 5;

  if (action === "CALL") return Math.ceil(price / step) * step;
  if (action === "PUT") return Math.floor(price / step) * step;
  return Math.round(price / step) * step;
}

// Date helpers
function closestFriday(date) {
  const d = new Date(date);
  const day = d.getDay();
  const diff = 5 - day;
  const base = new Date(d);
  base.setDate(d.getDate() + diff);

  const prev = new Date(base);
  prev.setDate(prev.getDate() - 7);

  const next = new Date(base);
  next.setDate(next.getDate() + 7);

  const arr = [prev, base, next];
  let best = base;
  let bestDiff = Math.abs(base - d);

  for (const x of arr) {
    const v = Math.abs(x - d);
    if (v < bestDiff) {
      best = x;
      bestDiff = v;
    }
  }
  return best;
}

function computeDTE(dateStr) {
  const trade = new Date(dateStr + "T00:00:00");
  const target = new Date(trade);
  target.setDate(target.getDate() + 31);
  const expiry = closestFriday(target);

  const days = Math.round(
    (expiry - trade) / (24 * 60 * 60 * 1000)
  );

  return {
    dte: days,
    expiryDate: expiry.toISOString().slice(0, 10),
  };
}

/* ----------------------------------------------------
   Option C Stop-Loss / Take-Profit (Percentile + Win Weight)
---------------------------------------------------- */
function percentile(arr, p) {
  if (!arr.length) return null;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.floor(p * (sorted.length - 1));
  return sorted[idx];
}

function computeTradeHints(symbol) {
  const sym = stats.value[symbol];
  if (!sym || !sym.entries) {
    return { stopLoss: null, takeProfit: null };
  }

  const pnls = sym.entries
    .map(e => Number(e.pnl_pct))
    .filter(v => !isNaN(v));

  if (!pnls.length) return { stopLoss: null, takeProfit: null };

  // Percentile-based thresholds
  const sl_raw = percentile(pnls, 0.20);
  const tp_raw = percentile(pnls, 0.80);

  // Weight wins at +75%
  const tp_boosted = tp_raw * 1.75;

  return {
    stopLoss: sl_raw != null ? (sl_raw * 100).toFixed(2) : null,
    takeProfit: tp_boosted != null ? (tp_boosted * 100).toFixed(2) : null,
  };
}

/* ----------------------------------------------------
   API + State
---------------------------------------------------- */
const API = "http://localhost:8003";

const models = ref([]);
const allRows = ref([]);
const stats = ref({});
const loading = ref(false);
const error = ref("");

const targetDate = ref(getTomorrowISO());

function getTomorrowISO() {
  const d = new Date();
  d.setDate(d.getDate() + 1);
  return d.toISOString().slice(0, 10);
}

/* ----------------------------------------------------
   Load models & stats
---------------------------------------------------- */
async function loadModels() {
  try {
    const res = await fetch(`${API}/models`);
    const data = await res.json();
    models.value = data.map((m) =>
      typeof m === "string" ? m : m.full_name
    );
  } catch (err) {
    console.error(err);
    models.value = [];
  }
}

async function loadStats() {
  try {
    const res = await fetch(`${API}/stats`);
    stats.value = await res.json();
  } catch (err) {
    console.error("Failed to load stats:", err);
    stats.value = {};
  }
}

/* ----------------------------------------------------
   Run batch_predict for ALL models
---------------------------------------------------- */
async function runAll() {
  if (!models.value.length) {
    error.value = "No models found";
    return;
  }

  loading.value = true;
  error.value = "";
  allRows.value = [];

  try {
    const tDate = targetDate.value;

    const tasks = models.value.map(async (modelName) => {
      const res = await fetch(`${API}/batch_predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: modelName, target_date: tDate }),
      });

      if (!res.ok) return [];

      const json = await res.json();
      const preds = json.predictions || [];

      return preds.map((p, idx) => {
        const price = p.price ?? null;
        const action = (p.action || "HOLD").toUpperCase();
        const confidence = p.confidence ?? 0;

        const { dte, expiryDate } = computeDTE(p.date || tDate);
        const strike = calcStrike(price, action);

        return {
          id: `${modelName}-${p.symbol}-${idx}`,
          model: modelName,
          symbol: p.symbol,
          action,
          confidence,
          price,
          date: p.date || tDate,
          strike,
          dte,
          expiryDate,
        };
      });
    });

    allRows.value = (await Promise.all(tasks)).flat();
  } catch (err) {
    console.error(err);
    error.value = "Failed to run predictions.";
  } finally {
    loading.value = false;
  }
}

/* ----------------------------------------------------
   Consensus Aggregation
---------------------------------------------------- */
const aggregated = computed(() => {
  const grouped = {};

  for (const r of allRows.value) {
    if (!grouped[r.symbol]) {
      grouped[r.symbol] = {
        symbol: r.symbol,
        actions: [],
        confidences: [],
        prices: [],
        dates: [],
        perModel: [],
      };
    }
    grouped[r.symbol].actions.push(r.action);
    grouped[r.symbol].confidences.push(Number(r.confidence));
    grouped[r.symbol].prices.push(Number(r.price));
    grouped[r.symbol].dates.push(r.date);

    grouped[r.symbol].perModel.push({
      model: r.model,
      action: r.action,
      confidence: Number(r.confidence),
    });
  }

  const rows = [];

  for (const symbol in grouped) {
    const g = grouped[symbol];

    // Majority Action
    const counts = {
      CALL: g.actions.filter((a) => a === "CALL").length,
      PUT: g.actions.filter((a) => a === "PUT").length,
      HOLD: g.actions.filter((a) => a === "HOLD").length,
    };

    const total = g.actions.length;
    const maxCount = Math.max(counts.CALL, counts.PUT, counts.HOLD);

    const majorityAction =
      maxCount === counts.CALL
        ? "CALL"
        : maxCount === counts.PUT
        ? "PUT"
        : "HOLD";

    const majorityRatio = maxCount / total;

    // Weighted Action
    let wCALL = 0,
      wPUT = 0,
      wHOLD = 0;

    for (const rec of g.perModel) {
      if (rec.action === "CALL") wCALL += rec.confidence;
      else if (rec.action === "PUT") wPUT += rec.confidence;
      else wHOLD += rec.confidence;
    }

    const weightedAction =
      wCALL >= wPUT && wCALL >= wHOLD
        ? "CALL"
        : wPUT >= wCALL && wPUT >= wHOLD
        ? "PUT"
        : "HOLD";

    // Avg Confidence
    const avgConfidence =
      g.confidences.reduce((a, b) => a + b, 0) / g.confidences.length;

    // Accuracy
    const symStats = stats.value[symbol] || {};
    const rawAcc = symStats.model_accuracy || {};

    const perModelAcc = {};
    for (const [model, acc] of Object.entries(rawAcc)) {
      perModelAcc[model.replace(".zip", "")] = acc;
    }

    let bestModel = null;
    let bestAcc = -1;
    for (const [model, acc] of Object.entries(perModelAcc)) {
      if (acc > bestAcc) {
        bestAcc = acc;
        bestModel = model;
      }
    }

    const avgAcc = Object.values(perModelAcc).length
      ? Object.values(perModelAcc).reduce((a, b) => a + b, 0) /
        Object.values(perModelAcc).length
      : null;

    // SmartRank
    const wTotal = wCALL + wPUT + wHOLD || 1;
    const wNorm =
      weightedAction === "CALL"
        ? wCALL / wTotal
        : weightedAction === "PUT"
        ? wPUT / wTotal
        : wHOLD / wTotal;

    const pnlNorm = ((symStats.overall_pnl ?? 0) + 1) / 2;

    const smartRank =
      0.4 * wNorm +
      0.2 * (avgConfidence / 100) +
      0.2 * (avgAcc ?? 0.5) +
      0.2 * pnlNorm;

    // Final
    const price = g.prices[g.prices.length - 1];
    const date = g.dates[g.dates.length - 1];
    const strike = calcStrike(price, majorityAction);
    const { dte, expiryDate } = computeDTE(date);

    rows.push({
      symbol,
      majorityAction,
      majorityRatio,
      weightedAction,
      avgConfidence,
      bestModel,
      bestAcc,
      avgAcc,
      smartRank,
      price,
      strike,
      dte,
      expiryDate,
      date,
    });
  }

  return rows.sort((a, b) => {
    const col = sortColumn.value;
    const dir = sortDir.value;
    const x = a[col];
    const y = b[col];

    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;

    return (x > y ? 1 : x < y ? -1 : 0) * dir;
  });
});

/* ----------------------------------------------------
   Init
---------------------------------------------------- */
onMounted(() => {
  loadModels();
  loadStats();
});
</script>

<template>
  <div class="p-6 pb-24">
    <h2 class="text-2xl font-semibold text-center mb-6 dark:text-gray-100">
      All Models – Consensus Signal Table
    </h2>

    <!-- Controls -->
    <div class="flex items-center gap-4 mb-6">
      <div>
        <label class="text-sm dark:text-gray-300">Target Date</label>
        <input
          type="date"
          v-model="targetDate"
          class="bg-gray-800 border border-gray-600 rounded p-2 text-gray-100"
        />
      </div>

      <button
        @click="runAll"
        :disabled="loading"
        class="bg-blue-600 text-white px-5 py-2 rounded shadow hover:bg-blue-700 disabled:opacity-40"
      >
        {{ loading ? "Running..." : "Run All Models" }}
      </button>
    </div>

    <p v-if="error" class="text-red-400 mb-4">{{ error }}</p>

    <!-- TABLE -->
    <table
      v-if="aggregated.length"
      class="w-full text-sm text-gray-200 bg-gray-800 rounded"
    >
      <thead>
        <tr class="border-b border-gray-700">
          <th class="py-2 px-2 cursor-pointer" @click="sortBy('symbol')">
            Symbol
            <span v-if="sortColumn === 'symbol'">{{ sortDir === 1 ? "▲" : "▼" }}</span>
          </th>

          <th class="py-2 px-2 cursor-pointer" @click="sortBy('majorityAction')">
            Majority
            <span v-if="sortColumn === 'majorityAction'">{{ sortDir === 1 ? "▲" : "▼" }}</span>
          </th>

          <th class="py-2 px-2 cursor-pointer" @click="sortBy('weightedAction')">
            Weighted
            <span v-if="sortColumn === 'weightedAction'">{{ sortDir === 1 ? "▲" : "▼" }}</span>
          </th>

          <th class="py-2 px-2 cursor-pointer" @click="sortBy('majorityRatio')">
            Majority %
            <span v-if="sortColumn === 'majorityRatio'">{{ sortDir === 1 ? "▲" : "▼" }}</span>
          </th>

          <th class="py-2 px-2 cursor-pointer" @click="sortBy('avgConfidence')">
            Avg Conf
            <span v-if="sortColumn === 'avgConfidence'">{{ sortDir === 1 ? "▲" : "▼" }}</span>
          </th>

          <th class="py-2 px-2 cursor-pointer" @click="sortBy('bestModel')">
            Best Model
            <span v-if="sortColumn === 'bestModel'">{{ sortDir === 1 ? "▲" : "▼" }}</span>
          </th>

          <th class="py-2 px-2 cursor-pointer" @click="sortBy('bestAcc')">
            Best Acc
            <span v-if="sortColumn === 'bestAcc'">{{ sortDir === 1 ? "▲" : "▼" }}</span>
          </th>

          <th class="py-2 px-2 cursor-pointer" @click="sortBy('avgAcc')">
            Avg Acc
            <span v-if="sortColumn === 'avgAcc'">{{ sortDir === 1 ? "▲" : "▼" }}</span>
          </th>

          <th class="py-2 px-2 cursor-pointer" @click="sortBy('smartRank')">
            SmartRank
            <span v-if="sortColumn === 'smartRank'">{{ sortDir === 1 ? "▲" : "▼" }}</span>
          </th>

          <th class="py-2 px-2 cursor-pointer" @click="sortBy('price')">
            Price
            <span v-if="sortColumn === 'price'">{{ sortDir === 1 ? "▲" : "▼" }}</span>
          </th>

          <th class="py-2 px-2 cursor-pointer" @click="sortBy('strike')">
            Strike
            <span v-if="sortColumn === 'strike'">{{ sortDir === 1 ? "▲" : "▼" }}</span>
          </th>

          <th class="py-2 px-2 cursor-pointer" @click="sortBy('dte')">
            DTE
            <span v-if="sortColumn === 'dte'">{{ sortDir === 1 ? "▲" : "▼" }}</span>
          </th>

          <th class="py-2 px-2 cursor-pointer" @click="sortBy('expiryDate')">
            Expiry
            <span v-if="sortColumn === 'expiryDate'">{{ sortDir === 1 ? "▲" : "▼" }}</span>
          </th>
        </tr>
      </thead>

      <tbody>
        <tr
          v-for="row in aggregated"
          :key="row.symbol"
          @click="openDetails(row)"
          class="border-b border-gray-700 cursor-pointer hover:bg-gray-700"
        >
          <td class="py-1 px-2 font-semibold">{{ row.symbol }}</td>
          <td class="py-1 px-2">{{ row.majorityAction }}</td>
          <td class="py-1 px-2">{{ row.weightedAction }}</td>
          <td class="py-1 px-2">{{ (row.majorityRatio * 100).toFixed(1) }}%</td>
          <td class="py-1 px-2">{{ row.avgConfidence.toFixed(2) }}%</td>

          <td class="py-1 px-2">{{ row.bestModel }}</td>

          <td class="py-1 px-2">
            <span v-if="row.bestAcc != null">{{ (row.bestAcc * 100).toFixed(2) }}%</span>
            <span v-else>—</span>
          </td>

          <td class="py-1 px-2">
            <span v-if="row.avgAcc != null">{{ (row.avgAcc * 100).toFixed(2) }}%</span>
            <span v-else>—</span>
          </td>

          <td class="py-1 px-2 font-semibold">{{ row.smartRank.toFixed(4) }}</td>

          <td class="py-1 px-2">${{ row.price.toFixed(2) }}</td>
          <td class="py-1 px-2">${{ row.strike.toFixed(2) }}</td>
          <td class="py-1 px-2">{{ row.dte }}</td>
          <td class="py-1 px-2">{{ row.expiryDate }}</td>
        </tr>
      </tbody>
    </table>

    <p v-else class="text-gray-400 mt-6 text-center">
      No results yet. Run predictions above.
    </p>

    <!-- ✨ MODAL -->
    <div
      v-if="showModal"
      class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
    >
      <div
        class="bg-gray-900 text-gray-100 p-6 rounded-lg w-full max-w-md shadow-xl border border-gray-700"
      >
        <h3 class="text-xl font-semibold mb-4 text-center">
          {{ selectedRow.symbol }} — Trade Details
        </h3>

        <div class="space-y-2 text-sm">
          <p><strong>Majority Action:</strong> {{ selectedRow.majorityAction }}</p>

          <p><strong>Price:</strong> ${{ selectedRow.price.toFixed(2) }}</p>

          <p><strong>Strike:</strong> ${{ selectedRow.strike.toFixed(2) }}</p>

          <p><strong>DTE:</strong> {{ selectedRow.dte }}</p>

          <p><strong>Expiry:</strong> {{ selectedRow.expiryDate }}</p>

          <hr class="border-gray-600 my-3" />

          <!-- AI-Based Targets -->
          <p>
            <strong>Suggested Stop-Loss:</strong>
            {{
              computeTradeHints(selectedRow.symbol).stopLoss
                ? computeTradeHints(selectedRow.symbol).stopLoss + "%"
                : "—"
            }}
          </p>

          <p>
            <strong>Suggested Take-Profit:</strong>
            {{
              computeTradeHints(selectedRow.symbol).takeProfit
                ? computeTradeHints(selectedRow.symbol).takeProfit + "%"
                : "—"
            }}
          </p>
        </div>

        <button
          @click="closeModal"
          class="mt-6 w-full bg-blue-600 hover:bg-blue-700 text-white py-2 rounded shadow"
        >
          Close
        </button>
      </div>
    </div>
  </div>
</template>
