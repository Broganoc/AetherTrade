<template>
  <div class="p-6 min-h-screen">
    <h2 class="text-xl font-semibold mb-4 dark:text-gray-100">Training Dashboard</h2>

    <!-- Controls -->
    <div class="mb-4 flex flex-wrap items-center gap-2">
      <input
        v-model="selectedSymbol"
        list="symbols-list"
        placeholder="Enter stock symbol (comma-separated for multi)"
        class="px-3 py-2 border rounded dark:bg-gray-700 dark:text-gray-100"
      />
      <datalist id="symbols-list">
        <option v-for="symbol in symbols" :key="symbol" :value="symbol">{{ symbol }}</option>
      </datalist>

      <input
        v-model.number="timesteps"
        type="number"
        min="10000"
        step="10000"
        class="px-3 py-2 border rounded w-40 dark:bg-gray-700 dark:text-gray-100"
        placeholder="Timesteps"
      />
      <input
        v-model.number="chunks"
        type="number"
        min="2"
        step="1"
        class="px-3 py-2 border rounded w-28 dark:bg-gray-700 dark:text-gray-100"
        placeholder="Chunks"
      />
      <input
        v-model.number="evalEpisodes"
        type="number"
        min="1"
        step="1"
        class="px-3 py-2 border rounded w-36 dark:bg-gray-700 dark:text-gray-100"
        placeholder="Eval Episodes"
      />

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

    <!-- Progress & Live Metrics -->
    <div v-if="trainingStatus" class="mb-6 space-y-2">
      <div class="w-full bg-gray-200 rounded h-4 dark:bg-gray-700 overflow-hidden">
        <div
          class="h-4 text-xs text-white flex items-center justify-center transition-all duration-300 ease-out"
          :class="{
            'bg-blue-500': !isCompleted && !hasError && !isResumeMode,
            'bg-indigo-600': isResumeMode && !isCompleted && !hasError,
            'bg-green-600': isCompleted,
            'bg-red-600': hasError
          }"
          :style="{ width: trainingProgress + '%' }"
        >
          {{ trainingProgress }}%
        </div>
      </div>

      <div class="flex flex-wrap items-center justify-between text-sm text-gray-600 dark:text-gray-300">
        <div class="flex items-center gap-2">
          <span v-if="isResumeMode" class="px-2 py-0.5 rounded bg-indigo-100 text-indigo-700 dark:bg-indigo-900 dark:text-indigo-200 text-xs">
            Resuming model…
          </span>
          <span>{{ trainingStatus }}</span>
          <span v-if="chunk && chunksTotal" class="ml-2">
            (Chunk {{ chunk }}/{{ chunksTotal }})
          </span>
        </div>
        <div class="text-xs text-gray-500 dark:text-gray-400">
          <span v-if="elapsedSeconds !== null">Elapsed: {{ formatDuration(elapsedSeconds) }}</span>
          <span v-if="etaSeconds !== null" class="ml-2">ETA: {{ formatDuration(etaSeconds) }}</span>
          <span v-if="lastTimestamp" class="ml-2">Updated: {{ lastTimestamp }}</span>
        </div>
      </div>

      <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2 text-xs">
        <div class="p-2 rounded border dark:border-gray-700">
          <div class="text-gray-500 dark:text-gray-400">Train Reward</div>
          <div class="font-semibold">
            {{ meanReward !== null ? meanReward.toFixed(2) : "—" }}
          </div>
        </div>
        <div class="p-2 rounded border dark:border-gray-700">
          <div class="text-gray-500 dark:text-gray-400">Ep Len</div>
          <div class="font-semibold">
            {{ meanEpLength !== null ? meanEpLength.toFixed(1) : "—" }}
          </div>
        </div>
        <div class="p-2 rounded border dark:border-gray-700">
          <div class="text-gray-500 dark:text-gray-400">Val Mean</div>
          <div class="font-semibold">
            {{ valMean !== null ? valMean.toFixed(2) : "—" }}
          </div>
        </div>
        <div class="p-2 rounded border dark:border-gray-700">
          <div class="text-gray-500 dark:text-gray-400">Val Std</div>
          <div class="font-semibold">
            {{ valStd !== null ? valStd.toFixed(2) : "—" }}
          </div>
        </div>
        <div class="p-2 rounded border dark:border-gray-700">
          <div class="text-gray-500 dark:text-gray-400">Policy Loss</div>
          <div class="font-semibold">
            {{ policyLoss !== null ? policyLoss.toFixed(4) : "—" }}
          </div>
        </div>
        <div class="p-2 rounded border dark:border-gray-700">
          <div class="text-gray-500 dark:text-gray-400">Value Loss</div>
          <div class="font-semibold">
            {{ valueLoss !== null ? valueLoss.toFixed(4) : "—" }}
          </div>
        </div>
        <div class="p-2 rounded border dark:border-gray-700">
          <div class="text-gray-500 dark:text-gray-400">Entropy</div>
          <div class="font-semibold">
            {{ entropy !== null ? entropy.toFixed(4) : "—" }}
          </div>
        </div>
        <div class="p-2 rounded border dark:border-gray-700">
          <div class="text-gray-500 dark:text-gray-400">Expl. Var</div>
          <div class="font-semibold">
            {{ explainedVariance !== null ? explainedVariance.toFixed(3) : "—" }}
          </div>
        </div>
      </div>
    </div>

    <!-- Models Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      <div
        v-for="(model, index) in models"
        :key="model.full_name"
        class="bg-white dark:bg-gray-800 p-4 rounded-lg shadow hover:shadow-lg transition"
      >
        <div class="flex justify-between items-center">
          <h3 class="font-semibold text-gray-800 dark:text-gray-100">
            {{ model.model_name }} — {{ model.symbol }}
          </h3>
          <span class="text-sm text-gray-500 dark:text-gray-400">{{ model.trained_on }}</span>
        </div>

        <p class="text-sm text-gray-600 dark:text-gray-300 mt-1">Framework: {{ model.framework }}</p>

        <p class="text-sm text-gray-600 dark:text-gray-300">
          Mean Reward:
          {{
            model.metrics?.mean_reward !== null && model.metrics?.mean_reward !== undefined
              ? model.metrics.mean_reward.toFixed(4)
              : "N/A"
          }}
        </p>

        <p class="text-sm text-gray-600 dark:text-gray-300">
          Mean Episode Length:
          {{
            model.metrics?.mean_episode_length !== null && model.metrics?.mean_episode_length !== undefined
              ? model.metrics.mean_episode_length.toFixed(2)
              : "N/A"
          }}
        </p>

        <!-- Expand / Details -->
        <transition name="fade">
          <div
            v-if="model.expanded"
            class="mt-3 border-t border-gray-200 dark:border-gray-700 pt-2 text-sm text-gray-700 dark:text-gray-300"
          >
            <p><strong>Path:</strong> {{ model.path }}</p>
            <p class="mt-2"><strong>Metrics:</strong></p>
            <ul class="mt-1 space-y-1">
              <li>
                Mean Reward:
                {{
                  model.metrics?.mean_reward !== null && model.metrics?.mean_reward !== undefined
                    ? model.metrics.mean_reward.toFixed(4)
                    : "N/A"
                }}
              </li>
              <li>
                Mean Episode Length:
                {{
                  model.metrics?.mean_episode_length !== null && model.metrics?.mean_episode_length !== undefined
                    ? model.metrics.mean_episode_length.toFixed(2)
                    : "N/A"
                }}
              </li>
            </ul>
          </div>
        </transition>

        <!-- Card actions -->
        <div class="mt-3 flex flex-wrap items-center gap-2">
          <button
            class="text-blue-600 hover:text-blue-700 text-sm"
            @click="toggleExpanded(index)"
          >
            {{ model.expanded ? "Hide details" : "Show details" }}
          </button>

          <!-- Train More toggle -->
          <button
            class="text-indigo-600 hover:text-indigo-700 text-sm"
            @click="toggleTrainMore(index)"
          >
            {{ model.showTrainMore ? "Close" : "Train More" }}
          </button>
        </div>

        <!-- Train More inline controls -->
        <transition name="fade">
          <div v-if="model.showTrainMore" class="mt-3 p-3 rounded border dark:border-gray-700">
            <div class="flex items-center gap-2">
              <label class="text-sm text-gray-600 dark:text-gray-300">Extra Timesteps</label>
              <input
                v-model.number="model.extraTimesteps"
                type="number"
                min="10000"
                step="10000"
                class="px-2 py-1 border rounded w-40 dark:bg-gray-700 dark:text-gray-100"
              />
              <button
                class="bg-indigo-600 hover:bg-indigo-700 text-white px-3 py-1.5 rounded shadow text-sm disabled:opacity-50"
                :disabled="isTraining"
                @click="startResume(model)"
              >
                {{ isTraining ? "Resuming…" : "Start" }}
              </button>
            </div>
            <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Continues training this model and overwrites the existing checkpoint.
            </p>
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
      hasError: false,
      isCompleted: false,
      isResumeMode: false,

      // live metrics
      meanReward: null,
      meanEpLength: null,
      valMean: null,
      valStd: null,
      policyLoss: null,
      valueLoss: null,
      entropy: null,
      explainedVariance: null,

      // timing & chunk info
      chunk: null,
      chunksTotal: null,
      elapsedSeconds: null,
      etaSeconds: null,
      lastTimestamp: null,

      // inputs
      timesteps: 1_000_000,
      chunks: 20,
      evalEpisodes: 5,
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
        // add UI state to each model card
        this.models = data.map((m) => ({
          ...m,
          expanded: false,
          showTrainMore: false,
          extraTimesteps: 1_000_000
        }));
      } catch (err) {
        console.error("Failed to fetch models:", err);
      }
    },
    toggleExpanded(index) {
      this.models[index].expanded = !this.models[index].expanded;
    },
    toggleTrainMore(index) {
      this.models[index].showTrainMore = !this.models[index].showTrainMore;
    },
    formatDuration(sec) {
      if (sec === null || sec === undefined) return "—";
      const s = Math.max(0, Math.round(sec));
      const h = Math.floor(s / 3600);
      const m = Math.floor((s % 3600) / 60);
      const rs = s % 60;
      if (h > 0) return `${h}h ${m}m ${rs}s`;
      if (m > 0) return `${m}m ${rs}s`;
      return `${rs}s`;
    },
    resetLiveState() {
      this.trainingProgress = 0;
      this.trainingStatus = "";
      this.meanReward = this.meanEpLength = this.valMean = this.valStd = null;
      this.policyLoss = this.valueLoss = this.entropy = this.explainedVariance = null;
      this.chunk = this.chunksTotal = null;
      this.elapsedSeconds = this.etaSeconds = null;
      this.lastTimestamp = null;
      this.hasError = false;
      this.isCompleted = false;
    },
    async runTraining() {
      if (!this.selectedSymbol) {
        this.trainingStatus = "Please enter a stock symbol!";
        return;
      }

      // reset state
      this.isResumeMode = false;
      this.isTraining = true;
      this.resetLiveState();
      this.trainingStatus = `Training started for ${this.selectedSymbol}...`;

      const url = new URL("http://localhost:8001/train_stream");
      url.searchParams.set("symbol", this.selectedSymbol);
      url.searchParams.set("timesteps", this.timesteps);
      url.searchParams.set("chunks", this.chunks);
      url.searchParams.set("eval_episodes", this.evalEpisodes);

      this.startSSE(url.toString());
    },
    startResume(model) {
      // model.path ends with the filename we need
      const filename = model.path.split("/").pop();
      if (!filename) return;

      // reset state
      this.isResumeMode = true;
      this.isTraining = true;
      this.resetLiveState();
      this.trainingStatus = `Resuming ${filename}...`;

      const url = new URL("http://localhost:8001/resume_stream");
      url.searchParams.set("model_name", filename);
      url.searchParams.set("timesteps", model.extraTimesteps || 1000000);
      // match global defaults for consistency
      url.searchParams.set("chunks", this.chunks);
      url.searchParams.set("eval_episodes", this.evalEpisodes);

      this.startSSE(url.toString());
    },
    startSSE(url) {
      if (this.eventSource) this.eventSource.close();
      this.eventSource = new EventSource(url);

      this.eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.status === "started") {
            this.trainingProgress = data.progress || 0;
            const label = data.mode === "resume" ? "Resuming model" : "Training started for";
            this.trainingStatus = `${label} ${data.symbols?.join(", ") || ""}`.trim();
            this.chunksTotal = data.chunks || null;
          } else if (data.status === "training") {
            this.trainingProgress = data.progress || 0;

            // Set label respecting mode
            const label = data.mode === "resume" ? "Resuming model" : "Training";
            this.trainingStatus = `${label} ${data.symbols?.join(", ") || ""}`.trim();

            // metrics
            this.meanReward = data.mean_reward ?? this.meanReward;
            this.meanEpLength = data.mean_episode_length ?? this.meanEpLength;
            this.valMean = data.val_mean ?? this.valMean;
            this.valStd = data.val_std ?? this.valStd;
            this.policyLoss = data.policy_loss ?? this.policyLoss;
            this.valueLoss = data.value_loss ?? this.valueLoss;
            this.entropy = data.entropy ?? this.entropy;
            this.explainedVariance = data.explained_variance ?? this.explainedVariance;

            // timing
            this.chunk = data.chunk ?? this.chunk;
            this.chunksTotal = data.chunks ?? this.chunksTotal;
            this.elapsedSeconds = data.elapsed_seconds ?? this.elapsedSeconds;
            this.etaSeconds = data.eta_seconds ?? this.etaSeconds;
            this.lastTimestamp = data.timestamp ?? this.lastTimestamp;
          } else if (data.status === "completed") {
            this.trainingProgress = 100;
            const label = data.mode === "resume" ? "✅ Resume completed" : "✅ Training completed";
            this.trainingStatus = `${label} ${data.symbols?.join(", ") || ""}`.trim();
            this.valMean = data.val_mean ?? this.valMean;
            this.valStd = data.val_std ?? this.valStd;
            this.isCompleted = true;
            this.isTraining = false;

            // Immediately refresh model list after completion
            setTimeout(() => this.fetchModels(), 1000);


            // Give user a visual pause before clearing status
            setTimeout(() => {
              this.trainingStatus = "";
              this.trainingProgress = 0;
            }, 2000);

            if (this.eventSource) {
              this.eventSource.close();
              this.eventSource = null;
            }
          } else if (data.status === "error") {
            this.hasError = true;
            this.isTraining = false;
            this.trainingStatus = `❌ Error: ${data.message || "Unknown error"}`;
            if (this.eventSource) {
              this.eventSource.close();
              this.eventSource = null;
            }
          }
        } catch (e) {
          console.error("SSE parse error:", e);
        }
      };

      this.eventSource.onerror = (err) => {
        console.error("SSE Error:", err);
        this.trainingStatus = "⚠️ Connection lost.";
        this.isTraining = false;
        this.hasError = true;
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
      this.isCompleted = false;
      this.hasError = false;
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
