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

      <!-- REAL STOP BUTTON -->
      <button
        v-if="isTraining"
        @click="cancelTraining"
        class="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded shadow"
      >
        Stop
      </button>
    </div>

    <!-- Progress & Live Metrics -->
    <div v-if="trainingStatus" class="mb-6 space-y-2">
      <div class="w-full bg-gray-200 rounded h-4 dark:bg-gray-700 overflow-hidden">
        <div
          class="h-4 text-xs text-white flex items-center justify-center transition-all duration-300 ease-out"
          :class="{
            'bg-blue-500': !isCompleted && !hasError && !isResumeMode && !isFullTrainMode,
            'bg-indigo-600': isResumeMode,
            'bg-purple-600': isFullTrainMode && !isCompleted,
            'bg-green-600': isCompleted,
            'bg-red-600': hasError
          }"
          :style="{ width: trainingProgress + '%' }"
        >
          {{ trainingProgress }}%
        </div>
      </div>

      <!-- Status Line -->
      <div class="flex flex-wrap items-center justify-between text-sm text-gray-600 dark:text-gray-300">
        <div class="flex items-center gap-2">
          <span v-if="isResumeMode"
            class="px-2 py-0.5 rounded bg-indigo-100 text-indigo-700 dark:bg-indigo-900 dark:text-indigo-200 text-xs">
            Resuming…
          </span>
          <span v-if="isFullTrainMode"
            class="px-2 py-0.5 rounded bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-200 text-xs">
            Full Training…
          </span>

          <span>{{ trainingStatus }}</span>

          <!-- Chunk counter -->
          <span v-if="chunk && chunksTotal" class="ml-2 text-xs">
            (Chunk {{ chunk }}/{{ chunksTotal }})
          </span>
        </div>

        <div class="text-xs text-gray-500 dark:text-gray-400">
          <span v-if="elapsedSeconds !== null">Elapsed: {{ formatDuration(elapsedSeconds) }}</span>
          <span v-if="etaSeconds !== null" class="ml-2">ETA: {{ formatDuration(etaSeconds) }}</span>
          <span v-if="lastTimestamp" class="ml-2">Updated: {{ lastTimestamp }}</span>
        </div>
      </div>

      <!-- Metrics Grid -->
      <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2 text-xs">
        <div v-for="m in metricList" :key="m.key" class="p-2 rounded border dark:border-gray-700">
          <div class="text-gray-500 dark:text-gray-400">{{ m.label }}</div>
          <div class="font-semibold">
            {{ m.value !== null ? m.value : "—" }}
          </div>
        </div>
      </div>

      <!-- Full Train Warnings -->
      <div v-if="fullTrainWarnings" class="mt-2 text-xs text-yellow-600 dark:text-yellow-400">
        <div v-for="(val, key) in fullTrainWarnings" :key="key" v-if="val">
          ⚠ {{ key.replace('_',' ') }} detected
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
          {{ model.metrics?.mean_reward?.toFixed(4) ?? "N/A" }}
        </p>

        <p class="text-sm text-gray-600 dark:text-gray-300">
          Mean Episode Length:
          {{ model.metrics?.mean_episode_length?.toFixed(2) ?? "N/A" }}
        </p>

        <!-- Expand -->
        <transition name="fade">
          <div
            v-if="model.expanded"
            class="mt-3 border-t border-gray-200 dark:border-gray-700 pt-2 text-sm text-gray-700 dark:text-gray-300"
          >
            <p><strong>Path:</strong> {{ model.path }}</p>

            <p class="mt-2"><strong>Best Checkpoint:</strong></p>
            <p v-if="model.best?.val_reward">
              Best Val Reward: {{ model.best.val_reward.toFixed(4) }}
            </p>
            <p v-else>No best checkpoint saved yet</p>
          </div>
        </transition>

        <!-- Card actions -->
        <div class="mt-3 flex flex-wrap items-center gap-3">
          <button class="text-blue-600 hover:text-blue-700 text-sm" @click="toggleExpanded(index)">
            {{ model.expanded ? "Hide details" : "Show details" }}
          </button>

          <!-- Train More -->
          <button class="text-indigo-600 hover:text-indigo-700 text-sm" @click="toggleTrainMore(index)">
            {{ model.showTrainMore ? "Close" : "Train More" }}
          </button>

          <!-- Full Train -->
          <button class="text-purple-600 hover:text-purple-700 text-sm" @click="toggleFullTrain(index)">
            {{ model.showFullTrain ? "Close" : "Full Train" }}
          </button>
        </div>

        <!-- Train More inline -->
        <transition name="fade">
          <div v-if="model.showTrainMore" class="mt-3 p-3 rounded border dark:border-gray-700">
            <div class="flex items-center gap-2">
              <label class="text-sm text-gray-600 dark:text-gray-300 w-28">Extra Timesteps</label>
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

        <!-- Full Train inline -->
        <transition name="fade">
          <div v-if="model.showFullTrain" class="mt-3 p-3 rounded border dark:border-gray-700">
            <button
              class="bg-purple-600 hover:bg-purple-700 text-white px-3 py-1.5 rounded shadow text-sm disabled:opacity-50"
              :disabled="isTraining"
              @click="startFullTrain(model)"
            >
              {{ isTraining ? "Training…" : "Start Full Train" }}
            </button>

            <p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Runs continuous training until early-stop metrics trigger or manually cancelled.
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
      isFullTrainMode: false,

      // Live Metrics
      meanReward: null,
      meanEpLength: null,
      valMean: null,
      valStd: null,
      policyLoss: null,
      valueLoss: null,
      entropy: null,
      explainedVariance: null,
      approxKL: null,
      clipFraction: null,

      fullTrainWarnings: null,

      // chunk progress
      chunk: null,
      chunksTotal: null,

      elapsedSeconds: null,
      etaSeconds: null,
      lastTimestamp: null,

      // Training inputs
      timesteps: 1_000_000,
      chunks: 20,
      evalEpisodes: 5
    };
  },

  computed: {
    metricList() {
      return [
        { key: "reward", label: "Train Reward", value: this.formatVal(this.meanReward) },
        { key: "eplen", label: "Ep Len", value: this.formatVal(this.meanEpLength) },
        { key: "valmean", label: "Val Mean", value: this.formatVal(this.valMean) },
        { key: "valstd", label: "Val Std", value: this.formatVal(this.valStd) },
        { key: "pol", label: "Policy Loss", value: this.formatVal(this.policyLoss) },
        { key: "valloss", label: "Value Loss", value: this.formatVal(this.valueLoss) },
        { key: "ent", label: "Entropy", value: this.formatVal(this.entropy) },
        { key: "expl", label: "Expl. Var", value: this.formatVal(this.explainedVariance) },
        { key: "kl", label: "Approx KL", value: this.formatVal(this.approxKL) },
        { key: "clip", label: "Clip Frac", value: this.formatVal(this.clipFraction) }
      ];
    }
  },

  created() {
    this.fetchModels();
  },

  beforeUnmount() {
    if (this.eventSource) this.eventSource.close();
  },

  methods: {
    formatVal(v) {
      return v !== null && v !== undefined ? Number(v).toFixed(3) : "—";
    },

    async fetchModels() {
      try {
        const res = await fetch("http://localhost:8001/models");
        const data = await res.json();

        this.models = data.map(m => ({
          ...m,
          expanded: false,
          showTrainMore: false,
          showFullTrain: false,
          extraTimesteps: 1_000_000
        }));
      } catch (err) {
        console.error("Failed to fetch models:", err);
      }
    },

    toggleExpanded(i) {
      this.models[i].expanded = !this.models[i].expanded;
    },

    toggleTrainMore(i) {
      this.models[i].showTrainMore = !this.models[i].showTrainMore;
      this.models[i].showFullTrain = false;
    },

    toggleFullTrain(i) {
      this.models[i].showFullTrain = !this.models[i].showFullTrain;
      this.models[i].showTrainMore = false;
    },

    resetLiveState() {
      this.trainingProgress = 0;
      this.trainingStatus = "";
      this.hasError = false;
      this.isCompleted = false;

      this.meanReward = this.meanEpLength = this.valMean = this.valStd = null;
      this.policyLoss = this.valueLoss = this.entropy = this.explainedVariance = null;
      this.approxKL = this.clipFraction = null;

      this.chunk = this.chunksTotal = null;

      this.fullTrainWarnings = null;

      this.elapsedSeconds = this.etaSeconds = null;
      this.lastTimestamp = null;
    },

    // ----------- Normal Training -----------
    async runTraining() {
      if (!this.selectedSymbol) {
        this.trainingStatus = "Please enter a stock symbol!";
        return;
      }

      // clear cancel flag
      await fetch("http://localhost:8001/cancel_training", { method: "DELETE" });

      this.isTraining = true;
      this.isResumeMode = false;
      this.isFullTrainMode = false;
      this.resetLiveState();
      this.trainingStatus = `Training started for ${this.selectedSymbol}...`;

      const url = new URL("http://localhost:8001/train_stream");
      url.searchParams.set("symbol", this.selectedSymbol);
      url.searchParams.set("timesteps", this.timesteps);
      url.searchParams.set("chunks", this.chunks);
      url.searchParams.set("eval_episodes", this.evalEpisodes);

      this.startSSE(url.toString());
    },

    // ----------- Resume Training -----------
    async startResume(model) {
      const filename = model.path.split("/").pop();
      if (!filename) return;

      await fetch("http://localhost:8001/cancel_training", { method: "DELETE" });

      this.isTraining = true;
      this.isResumeMode = true;
      this.isFullTrainMode = false;
      this.resetLiveState();
      this.trainingStatus = `Resuming ${filename}...`;

      const url = new URL("http://localhost:8001/resume_stream");
      url.searchParams.set("model_name", filename);
      url.searchParams.set("timesteps", model.extraTimesteps);
      url.searchParams.set("chunks", this.chunks);
      url.searchParams.set("eval_episodes", this.evalEpisodes);

      this.startSSE(url.toString());
    },

    // ----------- Full Training -----------
    async startFullTrain(model) {
      const filename = model.path.split("/").pop();
      if (!filename) return;

      await fetch("http://localhost:8001/cancel_training", { method: "DELETE" });

      this.isTraining = true;
      this.isFullTrainMode = true;
      this.isResumeMode = false;
      this.resetLiveState();

      this.trainingStatus = `Full Training ${filename}...`;

      const url = new URL("http://localhost:8001/full_train_stream");
      url.searchParams.set("model_name", filename);
      url.searchParams.set("chunks_per_round", this.chunks);
      url.searchParams.set("eval_episodes", this.evalEpisodes);

      this.startSSE(url.toString());
    },

    // -------- SSE Unified Handler --------
    startSSE(url) {
      if (this.eventSource) this.eventSource.close();

      this.eventSource = new EventSource(url);

      this.eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // ---- CANCELLED ----
          if (data.status === "cancelled") {
            this.trainingStatus = "Training Cancelled";
            this.isTraining = false;
            if (this.eventSource) this.eventSource.close();
            this.eventSource = null;
            return;
          }

          // ---- STARTED ----
          if (data.status === "started") {
            this.trainingProgress = data.progress || 0;
            this.chunksTotal = data.chunks || null;

            this.trainingStatus = `Started ${data.mode} ${data.symbols?.join(", ") || ""}`.trim();
          }

          // ---- TRAINING ----
          else if (data.status === "training") {
            this.trainingProgress = data.progress ?? this.trainingProgress;

            if (data.mode === "resume") {
              this.trainingStatus = `Resuming ${data.symbols?.join(", ")}`;
            } else if (data.mode === "full_train") {
              this.trainingStatus = "Full training…";
            } else {
              this.trainingStatus = `Training ${data.symbols?.join(", ")}`;
            }

            this.chunk = data.chunk ?? this.chunk;
            this.chunksTotal = data.chunks ?? this.chunksTotal;

            this.meanReward = data.mean_reward ?? this.meanReward;
            this.meanEpLength = data.mean_episode_length ?? this.meanEpLength;
            this.valMean = data.val_mean ?? this.valMean;
            this.valStd = data.val_std ?? this.valStd;

            this.policyLoss = data.policy_loss ?? this.policyLoss;
            this.valueLoss = data.value_loss ?? this.valueLoss;
            this.entropy = data.entropy ?? this.entropy;
            this.explainedVariance = data.explained_variance ?? this.explainedVariance;
            this.approxKL = data.approx_kl ?? this.approxKL;
            this.clipFraction = data.clip_fraction ?? this.clipFraction;

            if (data.warnings) {
              this.fullTrainWarnings = data.warnings;
            }

            this.elapsedSeconds = data.elapsed_seconds ?? this.elapsedSeconds;
            this.etaSeconds = data.eta_seconds ?? this.etaSeconds;
            this.lastTimestamp = data.timestamp ?? this.lastTimestamp;
          }

          // ---- COMPLETED ----
          else if (data.status === "completed") {
            this.trainingProgress = 100;
            this.isCompleted = true;
            this.isTraining = false;

            this.trainingStatus = data.stopped_early
              ? `Full training stopped early: ${data.reason}`
              : "Training completed";

            setTimeout(() => this.fetchModels(), 1000);

            if (this.eventSource) {
              this.eventSource.close();
              this.eventSource = null;
            }

            setTimeout(() => {
              this.trainingStatus = "";
              this.trainingProgress = 0;
            }, 2000);
          }

          // ---- ERROR ----
          else if (data.status === "error") {
            this.hasError = true;
            this.isTraining = false;
            this.trainingStatus = `Error: ${data.message}`;
            if (this.eventSource) this.eventSource.close();
            this.eventSource = null;
          }

        } catch (err) {
          console.error("SSE parse error:", err);
        }
      };

      this.eventSource.onerror = (err) => {
        console.error("SSE Error:", err);
        this.trainingStatus = "Connection lost.";
        this.isTraining = false;
        this.hasError = true;
        if (this.eventSource) this.eventSource.close();
        this.eventSource = null;
      };
    },

    // ----------- REAL STOP BUTTON -----------
    async cancelTraining() {
      try {
        await fetch("http://localhost:8001/cancel_training", {
          method: "DELETE"
        });
      } catch (err) {
        console.error("Failed to send cancel:", err);
      }

      if (this.eventSource) {
        this.eventSource.close();
        this.eventSource = null;
      }

      this.trainingStatus = "Stopping…";
      this.isTraining = false;
    },

    formatDuration(sec) {
      if (sec === null) return "—";
      const s = Math.max(0, Math.round(sec));
      const h = Math.floor(s / 3600);
      const m = Math.floor((s % 3600) / 60);
      const rs = s % 60;
      if (h > 0) return `${h}h ${m}m ${rs}s`;
      if (m > 0) return `${m}m ${rs}s`;
      return `${rs}s`;
    }
  }
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
