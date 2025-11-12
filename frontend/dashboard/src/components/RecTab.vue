<script setup>
import { ref, onMounted } from "vue";

const models = ref([]);
const selectedModel = ref("");
const predictions = ref([]);
const logs = ref([]);
const selectedLog = ref("");
const loading = ref(false);

// ------------------------------------------
// Fetch available models
// ------------------------------------------
async function loadModels() {
  const res = await fetch("http://localhost:8003/models");
  models.value = await res.json();
}

// ------------------------------------------
// Run prediction with selected model
// ------------------------------------------
async function runPredictions() {
  if (!selectedModel.value) {
    alert("Please select a model first.");
    return;
  }
  loading.value = true;
  const url = new URL("http://localhost:8003/predict");
  url.searchParams.set("model_name", selectedModel.value);
  const res = await fetch(url, { method: "POST" });
  const data = await res.json();
  predictions.value = data.predictions || [];
  loading.value = false;
}

onMounted(loadModels);
</script>

<template>
  <div class="p-6">
    <h2 class="text-xl font-semibold mb-4 dark:text-gray-100">Model Recommendations</h2>

    <div class="flex items-center gap-4 mb-6">
      <select v-model="selectedModel" class="p-2 rounded bg-gray-700 text-white">
        <option value="">Select Model</option>
        <option v-for="m in models" :key="m.full_name" :value="m.full_name">
          {{ m.full_name }}
        </option>
      </select>
      <button
        @click="runPredictions"
        :disabled="loading"
        class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
      >
        Run Predictions
      </button>
    </div>

    <table v-if="predictions.length" class="w-full text-sm text-gray-200">
      <thead>
        <tr class="border-b border-gray-600">
          <th class="text-left py-2">Symbol</th>
          <th class="text-left">Action</th>
          <th class="text-left">Confidence</th>
          <th class="text-left">Price</th>
          <th class="text-left">Date</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="p in predictions" :key="p.symbol" class="border-b border-gray-700">
          <td>{{ p.symbol }}</td>
          <td :class="{
            'text-green-400': p.action === 'CALL',
            'text-red-400': p.action === 'PUT',
            'text-yellow-300': p.action === 'HOLD'
          }">{{ p.action }}</td>
          <td>{{ p.confidence }}%</td>
          <td>${{ p.price }}</td>
          <td>{{ p.date }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
