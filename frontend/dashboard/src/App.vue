<template>
  <div>
    <!-- Navbar -->
    <nav class="navbar">
      <div class="font-bold text-lg">AetherTrade Dashboard</div>
      <div>
        <a href="#" @click.prevent="activeTab = 'training'" :class="{ 'text-blue-600': activeTab==='training' }">Training</a>
        <a href="#" @click.prevent="activeTab = 'simulator'" :class="{ 'text-blue-600': activeTab==='simulator' }">Simulator</a>
      </div>
    </nav>

    <!-- Tab content -->
    <div class="tab-content">
      <TrainingTab v-if="activeTab === 'training'" />
      <SimulatorTab v-if="activeTab === 'simulator'" />
    </div>
  </div>
</template>

<script>
import TrainingTab from './components/TrainingTab.vue'
import SimulatorTab from './components/SimulatorTab.vue'

export default {
  components: { TrainingTab, SimulatorTab },
  data() {
    return {
      activeTab: 'training'
    }
  }
}
</script>
