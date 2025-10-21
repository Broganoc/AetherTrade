<template>
  <div :class="{ dark: isDarkMode }" class="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-200">
    <!-- NavBar -->
    <NavBar
      :active-tab="activeTab"
      :is-dark="isDarkMode"
      @toggle-dark="toggleDarkMode"
      @change-tab="activeTab = $event"
    />

    <!-- Tab content -->
    <div class="tab-content p-6">
      <TrainingTab v-if="activeTab === 'training'" />
      <SimulatorTab v-if="activeTab === 'simulator'" />
    </div>
  </div>
</template>

<script>
import TrainingTab from './components/TrainingTab.vue'
import SimulatorTab from './components/SimulatorTab.vue'
import NavBar from './components/NavBar.vue'

export default {
  components: { TrainingTab, SimulatorTab, NavBar },
  data() {
    return {
      activeTab: 'training',
      isDarkMode: false
    }
  },
  methods: {
    toggleDarkMode() {
      this.isDarkMode = !this.isDarkMode
    }
  }
}
</script>
