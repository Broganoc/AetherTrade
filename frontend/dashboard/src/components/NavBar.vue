<script setup lang="ts">
import { defineProps, defineEmits } from 'vue'

const props = defineProps({
  isDark: Boolean,
  activeTab: String
})

const emit = defineEmits(['toggle-dark', 'change-tab'])

const toggleTheme = () => emit('toggle-dark')
const selectTab = (tab: string) => emit('change-tab', tab)
</script>

<template>
  <nav class="flex justify-between items-center px-6 py-4 bg-white dark:bg-gray-800 shadow">
    <!-- Logo -->
    <div class="font-bold text-lg text-gray-900 dark:text-gray-100">AetherTrade Dashboard</div>

    <!-- Tabs + Toggle -->
    <div class="flex items-center space-x-4">
      <a href="#" @click.prevent="selectTab('training')"
         :class="[
           'px-3 py-1 rounded',
           activeTab==='training'
             ? 'bg-blue-600 text-white dark:bg-blue-500 dark:text-white'
             : 'text-gray-700 hover:text-blue-600 dark:text-gray-200 dark:hover:text-blue-400'
         ]">
        Training
      </a>

      <a href="#" @click.prevent="selectTab('simulator')"
         :class="[
           'px-3 py-1 rounded',
           activeTab==='simulator'
             ? 'bg-blue-600 text-white dark:bg-blue-500 dark:text-white'
             : 'text-gray-700 hover:text-blue-600 dark:text-gray-200 dark:hover:text-blue-400'
         ]">
        Simulator
      </a>

<!--      <button @click="toggleTheme"-->
<!--              class="ml-4 px-3 py-1 rounded bg-gray-200 dark:bg-gray-700 dark:text-gray-100 hover:bg-gray-300 dark:hover:bg-gray-600 transition">-->
<!--        {{ props.isDark ? 'Light Mode' : 'Dark Mode' }}-->
<!--      </button>-->
    </div>
  </nav>
</template>
