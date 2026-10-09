<script setup lang="ts">
defineProps<{ value: string; label?: string }>()
</script>

<template>
  <span class="status-badge" :class="`status-${value}`">{{ label ?? value }}</span>
</template>
