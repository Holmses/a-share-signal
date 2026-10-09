<script setup lang="ts">
import { LoaderCircle } from '@lucide/vue'

defineProps<{ label?: string }>()
</script>

<template>
  <div class="loading-state" role="status">
    <LoaderCircle :size="22" class="spin" />
    <span>{{ label ?? '正在读取数据' }}</span>
  </div>
</template>
