<script setup lang="ts">
import { Activity, BarChart3, CandlestickChart, FlaskConical, ListChecks } from '@lucide/vue'

const navigation = [
  { to: '/', label: '今日交易台', icon: Activity },
  { to: '/stock', label: '个股详情', icon: CandlestickChart },
  { to: '/backtest', label: '回测实验', icon: FlaskConical },
  { to: '/compare', label: '策略对比', icon: BarChart3 },
  { to: '/tasks', label: '任务与数据', icon: ListChecks },
]
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <div class="brand-block">
        <div class="brand-mark">AS</div>
        <div>
          <strong>A 股策略控制台</strong>
          <span>日线研究与模拟盘复盘</span>
        </div>
      </div>
      <nav class="primary-nav" aria-label="主导航">
        <RouterLink v-for="item in navigation" :key="item.to" :to="item.to">
          <component :is="item.icon" :size="17" aria-hidden="true" />
          {{ item.label }}
        </RouterLink>
      </nav>
      <div class="environment-badge">
        <span class="status-dot"></span>
        仅研究
      </div>
    </header>
    <main class="page-frame">
      <RouterView />
    </main>
  </div>
</template>
