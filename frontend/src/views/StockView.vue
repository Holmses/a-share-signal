<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { Download, ImageDown, Search } from '@lucide/vue'
import { useRoute, useRouter } from 'vue-router'
import { api } from '../api'
import LoadingState from '../components/LoadingState.vue'
import StatusBadge from '../components/StatusBadge.vue'
import StockChart from '../components/StockChart.vue'
import { formatDate, formatMoney, formatNumber, statusLabel, valueClass } from '../format'
import type { ResearchResult, StockDetail } from '../types'

const route = useRoute()
const router = useRouter()
const detail = ref<StockDetail | null>(null)
const results = ref<ResearchResult[]>([])
const resultId = ref('')
const range = ref<'1y' | 'all'>('1y')
const query = ref('')
const suggestions = ref<Array<Record<string, string>>>([])
const loading = ref(false)
const error = ref('')
const chartRef = ref<InstanceType<typeof StockChart> | null>(null)
let searchTimer: number | null = null

const symbol = computed(() => String(route.params.symbol || ''))
const focusDate = computed(() => typeof route.query.trade_date === 'string' ? route.query.trade_date.replaceAll('-', '') : '')

async function resolveDefaultSymbol() {
  if (symbol.value) return
  const dashboard = await api.dashboard()
  const fallback = dashboard.positions[0]?.symbol ?? dashboard.candidates[0]?.symbol
  if (fallback) await router.replace(`/stock/${fallback}`)
}

async function load() {
  if (!symbol.value) return
  loading.value = true
  error.value = ''
  try {
    detail.value = await api.stock(symbol.value, range.value, resultId.value || undefined)
    query.value = `${detail.value.symbol} ${detail.value.name}`
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : '读取个股数据失败'
  } finally {
    loading.value = false
  }
}

async function search() {
  suggestions.value = query.value.trim() ? await api.searchStocks(query.value.trim()) : []
}

async function selectStock(value: Record<string, string>) {
  suggestions.value = []
  await router.push(`/stock/${value.ts_code}`)
}

function downloadPng() {
  const url = chartRef.value?.exportPng()
  if (!url || !detail.value) return
  const link = document.createElement('a')
  link.href = url
  link.download = `${detail.value.symbol}-trades.png`
  link.click()
}

function downloadCsv() {
  if (!detail.value) return
  const params = new URLSearchParams({ range: range.value })
  if (resultId.value) params.set('result_id', resultId.value)
  window.location.href = `/api/stocks/${detail.value.symbol}/export.csv?${params}`
}

watch(() => route.params.symbol, load)
watch([range, resultId], load)
watch(query, () => {
  if (searchTimer) window.clearTimeout(searchTimer)
  searchTimer = window.setTimeout(search, 220)
})

onMounted(async () => {
  results.value = (await api.results()).filter((value) => Boolean(value.trades_path))
  resultId.value = typeof route.query.result_id === 'string' ? route.query.result_id : ''
  if (focusDate.value) range.value = 'all'
  await resolveDefaultSymbol()
  await nextTick()
  await load()
})
</script>

<template>
  <div>
    <div class="page-header">
      <div>
        <h1 class="page-title">个股详情</h1>
        <p class="page-subtitle">不复权日线 · 信号与实际成交分层 · 默认最近一年</p>
      </div>
      <div class="header-actions">
        <button class="button" :disabled="!detail?.bars.length" @click="downloadCsv"><Download :size="15" />导出 CSV</button>
        <button class="button" :disabled="!detail?.bars.length" @click="downloadPng"><ImageDown :size="15" />导出 PNG</button>
      </div>
    </div>

    <section class="section stock-toolbar">
      <div class="stock-search">
        <Search :size="16" />
        <input v-model="query" aria-label="搜索股票" placeholder="输入代码或名称" @keydown.enter="suggestions[0] && selectStock(suggestions[0])" />
        <div v-if="suggestions.length" class="search-results">
          <button v-for="item in suggestions" :key="item.ts_code" @click="selectStock(item)">
            <strong>{{ item.ts_code }}</strong><span>{{ item.name }}</span><small>{{ item.industry }}</small>
          </button>
        </div>
      </div>
      <select v-model="resultId" aria-label="交易数据来源">
        <option value="">模拟盘与当前交易记录</option>
        <option v-for="item in results" :key="item.id" :value="item.id">{{ item.title }}</option>
      </select>
      <div class="segmented">
        <button :class="{ active: range === '1y' }" @click="range = '1y'">最近一年</button>
        <button :class="{ active: range === 'all' }" @click="range = 'all'">全部历史</button>
      </div>
    </section>

    <LoadingState v-if="loading" label="正在读取日线和交易事件" />
    <div v-else-if="error" class="error-state">{{ error }}</div>
    <template v-else-if="detail">
      <section class="section">
        <div class="section-heading">
          <div><h2>{{ detail.symbol }} {{ detail.name }}</h2><p>{{ detail.industry ?? '行业未分类' }} · 不复权 · {{ detail.bars.length }} 个交易日</p></div>
          <StatusBadge :value="detail.execution_audit.t_plus_one_valid ? 'completed' : 'failed'" :label="detail.execution_audit.t_plus_one_valid ? 'T+1 审计通过' : '存在执行异常'" />
        </div>
        <div class="panel chart-panel">
          <StockChart v-if="detail.bars.length" ref="chartRef" :bars="detail.bars" :events="detail.events" :focus-date="focusDate" />
          <div v-else class="chart-empty">没有可绘制的日线行情</div>
        </div>
      </section>

      <section class="section">
        <div class="metric-strip audit-strip">
          <div class="metric"><span class="metric-label">实际成交</span><strong class="metric-value">{{ detail.execution_audit.actual_trades }}</strong></div>
          <div class="metric"><span class="metric-label">可配对信号</span><strong class="metric-value">{{ detail.execution_audit.signals_with_execution }}</strong></div>
          <div class="metric"><span class="metric-label">延迟成交</span><strong class="metric-value">{{ detail.execution_audit.delayed_executions }}</strong></div>
          <div class="metric"><span class="metric-label">T+1 异常</span><strong class="metric-value" :class="detail.execution_audit.invalid_t_plus_one ? 'negative' : ''">{{ detail.execution_audit.invalid_t_plus_one }}</strong></div>
          <div class="metric"><span class="metric-label">到期未成交</span><strong class="metric-value" :class="detail.execution_audit.unfilled_orders ? 'negative' : ''">{{ detail.execution_audit.unfilled_orders }}</strong></div>
          <div class="metric"><span class="metric-label">候选拒绝</span><strong class="metric-value">{{ detail.execution_audit.rejected_signals }}</strong></div>
        </div>
      </section>

      <section class="section">
        <div class="section-heading"><div><h2>交易事件</h2><p>同一笔交易分别列出信号日和成交日</p></div></div>
        <table class="data-table">
          <thead><tr><th style="width: 110px">日期</th><th style="width: 105px">关联日期</th><th style="width: 95px">层级</th><th style="width: 70px">动作</th><th class="number" style="width: 100px">价格</th><th class="number" style="width: 90px">数量</th><th style="width: 100px">市场状态</th><th class="number" style="width: 120px">单笔盈亏</th><th>原因</th></tr></thead>
          <tbody>
            <tr v-for="(event, index) in detail.events" :key="`${event.date}-${event.kind}-${event.action}-${index}`">
              <td>{{ formatDate(event.date) }}</td>
              <td>{{ formatDate(event.kind === 'execution' ? event.signal_date : event.execution_date) }}</td>
              <td><StatusBadge :value="event.kind === 'execution' ? event.delayed ? 'running' : 'completed' : ['rejected', 'unfilled'].includes(event.kind) ? 'failed' : 'pending'" :label="event.kind === 'execution' ? event.delayed ? '延迟成交' : '实际成交' : event.kind === 'signal' ? '策略信号' : statusLabel(event.kind)" /></td>
              <td :class="event.kind === 'rejected' ? '' : event.action === 'BUY' ? 'positive' : 'negative'">{{ event.action === 'BUY' ? '买入' : '卖出' }}</td>
              <td class="number">{{ formatNumber(event.price) }}</td>
              <td class="number">{{ event.quantity ?? '--' }}</td>
              <td>{{ event.market_state ?? '--' }}</td>
              <td class="number" :class="valueClass(event.pnl)">{{ formatMoney(event.pnl) }}</td>
              <td class="reason-cell">{{ event.reason }}</td>
            </tr>
            <tr v-if="!detail.events.length"><td class="empty-row" colspan="9">所选数据源没有该股票的交易事件</td></tr>
          </tbody>
        </table>
      </section>
    </template>
  </div>
</template>

<style scoped>
.stock-toolbar { display: grid; grid-template-columns: minmax(380px, 1fr) 360px auto; align-items: center; gap: 12px; padding-top: 0; }
.stock-toolbar > select { height: 38px; padding: 0 9px; border: 1px solid var(--line-strong); border-radius: 4px; background: white; }
.stock-search { position: relative; height: 38px; display: grid; grid-template-columns: 30px 1fr; align-items: center; border: 1px solid var(--line-strong); background: white; border-radius: 4px; padding: 0 8px; }
.stock-search svg { color: var(--muted); }
.stock-search input { border: 0; outline: 0; height: 100%; min-width: 0; }
.search-results { position: absolute; z-index: 30; top: 42px; left: 0; right: 0; border: 1px solid var(--line); background: white; box-shadow: 0 10px 24px rgba(35, 52, 58, 0.12); }
.search-results button { width: 100%; height: 40px; padding: 0 10px; display: grid; grid-template-columns: 95px 1fr 110px; align-items: center; text-align: left; border: 0; border-bottom: 1px solid var(--line); background: white; }
.search-results button:hover { background: #f4f8f9; }
.search-results span, .search-results small { color: var(--muted); }
.chart-panel { padding: 5px 0 0; }
.audit-strip { grid-template-columns: repeat(6, 1fr); }
</style>
