<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { RefreshCw, Search } from '@lucide/vue'
import { api } from '../api'
import EquityChart from '../components/EquityChart.vue'
import LoadingState from '../components/LoadingState.vue'
import { formatDate, formatMoney, formatNumber, formatPct, valueClass } from '../format'
import type { ResearchResult } from '../types'

interface CurvePoint { date: string; value: number; drawdown?: number }
interface ComparePayload {
  results: ResearchResult[]
  series: Array<{ id: string; name: string; data: CurvePoint[] }>
  benchmark: { code: string; name: string; data: CurvePoint[] } | null
  warnings: string[]
}

const allResults = ref<ResearchResult[]>([])
const selectedIds = ref<string[]>([])
const benchmark = ref('000300.SH')
const mode = ref<'equity' | 'drawdown' | 'excess'>('equity')
const compare = ref<ComparePayload | null>(null)
const activeResultId = ref('')
const attribution = ref<Array<Record<string, unknown>>>([])
const trades = ref<Array<Record<string, unknown>>>([])
const dimension = ref('exit_reason')
const loading = ref(true)
const error = ref('')
const resultQuery = ref('')

const comparable = computed(() => {
  const query = resultQuery.value.trim().toLowerCase()
  return allResults.value.filter((item) => {
    if (!item.equity_path || item.metrics.total_return == null) return false
    return !query || [item.id, item.title, item.strategy, item.kind].some((value) => String(value).toLowerCase().includes(query))
  })
})
const activeAttribution = computed(() => attribution.value.filter((row) => row.dimension === dimension.value))

async function loadResults() {
  allResults.value = await api.results()
  if (!selectedIds.value.length) {
    const candidates = comparable.value
    const baseline = candidates.find((item) => item.protected)
    if (baseline) {
      selectedIds.value = [baseline.id]
      const challenger = candidates.find((item) => !item.protected && item.start_date === baseline.start_date && item.end_date === baseline.end_date)
      if (challenger) selectedIds.value.push(challenger.id)
    } else if (candidates[0]) selectedIds.value = [candidates[0].id]
  }
}

async function runCompare() {
  if (!selectedIds.value.length) return
  loading.value = true
  error.value = ''
  try {
    compare.value = await api.compare(selectedIds.value, benchmark.value) as unknown as ComparePayload
    if (!activeResultId.value || !compare.value.results.some((item) => item.id === activeResultId.value)) {
      activeResultId.value = compare.value.results[0]?.id ?? ''
    }
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : '策略比较失败'
  } finally {
    loading.value = false
  }
}

async function loadDetails() {
  if (!activeResultId.value) return
  const [result, resultTrades] = await Promise.all([
    api.result(activeResultId.value),
    api.resultTrades(activeResultId.value, 60),
  ])
  attribution.value = result.attribution
  trades.value = resultTrades
}

function toggleResult(id: string) {
  if (selectedIds.value.includes(id)) selectedIds.value = selectedIds.value.filter((value) => value !== id)
  else if (selectedIds.value.length < 4) selectedIds.value = [...selectedIds.value, id]
}

watch([selectedIds, benchmark], runCompare, { deep: true })
watch(activeResultId, loadDetails)
onMounted(async () => { await loadResults(); await runCompare(); await loadDetails() })
</script>

<template>
  <div>
    <div class="page-header">
      <div><h1 class="page-title">策略对比</h1><p class="page-subtitle">最多同时比较 4 个版本；系统自动补入同区间基线</p></div>
      <div class="header-actions">
        <select v-model="benchmark" class="compact-select" aria-label="选择基准">
          <option value="000300.SH">沪深300</option><option value="000905.SH">中证500</option><option value="000852.SH">中证1000</option><option value="399006.SZ">创业板指</option><option value="000688.SH">科创50</option>
        </select>
        <button class="button icon-only" title="刷新比较" @click="runCompare"><RefreshCw :size="15" /></button>
      </div>
    </div>

    <section class="section compare-grid">
      <aside class="panel result-picker">
        <div class="panel-header"><h3>历史策略版本</h3><span class="small muted">{{ selectedIds.length }}/4 · {{ comparable.length }} 条</span></div>
        <div class="panel-body result-picker-body">
          <label class="picker-search"><Search :size="14" /><input v-model="resultQuery" aria-label="搜索策略版本" placeholder="策略、区间或 ID" /></label>
          <div class="checkbox-list">
          <label v-for="item in comparable" :key="item.id" class="checkbox-item" :title="item.title">
            <input type="checkbox" :checked="selectedIds.includes(item.id)" :disabled="!selectedIds.includes(item.id) && selectedIds.length >= 4" @change="toggleResult(item.id)" />
            <span><strong>{{ item.protected ? '基线 · ' : '' }}{{ item.title }}</strong><br><span class="muted">{{ formatDate(item.start_date) }} - {{ formatDate(item.end_date) }} · {{ formatPct(item.metrics.total_return) }}</span></span>
          </label>
          <div v-if="!comparable.length" class="empty-picker">没有匹配的可比较结果</div>
          </div>
        </div>
      </aside>

      <div class="panel">
        <div class="panel-header">
          <h3>组合表现</h3>
          <div class="segmented"><button :class="{ active: mode === 'equity' }" @click="mode = 'equity'">净值</button><button :class="{ active: mode === 'drawdown' }" @click="mode = 'drawdown'">回撤</button><button :class="{ active: mode === 'excess' }" @click="mode = 'excess'">超额</button></div>
        </div>
        <LoadingState v-if="loading" label="正在对齐净值和基准日期" />
        <div v-else-if="error" class="error-state">{{ error }}</div>
        <EquityChart v-else-if="compare" :series="compare.series" :benchmark="compare.benchmark" :mode="mode" />
      </div>
    </section>

    <div v-if="compare?.warnings.length" class="notice-stack"><div v-for="warning in compare.warnings" :key="warning" class="notice">{{ warning }}</div></div>

    <section v-if="compare" class="section">
      <div class="section-heading"><div><h2>统一指标矩阵</h2><p>收益不能单独作为优劣判断</p></div></div>
      <table class="data-table metrics-table">
        <thead><tr><th>策略</th><th class="number">累计收益</th><th class="number">年化</th><th class="number">最大回撤</th><th class="number">Sharpe</th><th class="number">Calmar</th><th class="number">换手</th><th class="number">胜率</th><th class="number">Profit Factor</th><th class="number">持仓日</th><th class="number">资金利用</th></tr></thead>
        <tbody><tr v-for="item in compare.results" :key="item.id" :class="{ activeRow: item.id === activeResultId }" @click="activeResultId = item.id">
          <td><strong>{{ item.protected ? '基线 · ' : '' }}{{ item.strategy }}</strong><br><span class="small muted">{{ formatDate(item.start_date) }} - {{ formatDate(item.end_date) }}</span></td>
          <td class="number" :class="valueClass(item.metrics.total_return)">{{ formatPct(item.metrics.total_return) }}</td>
          <td class="number" :class="valueClass(item.metrics.annual_return)">{{ formatPct(item.metrics.annual_return) }}</td>
          <td class="number negative">{{ formatPct(item.metrics.max_drawdown) }}</td>
          <td class="number">{{ formatNumber(item.metrics.sharpe, 3) }}</td><td class="number">{{ formatNumber(item.metrics.calmar, 3) }}</td>
          <td class="number">{{ formatNumber(item.metrics.turnover) }}</td><td class="number">{{ formatPct(item.metrics.win_rate) }}</td>
          <td class="number">{{ formatNumber(item.metrics.profit_factor, 3) }}</td><td class="number">{{ formatNumber(item.metrics.average_holding_days, 1) }}</td><td class="number">{{ formatPct(item.metrics.average_invested_ratio) }}</td>
        </tr></tbody>
      </table>
    </section>

    <section v-if="activeResultId" class="section split-layout">
      <div>
        <div class="section-heading"><div><h2>交易归因</h2><p>点击矩阵中的策略切换归因对象</p></div><select v-model="dimension" class="compact-select"><option value="exit_reason">退出原因</option><option value="year">年份</option><option value="market_state">市场状态</option><option value="industry_style">行业风格</option></select></div>
        <table class="data-table"><thead><tr><th>分组</th><th class="number">交易数</th><th class="number">总盈亏</th><th class="number">平均盈亏</th><th class="number">胜率</th><th class="number">平均持仓</th></tr></thead><tbody>
          <tr v-for="row in activeAttribution" :key="String(row.group)"><td>{{ row.group }}</td><td class="number">{{ row.trades }}</td><td class="number" :class="valueClass(Number(row.total_pnl))">{{ formatMoney(Number(row.total_pnl)) }}</td><td class="number" :class="valueClass(Number(row.average_pnl))">{{ formatMoney(Number(row.average_pnl)) }}</td><td class="number">{{ formatPct(Number(row.win_rate)) }}</td><td class="number">{{ formatNumber(Number(row.average_holding_days), 1) }}</td></tr>
          <tr v-if="!activeAttribution.length"><td colspan="6" class="empty-row">该历史结果缺少此维度字段</td></tr>
        </tbody></table>
      </div>
      <div>
        <div class="section-heading"><div><h2>最近退出交易</h2><p>进入个股页面查看完整买卖点</p></div></div>
        <table class="data-table"><thead><tr><th style="width:100px">退出日</th><th>股票</th><th class="number" style="width:105px">盈亏</th><th style="width:120px">原因</th></tr></thead><tbody>
          <tr v-for="trade in trades" :key="`${trade.trade_date}-${trade.symbol}`"><td>{{ formatDate(String(trade.trade_date)) }}</td><td><RouterLink class="symbol-link" :to="{ name: 'stock', params: { symbol: String(trade.symbol) }, query: { result_id: activeResultId, trade_date: String(trade.trade_date) } }">{{ trade.symbol }}</RouterLink><span class="symbol-name">{{ trade.name }}</span></td><td class="number" :class="valueClass(Number(trade.pnl))">{{ formatMoney(Number(trade.pnl)) }}</td><td class="truncate" :title="String(trade.exit_reason ?? '')">{{ trade.exit_reason ?? '--' }}</td></tr>
          <tr v-if="!trades.length"><td colspan="4" class="empty-row">该结果没有可配对的退出交易</td></tr>
        </tbody></table>
      </div>
    </section>
  </div>
</template>

<style scoped>
.compact-select { height: 34px; padding: 0 9px; border: 1px solid var(--line-strong); border-radius: 4px; background: white; }
.compare-grid { display: grid; grid-template-columns: 320px minmax(0, 1fr); gap: 16px; padding-top: 0; }
.result-picker { height: 400px; }
.result-picker-body { display: grid; grid-template-rows: 34px minmax(0, 1fr); gap: 10px; height: 350px; }
.result-picker .checkbox-list { max-height: none; }
.picker-search { height: 34px; display: grid; grid-template-columns: 24px 1fr; align-items: center; padding: 0 7px; border: 1px solid var(--line-strong); border-radius: 4px; color: var(--muted); }
.picker-search input { min-width: 0; border: 0; outline: 0; font-size: 11px; }
.checkbox-item strong { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.empty-picker { padding: 36px 0; text-align: center; color: var(--muted); font-size: 11px; }
.metrics-table th:first-child { width: 230px; }
.metrics-table tbody tr { cursor: pointer; }
.metrics-table .activeRow { background: var(--accent-soft); }
</style>
