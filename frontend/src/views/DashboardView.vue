<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { AlertTriangle, Eye, EyeOff, RefreshCw } from '@lucide/vue'
import { api } from '../api'
import LoadingState from '../components/LoadingState.vue'
import StatusBadge from '../components/StatusBadge.vue'
import { formatDate, formatMoney, formatNumber, formatPct, formatTimestamp, statusLabel, valueClass } from '../format'
import type { DashboardData } from '../types'

const data = ref<DashboardData | null>(null)
const loading = ref(true)
const error = ref('')
const revealAmounts = ref(false)

const account = computed(() => data.value?.account)

async function load() {
  loading.value = true
  error.value = ''
  try {
    data.value = await api.dashboard()
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : '读取交易台失败'
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>

<template>
  <div>
    <div class="page-header">
      <div>
        <h1 class="page-title">今日交易台</h1>
        <p class="page-subtitle">
          数据日 {{ formatDate(data?.data_trade_date) }} · 信号日 {{ formatDate(data?.signal_trade_date) }} · 计划交易日 {{ formatDate(data?.planned_trade_date) }} · 更新 {{ formatTimestamp(data?.updated_at) }}
        </p>
      </div>
      <div class="header-actions">
        <StatusBadge v-if="data" :value="data.warnings.length ? 'failed' : 'completed'" :label="data.warnings.length ? `数据告警 ${data.warnings.length}` : '数据正常'" />
        <StatusBadge v-if="data" :value="data.market_state" :label="data.market_state_label" />
        <button class="button icon-only" title="刷新数据" :disabled="loading" @click="load">
          <RefreshCw :size="16" />
        </button>
      </div>
    </div>

    <LoadingState v-if="loading" label="正在汇总模拟盘与最新策略信号" />
    <div v-else-if="error" class="error-state">{{ error }}</div>
    <template v-else-if="data">
      <div v-if="data.warnings.length" class="notice-stack">
        <div v-for="warning in data.warnings" :key="warning" class="notice">
          <AlertTriangle :size="16" />
          <span>{{ warning }}</span>
        </div>
      </div>

      <section class="section">
        <div class="section-heading">
          <div>
            <h2>模拟账户</h2>
            <p>金额默认隐藏，收益率始终可见</p>
          </div>
          <button class="button" @click="revealAmounts = !revealAmounts">
            <component :is="revealAmounts ? EyeOff : Eye" :size="15" />
            {{ revealAmounts ? '隐藏金额' : '显示金额' }}
          </button>
        </div>
        <div class="metric-strip">
          <div class="metric">
            <span class="metric-label">总资产</span>
            <strong class="metric-value">{{ formatMoney(account?.equity, !revealAmounts) }}</strong>
            <div class="metric-note">初始 {{ formatMoney(account?.initial_cash, !revealAmounts) }}</div>
          </div>
          <div class="metric">
            <span class="metric-label">累计收益</span>
            <strong class="metric-value" :class="valueClass(account?.total_return)">{{ formatPct(account?.total_return) }}</strong>
            <div class="metric-note">日线模拟账户</div>
          </div>
          <div class="metric">
            <span class="metric-label">当日盈亏</span>
            <strong class="metric-value" :class="valueClass(account?.daily_pnl)">{{ formatMoney(account?.daily_pnl, !revealAmounts) }}</strong>
            <div class="metric-note" :class="valueClass(account?.daily_return)">{{ formatPct(account?.daily_return) }}</div>
          </div>
          <div class="metric">
            <span class="metric-label">现金比例</span>
            <strong class="metric-value">{{ formatPct(account?.cash_ratio) }}</strong>
            <div class="metric-note">现金 {{ formatMoney(account?.cash, !revealAmounts) }}</div>
          </div>
          <div class="metric">
            <span class="metric-label">持仓市值</span>
            <strong class="metric-value">{{ formatMoney(account?.positions_market_value, !revealAmounts) }}</strong>
            <div class="metric-note">按最近完整日线估值</div>
          </div>
          <div class="metric">
            <span class="metric-label">持仓数量</span>
            <strong class="metric-value">{{ account?.position_count ?? 0 }}</strong>
            <div class="metric-note">最多 5 个策略仓位</div>
          </div>
        </div>
      </section>

      <section class="section">
        <div class="section-heading">
          <div><h2>当前持仓</h2><p>点击股票进入日 K 与交易事件复盘</p></div>
        </div>
        <table class="data-table">
          <thead><tr><th style="width: 210px">股票</th><th>买入日</th><th class="number">买入价</th><th class="number">现价</th><th class="number">数量</th><th class="number">浮盈亏</th><th class="number">收益率</th><th class="number">持有</th></tr></thead>
          <tbody>
            <tr v-for="position in data.positions" :key="position.symbol">
              <td><RouterLink class="symbol-link" :to="`/stock/${position.symbol}`">{{ position.symbol }}</RouterLink><span class="symbol-name">{{ position.name }}</span></td>
              <td>{{ formatDate(position.entry_date) }}</td>
              <td class="number">{{ formatNumber(position.entry_price) }}</td>
              <td class="number">{{ formatNumber(position.last_price) }}</td>
              <td class="number">{{ position.quantity }}</td>
              <td class="number" :class="valueClass(position.unrealized_pnl)">{{ formatMoney(position.unrealized_pnl, !revealAmounts) }}</td>
              <td class="number" :class="valueClass(position.unrealized_return)">{{ formatPct(position.unrealized_return) }}</td>
              <td class="number">{{ position.holding_days }} 日</td>
            </tr>
            <tr v-if="!data.positions.length"><td class="empty-row" colspan="8">当前没有持仓记录</td></tr>
          </tbody>
        </table>
      </section>

      <section class="section split-layout">
        <div>
          <div class="section-heading"><div><h2>今日交易计划</h2><p>信号与成交在个股详情中分层展示</p></div></div>
          <div class="panel">
            <div class="panel-header"><h3>买入计划</h3><StatusBadge value="pending" :label="`${data.buy_orders.length} 笔`" /></div>
            <table class="data-table" style="border: 0">
              <tbody>
                <tr v-for="order in data.buy_orders" :key="`buy-${order.symbol}`">
                  <td style="width: 190px"><RouterLink class="symbol-link" :to="`/stock/${order.symbol}`">{{ order.symbol }}</RouterLink><span class="symbol-name">{{ order.name }}</span></td>
                  <td class="number" style="width: 100px">{{ formatNumber(order.limit_price) }}</td>
                  <td class="reason-cell">{{ order.reason }}</td>
                </tr>
                <tr v-if="!data.buy_orders.length"><td class="empty-row" colspan="3">今日无买入计划</td></tr>
              </tbody>
            </table>
          </div>
          <div class="panel" style="margin-top: 12px">
            <div class="panel-header"><h3>卖出计划</h3><StatusBadge value="pending" :label="`${data.sell_orders.length} 笔`" /></div>
            <table class="data-table" style="border: 0">
              <tbody>
                <tr v-for="order in data.sell_orders" :key="`sell-${order.symbol}`">
                  <td style="width: 190px"><RouterLink class="symbol-link" :to="`/stock/${order.symbol}`">{{ order.symbol }}</RouterLink><span class="symbol-name">{{ order.name }}</span></td>
                  <td class="number" style="width: 100px">{{ formatNumber(order.limit_price) }}</td>
                  <td class="reason-cell">{{ order.reason }}</td>
                </tr>
                <tr v-if="!data.sell_orders.length"><td class="empty-row" colspan="3">今日无卖出计划</td></tr>
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <div class="section-heading"><div><h2>策略备注</h2><p>来自最新计划，不在网页中改写</p></div></div>
          <div class="panel"><div class="panel-body"><ul class="notes-list"><li v-for="note in data.notes" :key="note">{{ note }}</li></ul></div></div>
        </div>
      </section>

      <section class="section">
        <div class="section-heading"><div><h2>Top20 买入候选</h2><p>解释排名、市场门控和最终未入选原因</p></div></div>
        <table class="data-table">
          <thead><tr><th style="width: 64px">排名</th><th style="width: 220px">股票</th><th style="width: 120px">行业</th><th class="number" style="width: 90px">评分</th><th style="width: 105px">状态</th><th>结果说明</th></tr></thead>
          <tbody>
            <tr v-for="candidate in data.candidates" :key="candidate.symbol">
              <td>{{ candidate.rank == null ? '--' : `#${candidate.rank}` }}</td>
              <td><RouterLink class="symbol-link" :to="`/stock/${candidate.symbol}`">{{ candidate.symbol }}</RouterLink><span class="symbol-name">{{ candidate.name }}</span></td>
              <td>{{ candidate.industry }}</td>
              <td class="number">{{ formatNumber(candidate.score, 4) }}</td>
              <td><StatusBadge :value="candidate.status" :label="statusLabel(candidate.status)" /></td>
              <td class="reason-cell">{{ candidate.rejected_reason ?? '进入最终交易计划' }}</td>
            </tr>
          </tbody>
        </table>
      </section>
    </template>
  </div>
</template>

<style scoped>
.notes-list { margin: 0; padding-left: 18px; color: var(--muted); font-size: 12px; line-height: 20px; }
.notes-list li + li { margin-top: 8px; }
</style>
