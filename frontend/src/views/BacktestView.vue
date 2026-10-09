<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { Play, RotateCcw } from '@lucide/vue'
import { api } from '../api'
import StatusBadge from '../components/StatusBadge.vue'
import { formatDate, statusLabel } from '../format'
import type { ResearchTask } from '../types'

const baseline = {
  top_n: 5,
  max_positions: 5,
  market_min_breadth: 0.5,
  market_min_return_20d: 0,
  aggressive_position_size_multiplier: 0.5,
  hard_exit_days: 23,
  exit_profile: 'legacy',
  winner_bypass_peak_pct: null as number | null,
  risk_off_failed_days: null as number | null,
  high_drawdown_pct: null as number | null,
  chandelier_atr_multiplier: null as number | null,
  trend_decay: false,
}

const form = reactive({ start_date: '20240102', end_date: '20260630', ...baseline })
const submitting = ref(false)
const error = ref('')
const submitted = ref<ResearchTask[]>([])
const enableWinnerBypass = ref(false)
const enableRiskOff = ref(false)
const enableHighDrawdown = ref(false)
const enableChandelier = ref(false)

const differences = computed(() => Object.entries(baseline).filter(([key, value]) => form[key as keyof typeof form] !== value))

function reset() {
  Object.assign(form, baseline)
  enableWinnerBypass.value = false
  enableRiskOff.value = false
  enableHighDrawdown.value = false
  enableChandelier.value = false
}

async function submit() {
  submitting.value = true
  error.value = ''
  try {
    const payload = {
      ...form,
      winner_bypass_peak_pct: enableWinnerBypass.value ? form.winner_bypass_peak_pct ?? 0.08 : null,
      risk_off_failed_days: enableRiskOff.value ? form.risk_off_failed_days ?? 12 : null,
      high_drawdown_pct: enableHighDrawdown.value ? form.high_drawdown_pct ?? 0.1 : null,
      chandelier_atr_multiplier: enableChandelier.value ? form.chandelier_atr_multiplier ?? 3 : null,
    }
    const result = await api.createBacktest(payload)
    submitted.value = result.tasks
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : '无法创建回测任务'
  } finally {
    submitting.value = false
  }
}

onMounted(async () => {
  const dashboard = await api.dashboard().catch(() => null)
  if (dashboard?.data_trade_date) form.end_date = dashboard.data_trade_date
})
</script>

<template>
  <div>
    <div class="page-header">
      <div><h1 class="page-title">回测实验</h1><p class="page-subtitle">白名单核心参数 · research-only · 单任务后台队列</p></div>
      <button class="button" @click="reset"><RotateCcw :size="15" />恢复基线</button>
    </div>

    <section class="section split-layout backtest-layout">
      <form class="panel" @submit.prevent="submit">
        <div class="panel-header"><h3>实验参数</h3><StatusBadge value="pending" label="不会修改生产配置" /></div>
        <div class="panel-body">
          <div class="field-grid">
            <div class="field"><label>开始日期</label><input v-model="form.start_date" pattern="\d{8}" required /></div>
            <div class="field"><label>结束日期</label><input v-model="form.end_date" pattern="\d{8}" required /></div>
            <div class="field"><label>TopN</label><input v-model.number="form.top_n" type="number" min="1" max="20" /></div>
            <div class="field"><label>最大持仓</label><input v-model.number="form.max_positions" type="number" min="1" max="10" /></div>
            <div class="field"><label>市场广度阈值</label><input v-model.number="form.market_min_breadth" type="number" min="0" max="1" step="0.01" /></div>
            <div class="field"><label>市场20日收益阈值</label><input v-model.number="form.market_min_return_20d" type="number" min="-0.3" max="0.3" step="0.01" /></div>
            <div class="field"><label>Aggressive 仓位系数</label><input v-model.number="form.aggressive_position_size_multiplier" type="number" min="0.05" max="1" step="0.05" /></div>
            <div class="field"><label>固定退出交易日</label><input v-model.number="form.hard_exit_days" type="number" min="1" max="120" /></div>
            <div class="field"><label>退出档案</label><select v-model="form.exit_profile"><option value="legacy">Legacy 分级移动止盈</option><option value="slow_profit_lock">Slow profit lock</option></select></div>
            <div class="field"><label>趋势衰减退出</label><div class="toggle-row"><input v-model="form.trend_decay" type="checkbox" />启用 MA20 / 均线结构衰减</div></div>
          </div>

          <div class="exit-options">
            <div class="option-row"><label><input v-model="enableWinnerBypass" type="checkbox" />赢家免固定退出</label><input v-model.number="form.winner_bypass_peak_pct" :disabled="!enableWinnerBypass" type="number" min="0.01" max="1" step="0.01" placeholder="0.08" /></div>
            <div class="option-row"><label><input v-model="enableRiskOff" type="checkbox" />Risk-off 失败仓退出</label><input v-model.number="form.risk_off_failed_days" :disabled="!enableRiskOff" type="number" min="2" max="60" placeholder="12" /></div>
            <div class="option-row"><label><input v-model="enableHighDrawdown" type="checkbox" />近期高点回撤退出</label><input v-model.number="form.high_drawdown_pct" :disabled="!enableHighDrawdown" type="number" min="0.01" max="0.5" step="0.01" placeholder="0.10" /></div>
            <div class="option-row"><label><input v-model="enableChandelier" type="checkbox" />ATR Chandelier</label><input v-model.number="form.chandelier_atr_multiplier" :disabled="!enableChandelier" type="number" min="0.5" max="10" step="0.5" placeholder="3.0" /></div>
          </div>

          <div class="form-footer">
            <span class="form-error">{{ error }}</span>
            <button class="button primary" :disabled="submitting" type="submit"><Play :size="15" />{{ submitting ? '正在排队' : '启动研究回测' }}</button>
          </div>
        </div>
      </form>

      <div>
        <div class="panel">
          <div class="panel-header"><h3>相对基线差异</h3><span class="small muted">{{ differences.length }} 项</span></div>
          <div class="panel-body">
            <table v-if="differences.length" class="diff-table">
              <tr v-for="([key, baseValue]) in differences" :key="key"><td>{{ key }}</td><td class="mono">{{ baseValue ?? '关闭' }}</td><td>→</td><td class="mono">{{ form[key as keyof typeof form] ?? '关闭' }}</td></tr>
            </table>
            <div v-else class="empty-compact">当前参数与基线一致</div>
          </div>
        </div>
        <div v-if="submitted.length" class="panel" style="margin-top: 12px">
          <div class="panel-header"><h3>已创建任务</h3><RouterLink class="symbol-link" to="/tasks">查看队列</RouterLink></div>
          <div class="panel-body task-list">
            <div v-for="task in submitted" :key="task.id"><StatusBadge :value="task.status" :label="statusLabel(task.status)" /><span class="mono">{{ task.id }}</span><span>{{ formatDate(String(task.parameters.start_date)) }} - {{ formatDate(String(task.parameters.end_date)) }}</span></div>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.backtest-layout { grid-template-columns: minmax(720px, 1.5fr) minmax(380px, .7fr); padding-top: 0; }
.exit-options { margin-top: 18px; border-top: 1px solid var(--line); }
.option-row { min-height: 48px; display: grid; grid-template-columns: 1fr 130px; align-items: center; gap: 12px; border-bottom: 1px solid var(--line); font-size: 12px; }
.option-row label { display: flex; align-items: center; gap: 8px; }
.option-row input[type='checkbox'] { accent-color: var(--accent); }
.option-row > input { height: 34px; padding: 0 8px; border: 1px solid var(--line-strong); border-radius: 4px; }
.diff-table { width: 100%; border-collapse: collapse; font-size: 11px; }
.diff-table td { padding: 8px 5px; border-bottom: 1px solid var(--line); }
.diff-table td:nth-child(2), .diff-table td:nth-child(4) { text-align: right; }
.empty-compact { padding: 32px 0; text-align: center; color: var(--muted); font-size: 12px; }
.task-list { display: grid; gap: 10px; }
.task-list > div { display: grid; grid-template-columns: 72px 120px 1fr; align-items: center; gap: 8px; font-size: 11px; }
</style>
