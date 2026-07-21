<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { Archive, ChevronLeft, ChevronRight, Copy, FileClock, RefreshCw, RotateCcw, Search, Square, Trash2 } from '@lucide/vue'
import { api } from '../api'
import LoadingState from '../components/LoadingState.vue'
import StatusBadge from '../components/StatusBadge.vue'
import { formatDate, formatTimestamp, statusLabel } from '../format'
import type { ResearchResult, ResearchTask } from '../types'

interface BenchmarkHealth { code: string; name: string; available: boolean; start_date: string | null; end_date: string | null; updated_at: string | null }
interface ReportIndexHealth { scanned: number; indexed: number; failures: number; issues: Array<{ path: string; reason: string }> }
interface DataHealth { latest_daily_date?: string; daily_file_count?: number; daily_updated_at?: string; calendar_updated_at?: string; state_updated_at?: string; plan_updated_at?: string; reports_updated_at?: string; result_count?: number; report_index?: ReportIndexHealth; benchmarks?: BenchmarkHealth[]; warnings?: string[] }

const tasks = ref<ResearchTask[]>([])
const results = ref<ResearchResult[]>([])
const health = ref<DataHealth>({})
const loading = ref(true)
const error = ref('')
const selectedTask = ref<ResearchTask | null>(null)
const log = ref('')
const resultQuery = ref('')
const resultPage = ref(1)
const resultPageSize = 40
let timer: number | null = null

const selectedResult = computed(() => results.value.find((item) => item.id === selectedTask.value?.result_id) ?? null)
const filteredResults = computed(() => {
  const query = resultQuery.value.trim().toLowerCase()
  if (!query) return results.value
  return results.value.filter((item) => [item.id, item.title, item.kind, item.strategy].some((value) => String(value).toLowerCase().includes(query)))
})
const resultPageCount = computed(() => Math.max(1, Math.ceil(filteredResults.value.length / resultPageSize)))
const pagedResults = computed(() => filteredResults.value.slice((resultPage.value - 1) * resultPageSize, resultPage.value * resultPageSize))

async function load() {
  try {
    const [taskRows, resultRows, healthData] = await Promise.all([api.tasks(), api.results(true), api.dataHealth()])
    tasks.value = taskRows
    results.value = resultRows
    health.value = healthData as DataHealth
    if (selectedTask.value) {
      selectedTask.value = taskRows.find((task) => task.id === selectedTask.value?.id) ?? null
      if (selectedTask.value) log.value = await api.taskLog(selectedTask.value.id)
    }
    error.value = ''
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : '读取任务状态失败'
  } finally {
    loading.value = false
  }
}

async function refreshTasks() {
  try {
    const taskRows = await api.tasks()
    const needsResultRefresh = taskRows.some(
      (task) => task.result_id && !results.value.some((result) => result.id === task.result_id),
    )
    tasks.value = taskRows
    if (selectedTask.value) {
      selectedTask.value = taskRows.find((task) => task.id === selectedTask.value?.id) ?? null
      if (selectedTask.value) log.value = await api.taskLog(selectedTask.value.id)
    }
    if (needsResultRefresh) {
      const [resultRows, healthData] = await Promise.all([api.results(true), api.dataHealth()])
      results.value = resultRows
      health.value = healthData as DataHealth
    }
    error.value = ''
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : '刷新任务状态失败'
  }
}

async function inspect(task: ResearchTask) { selectedTask.value = task; log.value = await api.taskLog(task.id) }
async function cancel(task: ResearchTask) { await api.cancelTask(task.id); await load() }
async function reindex() { await api.reindex(); await load() }
async function archive(item: ResearchResult) { await api.archiveResult(item.id, !item.archived); await load() }
async function remove(item: ResearchResult) {
  if (!window.confirm(`确认删除研究结果 ${item.id} 及其关联文件？`)) return
  await api.deleteResult(item.id)
  await load()
}

function shellCommand(task: ResearchTask) {
  return task.command.map((part) => /^[A-Za-z0-9_./:=,@%+-]+$/.test(part) ? part : `'${part.replaceAll("'", `'\\''`)}'`).join(' ')
}

async function copyCommand(task: ResearchTask) {
  const value = shellCommand(task)
  if (navigator.clipboard) {
    try {
      await navigator.clipboard.writeText(value)
      return
    } catch {
      // Insecure LAN origins can expose the API but reject clipboard writes.
    }
  }
  const textarea = document.createElement('textarea')
  textarea.value = value
  textarea.style.position = 'fixed'
  textarea.style.opacity = '0'
  document.body.appendChild(textarea)
  textarea.select()
  document.execCommand('copy')
  textarea.remove()
}

onMounted(async () => { await load(); timer = window.setInterval(refreshTasks, 3000) })
onBeforeUnmount(() => { if (timer) window.clearInterval(timer) })
watch(resultQuery, () => { resultPage.value = 1 })
watch(resultPageCount, (count) => { if (resultPage.value > count) resultPage.value = count })
</script>

<template>
  <div>
    <div class="page-header">
      <div><h1 class="page-title">任务与数据</h1><p class="page-subtitle">串行研究队列、产物索引和数据源健康状态</p></div>
      <div class="header-actions"><button class="button" @click="reindex"><RotateCcw :size="15" />重建索引</button><button class="button icon-only" title="刷新" @click="load"><RefreshCw :size="15" /></button></div>
    </div>
    <LoadingState v-if="loading" />
    <div v-else-if="error" class="error-state">{{ error }}</div>
    <template v-else>
      <div v-if="health.warnings?.length" class="notice-stack"><div v-for="warning in health.warnings" :key="warning" class="notice">{{ warning }}</div></div>

      <section class="section">
        <div class="section-heading"><div><h2>数据状态</h2><p>网页只读，不接管现有 scheduler</p></div></div>
        <div class="metric-strip health-strip">
          <div class="metric"><span class="metric-label">最新完整日线</span><strong class="metric-value health-value">{{ formatDate(health.latest_daily_date) }}</strong><div class="metric-note">{{ health.daily_file_count ?? 0 }} 个文件 · {{ formatTimestamp(health.daily_updated_at) }}</div></div>
          <div class="metric"><span class="metric-label">交易日历</span><strong class="metric-value health-value">{{ health.calendar_updated_at ? '可读取' : '缺失' }}</strong><div class="metric-note">{{ formatTimestamp(health.calendar_updated_at) }}</div></div>
          <div class="metric"><span class="metric-label">模拟盘状态</span><strong class="metric-value health-value">{{ health.state_updated_at ? '可读取' : '缺失' }}</strong><div class="metric-note">{{ formatTimestamp(health.state_updated_at) }}</div></div>
          <div class="metric"><span class="metric-label">最新计划</span><strong class="metric-value health-value">{{ health.plan_updated_at ? '可读取' : '缺失' }}</strong><div class="metric-note">{{ formatTimestamp(health.plan_updated_at) }}</div></div>
          <div class="metric"><span class="metric-label">报告目录</span><strong class="metric-value health-value">{{ health.reports_updated_at ? '可读取' : '缺失' }}</strong><div class="metric-note">{{ formatTimestamp(health.reports_updated_at) }}</div></div>
          <div class="metric"><span class="metric-label">结果索引</span><strong class="metric-value health-value">{{ health.result_count ?? 0 }}</strong><div class="metric-note">{{ health.report_index?.indexed ?? 0 }}/{{ health.report_index?.scanned ?? 0 }} 已索引 · {{ health.report_index?.failures ?? 0 }} 个跳过</div></div>
        </div>
        <details v-if="health.report_index?.issues.length" class="index-issues panel">
          <summary>查看 {{ health.report_index.issues.length }} 个未索引报告</summary>
          <div v-for="issue in health.report_index.issues" :key="issue.path"><span class="mono">{{ issue.path }}</span><strong>{{ issue.reason }}</strong></div>
        </details>
        <table class="data-table benchmark-table">
          <thead><tr><th>基准</th><th>代码</th><th>状态</th><th>起始日</th><th>结束日</th><th>更新时间</th></tr></thead>
          <tbody><tr v-for="item in health.benchmarks" :key="item.code"><td>{{ item.name }}</td><td class="mono">{{ item.code }}</td><td><StatusBadge :value="item.available ? 'completed' : 'failed'" :label="item.available ? '已缓存' : '缺失'" /></td><td>{{ formatDate(item.start_date) }}</td><td>{{ formatDate(item.end_date) }}</td><td class="small muted">{{ item.updated_at ?? '--' }}</td></tr></tbody>
        </table>
      </section>

      <section class="section split-layout task-layout">
        <div>
          <div class="section-heading"><div><h2>后台研究任务</h2><p>同一时间只运行一个全 A 回测</p></div></div>
          <table class="data-table"><thead><tr><th style="width:130px">任务</th><th style="width:95px">状态</th><th>进度</th><th style="width:185px">创建时间</th><th style="width:80px">操作</th></tr></thead><tbody>
            <tr v-for="task in tasks" :key="task.id" @click="inspect(task)"><td class="mono">{{ task.id }}</td><td><StatusBadge :value="task.status" :label="statusLabel(task.status)" /></td><td><div class="progress-track"><div class="progress-bar" :style="{ width: `${task.progress}%` }"></div></div></td><td class="small">{{ task.created_at }}</td><td><button v-if="['queued','running'].includes(task.status)" class="button icon-only" title="取消任务" @click.stop="cancel(task)"><Square :size="14" /></button><button v-else class="button icon-only" title="查看日志" @click.stop="inspect(task)"><FileClock :size="14" /></button></td></tr>
            <tr v-if="!tasks.length"><td class="empty-row" colspan="5">还没有从网页启动的研究任务</td></tr>
          </tbody></table>
        </div>
        <div>
          <div class="section-heading"><div><h2>任务详情</h2><p>{{ selectedTask ? selectedTask.id : '选择左侧任务' }}</p></div><button v-if="selectedTask" class="button icon-only" title="复制复现命令" @click="copyCommand(selectedTask)"><Copy :size="14" /></button></div>
          <div v-if="selectedTask" class="task-detail panel">
            <dl class="task-meta">
              <div><dt>创建</dt><dd>{{ formatTimestamp(selectedTask.created_at) }}</dd></div>
              <div><dt>开始</dt><dd>{{ formatTimestamp(selectedTask.started_at) }}</dd></div>
              <div><dt>结束</dt><dd>{{ formatTimestamp(selectedTask.finished_at) }}</dd></div>
              <div><dt>日志</dt><dd class="mono">{{ selectedTask.log_path }}</dd></div>
            </dl>
            <div v-if="selectedTask.error" class="notice error task-error">{{ selectedTask.error }}</div>
            <div class="command-line mono">{{ shellCommand(selectedTask) }}</div>
            <div v-if="selectedResult" class="artifact-list">
              <div><strong>结果</strong><span class="mono">{{ selectedResult.summary_path }}</span></div>
              <div v-for="(path, name) in selectedResult.artifacts" :key="name"><strong>{{ name }}</strong><span class="mono">{{ path }}</span></div>
            </div>
            <pre class="log-view">{{ log || '暂无日志输出' }}</pre>
          </div>
          <div v-else class="panel empty-task-detail">选择任务后查看时间、复现命令、错误和产物路径</div>
        </div>
      </section>

      <section class="section">
        <div class="section-heading">
          <div><h2>结果索引</h2><p>导入历史结果只读；仅网页任务产物允许删除</p></div>
          <div class="result-tools">
            <label class="result-search"><Search :size="14" /><input v-model="resultQuery" aria-label="搜索结果" placeholder="结果、策略或 ID" /></label>
            <span class="small muted">{{ filteredResults.length }} 条 · {{ resultPage }}/{{ resultPageCount }}</span>
            <button class="button icon-only" title="上一页" :disabled="resultPage <= 1" @click="resultPage--"><ChevronLeft :size="14" /></button>
            <button class="button icon-only" title="下一页" :disabled="resultPage >= resultPageCount" @click="resultPage++"><ChevronRight :size="14" /></button>
          </div>
        </div>
        <table class="data-table"><thead><tr><th style="width:260px">结果</th><th style="width:120px">类型</th><th>区间</th><th style="width:90px">来源</th><th style="width:100px">状态</th><th style="width:110px">操作</th></tr></thead><tbody>
          <tr v-for="item in pagedResults" :key="item.id"><td class="truncate" :title="item.title">{{ item.title }}</td><td>{{ item.kind }}</td><td>{{ formatDate(item.start_date) }} - {{ formatDate(item.end_date) }}</td><td>{{ item.source === 'task' ? '网页任务' : '历史导入' }}</td><td><StatusBadge :value="item.archived ? 'cancelled' : 'completed'" :label="item.archived ? '已归档' : item.protected ? '受保护基线' : '可用'" /></td><td><div class="row-actions"><button class="button icon-only" :title="item.archived ? '取消归档' : '归档'" @click="archive(item)"><Archive :size="14" /></button><button class="button icon-only danger" title="删除" :disabled="item.protected || item.source !== 'task'" @click="remove(item)"><Trash2 :size="14" /></button></div></td></tr>
          <tr v-if="!pagedResults.length"><td colspan="6" class="empty-row">没有匹配的研究结果</td></tr>
        </tbody></table>
      </section>
    </template>
  </div>
</template>

<style scoped>
.health-strip { grid-template-columns: repeat(3, 1fr); margin-bottom: 12px; }
.health-strip .metric:nth-child(3) { border-right: 0; }
.health-strip .metric:nth-child(-n+3) { border-bottom: 1px solid var(--line); }
.health-value { font-size: 17px; }
.index-issues { margin-bottom: 12px; padding: 10px 13px; color: var(--muted); font-size: 11px; }
.index-issues summary { cursor: pointer; font-weight: 650; color: var(--text); }
.index-issues div { margin-top: 8px; display: grid; grid-template-columns: minmax(0, 1fr) 190px; gap: 12px; line-height: 16px; }
.index-issues span { overflow-wrap: anywhere; }
.index-issues strong { color: var(--rise); font-weight: 550; }
.benchmark-table th { width: auto; }
.task-layout { grid-template-columns: minmax(680px, 1.25fr) minmax(430px, .75fr); }
.task-layout tbody tr { cursor: pointer; }
.task-detail { overflow: hidden; }
.task-meta { margin: 0; padding: 12px 14px; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px 14px; border-bottom: 1px solid var(--line); }
.task-meta div { min-width: 0; }
.task-meta dt { color: var(--muted); font-size: 10px; }
.task-meta dd { margin: 3px 0 0; overflow-wrap: anywhere; font-size: 11px; }
.task-error { margin: 12px 14px 0; }
.command-line { margin: 12px 14px; padding: 9px 10px; max-height: 74px; overflow: auto; background: var(--surface-muted); color: var(--muted); font-size: 10px; line-height: 16px; overflow-wrap: anywhere; }
.artifact-list { margin: 0 14px 12px; display: grid; gap: 6px; }
.artifact-list div { display: grid; grid-template-columns: 62px minmax(0, 1fr); gap: 7px; font-size: 10px; line-height: 15px; }
.artifact-list span { color: var(--muted); overflow-wrap: anywhere; }
.task-detail .log-view { height: 230px; border-top: 1px solid var(--line); }
.empty-task-detail { height: 320px; display: grid; place-items: center; color: var(--muted); font-size: 12px; }
.result-tools { display: flex; align-items: center; gap: 7px; }
.result-search { height: 34px; width: 220px; display: grid; grid-template-columns: 25px 1fr; align-items: center; padding: 0 7px; border: 1px solid var(--line-strong); border-radius: 4px; background: white; color: var(--muted); }
.result-search input { min-width: 0; border: 0; outline: 0; font-size: 11px; }
.row-actions { display: flex; gap: 5px; }
</style>
