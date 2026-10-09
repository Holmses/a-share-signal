export function formatDate(value?: string | null): string {
  if (!value) return '--'
  const text = String(value).replaceAll('-', '')
  return /^\d{8}$/.test(text) ? `${text.slice(0, 4)}-${text.slice(4, 6)}-${text.slice(6)}` : String(value)
}

export function formatTimestamp(value?: string | null): string {
  if (!value) return '--'
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime())
    ? String(value)
    : parsed.toLocaleString('zh-CN', { hour12: false })
}

export function formatMoney(value?: number | null, hidden = false): string {
  if (hidden) return '••••••'
  if (value == null || !Number.isFinite(value)) return '--'
  return new Intl.NumberFormat('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value)
}

export function formatPct(value?: number | null, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return '--'
  return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(digits)}%`
}

export function formatNumber(value?: number | null, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return '--'
  return value.toFixed(digits)
}

export function valueClass(value?: number | null): string {
  if (value == null || value === 0) return ''
  return value > 0 ? 'positive' : 'negative'
}

export function statusLabel(value: string): string {
  return {
    planned: '计划买入',
    held: '当前持有',
    blocked: '市场拦截',
    ranked_out: '未入选',
    queued: '排队中',
    running: '运行中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消',
    rejected: '未成交',
    unfilled: '未成交',
  }[value] ?? value
}
