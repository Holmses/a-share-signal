import type { DashboardData, ResearchResult, ResearchTask, StockDetail } from './types'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(options?.headers ?? {}) },
    ...options,
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(payload.detail ?? `请求失败 (${response.status})`)
  }
  return response.json() as Promise<T>
}

export const api = {
  dashboard: () => request<DashboardData>('/api/dashboard'),
  dataHealth: () => request<Record<string, unknown>>('/api/data-health'),
  searchStocks: (query: string) => request<Array<Record<string, string>>>(`/api/stocks/search?q=${encodeURIComponent(query)}`),
  stock: (symbol: string, range = '1y', resultId?: string) => {
    const params = new URLSearchParams({ range })
    if (resultId) params.set('result_id', resultId)
    return request<StockDetail>(`/api/stocks/${encodeURIComponent(symbol)}?${params}`)
  },
  results: (includeArchived = false) => request<ResearchResult[]>(`/api/results?include_archived=${includeArchived}`),
  result: (id: string) => request<ResearchResult & { attribution: Array<Record<string, unknown>> }>(`/api/results/${id}`),
  resultTrades: (id: string, limit = 100) => request<Array<Record<string, unknown>>>(`/api/results/${id}/trades?limit=${limit}`),
  compare: (ids: string[], benchmark: string) => request<Record<string, unknown>>(`/api/compare?ids=${encodeURIComponent(ids.join(','))}&benchmark=${encodeURIComponent(benchmark)}`),
  reindex: () => request<{ scanned: number; indexed: number; failures: number }>('/api/results/reindex', { method: 'POST' }),
  createBacktest: (parameters: Record<string, unknown>) => request<{ tasks: ResearchTask[] }>('/api/backtests', { method: 'POST', body: JSON.stringify(parameters) }),
  tasks: () => request<ResearchTask[]>('/api/tasks'),
  taskLog: async (id: string) => {
    const response = await fetch(`/api/tasks/${id}/log`)
    if (!response.ok) throw new Error('无法读取任务日志')
    return response.text()
  },
  cancelTask: (id: string) => request<ResearchTask>(`/api/tasks/${id}/cancel`, { method: 'POST' }),
  archiveResult: (id: string, archived: boolean) => request(`/api/results/${id}/archive?archived=${archived}`, { method: 'POST' }),
  deleteResult: (id: string) => request(`/api/results/${id}?confirm=${encodeURIComponent(id)}`, { method: 'DELETE' }),
}
