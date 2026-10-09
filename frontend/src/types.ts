export type Numeric = number | null

export interface Position {
  symbol: string
  name: string
  entry_date: string
  entry_price: number
  quantity: number
  last_price: number
  market_value: number
  unrealized_pnl: number
  unrealized_return: number
  holding_days: number
}

export interface Order {
  action: string
  symbol: string
  name: string
  limit_price: Numeric
  quantity: number | null
  rank: number | null
  score: Numeric
  reason: string
}

export interface Candidate {
  symbol: string
  name: string
  industry: string
  rank: number | null
  score: Numeric
  signal_type: string
  market_state: string
  status: string
  rejected_reason: string | null
}

export interface DashboardData {
  data_trade_date: string
  signal_trade_date: string
  planned_trade_date: string
  updated_at: string
  market_state: string
  market_state_label: string
  account: {
    initial_cash: Numeric
    cash: Numeric
    equity: Numeric
    positions_market_value: Numeric
    daily_pnl: Numeric
    daily_return: Numeric
    total_return: Numeric
    cash_ratio: Numeric
    position_count: number
  }
  positions: Position[]
  buy_orders: Order[]
  sell_orders: Order[]
  hold_orders: Order[]
  notes: string[]
  candidates: Candidate[]
  warnings: string[]
}

export interface Bar {
  trade_date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  amount: number
  ma5: Numeric
  ma10: Numeric
  ma20: Numeric
  ma60: Numeric
}

export interface TradeEvent {
  date: string
  kind: 'signal' | 'execution' | 'pending' | 'unfilled' | 'rejected'
  action: 'BUY' | 'SELL'
  price: Numeric
  limit_price?: Numeric
  quantity: number | null
  reason: string
  market_state?: string
  pnl?: Numeric
  signal_date?: string
  execution_date?: string
  delayed?: boolean
}

export interface StockDetail {
  symbol: string
  name: string
  industry: string | null
  adjustment: string
  range: string
  bars: Bar[]
  events: TradeEvent[]
  execution_audit: {
    actual_trades: number
    signals_with_execution: number
    delayed_executions: number
    invalid_t_plus_one: number
    t_plus_one_valid: boolean
    unfilled_orders: number
    pending_orders: number
    rejected_signals: number
  }
  warnings: string[]
}

export interface Metrics {
  total_return: Numeric
  annual_return: Numeric
  max_drawdown: Numeric
  sharpe: Numeric
  calmar: Numeric
  turnover: Numeric
  win_rate: Numeric
  profit_factor: Numeric
  average_holding_days: Numeric
  average_invested_ratio: Numeric
  sell_trade_count: Numeric
}

export interface ResearchResult {
  id: string
  title: string
  kind: string
  strategy: string
  source: string
  status: string
  start_date: string | null
  end_date: string | null
  summary_path: string
  equity_path: string | null
  trades_path: string | null
  metrics: Metrics
  parameters: Record<string, unknown>
  artifacts: Record<string, string>
  command: string | null
  protected: boolean
  archived: boolean
  created_at: string
  updated_at: string
}

export interface ResearchTask {
  id: string
  result_id: string | null
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  parameters: Record<string, unknown>
  command: string[]
  log_path: string
  created_at: string
  started_at: string | null
  finished_at: string | null
  error: string | null
  cancel_requested: boolean
}
