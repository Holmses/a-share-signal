# 模拟仓位运行说明

## 目标

解决 `docs/architecture-review.md` 中提到的持仓管理风险：

- 不再手工维护 `current_positions.csv`
- 自动保留持仓历史快照
- 自动计算最新持仓盈亏
- 每日手动运行即可得到：
  - 最新模拟仓位
  - 最新盈亏
  - 下一交易日信号

## 命令

推荐每日使用一键流水线：

```bash
ashare-signal run-daily \
  --config configs/strategy.toml.example
```

该命令会自动执行：

- 同步 Tushare 到当天
- 构建最新完整缓存交易日的 universe
- 从 `[runtime].paper_start_date` 开始重算模拟组合
- 写入持仓、盈亏、交易记录和下一交易日信号

如果只想复用本地缓存做验证：

```bash
ashare-signal run-daily \
  --config configs/strategy.toml.example \
  --end-date 2026-04-07 \
  --skip-sync
```

底层模拟仓位命令仍可单独运行：

```bash
ashare-signal paper-trade \
  --config configs/strategy.toml.example \
  --start-date 2026-02-02 \
  --end-date 2026-04-07
```

## Docker 定时运行

先确认 `.env` 中有 `TUSHARE_TOKEN`，再启动长驻定时容器：

```bash
docker compose up -d --build ashare-signal-daily
```

定时参数在 `configs/strategy.toml.example` 中：

```toml
[backtest]
initial_cash = 1000000

[runtime]
paper_start_date = "2026-02-02"
daily_run_time = "18:30"
timezone = "Asia/Shanghai"
sync_lookback_days = 7
calendar_ahead_days = 14
```

手动跑一次容器内完整流程：

```bash
docker compose run --rm ashare-signal-daily \
  run-daily \
  --config configs/strategy.toml.example
```

## 输出

- `data/positions/current_positions.csv`
  - 当前模拟持仓
- `data/positions/latest_pnl.csv`
  - 最新收盘价下的持仓盈亏
- `data/positions/current_state.json`
  - 当前模拟组合状态
- `data/positions/snapshots/YYYYMMDD.csv`
  - 每个交易日的持仓快照
- `data/positions/trades.csv`
  - 模拟成交记录
- `reports/generated/signal-board-YYYYMMDD.md`
  - 下一交易日信号板

## 运行建议

每天收盘后优先使用 `run-daily` 或启动 `ashare-signal-daily` 容器。`sync-tushare`、`build-universe`、`paper-trade` 仍保留为底层排查命令。

## 可靠性边界

- 模拟仓位来源于同一套回测逻辑，不依赖人工改仓
- 当前仍然是日频近似撮合，不是逐笔成交回放
- 当前未处理：
  - 涨跌停不可成交
  - 一字板买不到/卖不出
  - 最小佣金门槛
  - 滑点动态变化

这些属于下一阶段风控和撮合精细化任务
