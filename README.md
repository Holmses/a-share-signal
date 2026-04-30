# A Share V1 Signal

面向 A 股日频信号板的量化项目骨架。

当前版本只覆盖 `V1 信号版` 的工程结构，不自动下单，不做盘中交易。程序目标是在每个交易日收盘后，给出下一交易日的 1 个买入候选和 1 个卖出候选，并维护一个固定 5 持仓的组合框架。

## 范围

- 市场：沪深 A 股普通股票
- 频率：日频
- 调仓：每天最多 1 买 1 卖
- 持仓：固定 5 只
- 输出：下一交易日建议委托价、买卖理由、风控备注

## 项目结构

```text
a-share-v1-signal/
├── configs/
├── data/
│   ├── processed/
│   └── raw/
├── logs/
├── reports/
│   └── generated/
├── src/
│   └── ashare_signal/
│       ├── backtest/
│       ├── data/
│       ├── features/
│       ├── portfolio/
│       ├── report/
│       ├── scheduler/
│       └── strategy/
└── tests/
```

## 快速开始

1. 创建虚拟环境并安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

2. 配置 Tushare Token

```bash
cp .env.example .env
```

CLI 会自动读取项目根目录下的 `.env`。

3. 同步 Tushare 原始数据

```bash
ashare-signal sync-tushare \
  --config configs/strategy.toml.example \
  --start-date 2026-01-01 \
  --end-date 2026-04-07
```

4. 生成过滤后的股票池与特征快照

```bash
ashare-signal build-universe \
  --config configs/strategy.toml.example \
  --as-of 2026-04-07
```

5. 基于当前持仓生成真实信号板

```bash
ashare-signal generate-signal \
  --config configs/strategy.toml.example \
  --holdings configs/current_positions.csv \
  --as-of 2026-04-07
```

6. 运行基础日频回测

```bash
ashare-signal backtest \
  --config configs/strategy.toml.example \
  --start-date 2026-02-02 \
  --end-date 2026-04-07
```

7. 同步模拟仓位并生成下一交易日信号

```bash
ashare-signal paper-trade \
  --config configs/strategy.toml.example \
  --start-date 2026-02-02 \
  --end-date 2026-04-07
```

8. 构建 Docker 镜像

```bash
docker build -t ashare-signal .
```

9. 用 Docker 运行

```bash
docker run --rm -it \
  -e TUSHARE_TOKEN="$TUSHARE_TOKEN" \
  -v "$PWD":/workspace \
  -w /workspace \
  ashare-signal sync-tushare \
  --config configs/strategy.toml.example \
  --start-date 2026-01-01 \
  --end-date 2026-04-07
```

10. 一次性执行每日完整流程

```bash
ashare-signal run-daily \
  --config configs/strategy.toml.example
```

这个命令会同步 Tushare、构建最新 universe、按 `[backtest].initial_cash` 重算模拟仓位，并生成下一交易日信号。

11. 用 Docker Compose 启动每日定时容器

```bash
docker compose up -d --build ashare-signal-daily
```

默认读取 `configs/strategy.toml.example` 中的 `[runtime].daily_run_time`，按北京时间每天执行一次。

## 当前命令

- `ashare-signal sync-tushare`
  - 接入 Tushare `trade_cal`、`stock_basic`、`daily`、`daily_basic`，并写入本地原始缓存
- `ashare-signal build-universe`
  - 从本地缓存构建过滤后的股票池和特征快照，结果输出到 `data/processed/universe/`
- `ashare-signal generate-signal`
  - 从本地 universe 快照和当前持仓 CSV 生成一份真实 Markdown 信号板
- `ashare-signal backtest`
  - 基于缓存日线和当前选股规则运行真实日频 T+1 回测
- `ashare-signal paper-trade`
  - 基于同一套回测逻辑同步模拟仓位、持仓快照、最新盈亏和下一交易日信号
- `ashare-signal run-daily`
  - 每日完整流水线：同步数据、构建最新股票池、更新模拟仓位、生成信号板
- `ashare-signal run-scheduler`
  - 容器内长驻定时器，按配置时间每天触发 `run-daily` 流水线

## 当前已实现的数据与特征

- `trade_cal`
  - 用于解析开市日，并在非交易日自动回退到最近交易日
- `stock_basic`
  - 用于股票基础信息、上市日期、交易所、市场分类
- `daily`
  - 用于日线价格、成交额、动量、均线、波动率和流动性特征
- `daily_basic`
  - 用于换手率、量比、市值和估值类字段

## 当前已实现的股票池过滤

- 仅保留沪深 A 股，排除北交所
- 排除 ST
- 排除停牌或当日无日线数据的股票
- 排除上市时间不足 `min_list_days` 的股票
- 排除价格低于 `min_price` 的股票
- 排除 20 日日均成交额低于 `min_avg_turnover` 的股票

## 当前已实现的信号生成

- 买入候选：
  - 从 `is_candidate=true` 的非持仓股票中选
  - 结合 `momentum_20d_rank_pct`、`avg_amount_20d_yuan`、`close_to_ma_10/20`、`volatility_20d`、`volume_ratio` 做打分
  - 权重可通过 `[selection]` 配置段调整
- 卖出候选：
  - 只从当前持仓中选
  - 结合 `momentum_20d`、`close_to_ma_10/20`、`volatility_20d` 计算持仓健康分，选择最弱的一只
  - 权重同样由 `[selection]` 配置段控制
- 生效日期：
  - 基于交易日历推进到下一个开市日，不再按自然日或普通工作日推断
- 执行门槛：
  - `min_buy_score` 控制是否允许开新仓
  - `rotation_edge` 控制满仓时是否值得执行 1 卖 1 买
  - `sell_health_exit_threshold` 控制持仓健康分低到什么程度才允许卖出轮动
  - `buy_max_close_to_ma20` 控制不追离 20 日均线过远的标的

## 当前已实现的基础回测

- 使用缓存的 `daily` / `daily_basic` / `trade_cal` / `stock_basic` 数据
- 使用与日报同一套 universe 构建、过滤和打分逻辑
- 日频、T+1、每次最多 1 买 1 卖
- 从空仓开始，按等权目标仓位逐步建到 5 持仓
- 买卖执行使用次交易日限价撮合近似：
  - 买入限价：`last_close * (1 + buy_markup)`
  - 卖出限价：`last_close * (1 - sell_markdown)`
- 成本包含：
  - 佣金 `commission_rate`
  - 卖出印花税 `stamp_duty_rate`
- 产出：
  - 回测 summary JSON
  - equity curve CSV
  - trade log CSV

## 当前已实现的模拟仓位管理

- 通过 `paper-trade` 命令自动维护：
  - `data/positions/current_positions.csv`
  - `data/positions/latest_pnl.csv`
  - `data/positions/current_state.json`
  - `data/positions/snapshots/YYYYMMDD.csv`
  - `data/positions/trades.csv`
- 不再依赖手工维护持仓 CSV
- 每次运行都会基于同一套历史缓存和策略逻辑重算到指定结束日，减少人工修正造成的漂移
- 适合每天手动运行一次，验证：
  - 数据同步是否完整
  - 模拟仓位是否稳定
  - 下一交易日信号是否合理

## 每日手动运行建议

推荐直接执行：

```bash
ashare-signal run-daily --config configs/strategy.toml.example
```

如果只想使用已有本地缓存验证流程：

```bash
ashare-signal run-daily \
  --config configs/strategy.toml.example \
  --end-date 2026-04-07 \
  --skip-sync
```

执行后查看：

- `data/positions/current_positions.csv`
- `data/positions/latest_pnl.csv`
- `reports/generated/signal-board-YYYYMMDD.md`
- `reports/generated/backtests/backtest-summary-*.json`

## 模拟资金与定时配置

在 `configs/strategy.toml.example` 中调整：

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

- `initial_cash` 是模拟初始资金
- `paper_start_date` 是收益率验证起点
- `daily_run_time` 是容器每天执行时间，建议放在 Tushare 日线数据稳定更新之后
- `sync_lookback_days` 会每天回刷最近几天数据，降低数据修订造成的偏差

## 当前已实现的特征

- `return_1d`
- `momentum_5d`
- `momentum_20d`
- `volatility_20d`
- `avg_amount_20d_yuan`
- `ma_10`
- `ma_20`
- `close_to_ma_10`
- `close_to_ma_20`
- `momentum_20d_rank_pct`
- `turnover_rate`
- `volume_ratio`
- `pe_ttm`
- `pb`
- `total_mv_yuan`
- `circ_mv_yuan`

## 下一步开发顺序

1. 将当前权重型规则打分替换为可配置的因子组合
2. 优化回测撮合与风控规则，补涨跌停和未成交逻辑
3. 增加数据质量检查报告，例如缺失交易日、空行情文件、异常涨跌幅
4. 增加阶段性收益报告，例如按周/月输出收益率、回撤、换手率
