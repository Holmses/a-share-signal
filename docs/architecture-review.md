# A 股量化项目架构与进度评估报告

**项目名称**: a-share-v1-signal  
**评估日期**: 2026-04-17  
**项目版本**: V1  
**评估人**: Claude

---

## 执行摘要

### 项目概况

**项目定位**: 面向 A 股日频信号生成系统，每个交易日收盘后生成下一交易日的买卖建议，维护固定 5 只股票组合。

**核心特性**:
- 市场：沪深 A 股普通股票
- 频率：日频策略
- 持仓：固定 5 只股票
- 轮动：每天最多 1 买 1 卖
- 输出：信号板（不自动下单）

### 总体评价

| 维度 | 评分 | 说明 |
|------|------|------|
| **完成度** | ⭐⭐⭐⭐⭐ (95%) | 核心功能已实现 |
| **架构质量** | ⭐⭐⭐⭐☆ (4/5) | 分层清晰，设计合理 |
| **代码质量** | ⭐⭐⭐⭐☆ (4/5) | 类型安全，可读性高 |
| **可维护性** | ⭐⭐⭐⭐☆ (4/5) | 模块职责明确 |
| **可扩展性** | ⭐⭐⭐⭐☆ (4/5) | 配置驱动，易扩展 |

### 关键发现

✅ **优势**:
1. 模块划分清晰，职责分离良好
2. 回测引擎完整，包含 T+1、交易成本等实盘细节
3. 配置驱动，策略参数可调
4. 领域模型清晰，使用 dataclass 定义业务概念
5. CLI 设计合理，易于自动化

⚠️ **风险**:
1. 缺少涨跌停风控，可能推荐无法成交的股票
2. 数据完整性校验不足
3. 持仓管理需要手动维护，容易出错
4. 日志和监控缺失，问题排查困难
5. 测试覆盖不足

---

## 1. 模块实现完成度

### 1.1 模块清单

| 模块 | 状态 | 完成度 | 关键文件 |
|------|------|--------|---------|
| **data/** | ✅ 完全实现 | 100% | `sync.py`, `tushare_client.py`, `repository.py` |
| **features/** | ✅ 完全实现 | 100% | `factors.py`, `pipeline.py` |
| **strategy/** | ✅ 完全实现 | 100% | `universe.py`, `selector.py`, `signal_board.py` |
| **portfolio/** | ✅ 基础实现 | 80% | `engine.py` |
| **backtest/** | ✅ 完全实现 | 100% | `engine.py` |
| **report/** | ✅ 完全实现 | 100% | `render.py` |
| **scheduler/** | ✅ 完全实现 | 100% | `jobs.py` |

### 1.2 数据层 (data/)

**实现功能**:
- ✅ Tushare API 客户端封装
- ✅ 交易日历、股票基础信息、日线数据同步
- ✅ 本地 CSV 缓存管理
- ✅ 按交易日分文件存储
- ✅ 增量同步支持

**数据源**:
- `trade_cal`: 交易日历
- `stock_basic`: 股票基础信息
- `daily`: 日线价格数据
- `daily_basic`: 日线基础数据（换手率、市值等）

**缓存结构**:
```
data/raw/tushare/
├── trade_cal/SSE.csv
├── stock_basic/L.csv
├── daily/20260407.csv
└── daily_basic/20260407.csv
```

### 1.3 特征工程 (features/)

**已实现特征** (共 16 个):

| 类别 | 特征 | 说明 |
|------|------|------|
| 收益率 | `return_1d` | 1 日收益率 |
| | `momentum_5d` | 5 日动量 |
| | `momentum_20d` | 20 日动量 |
| 波动率 | `volatility_20d` | 20 日波动率 |
| 流动性 | `avg_amount_20d_yuan` | 20 日日均成交额 |
| | `volume_ratio` | 量比 |
| 均线 | `ma_10` | 10 日移动平均线 |
| | `ma_20` | 20 日移动平均线 |
| | `close_to_ma_10` | 收盘价相对 10 日均线偏离度 |
| | `close_to_ma_20` | 收盘价相对 20 日均线偏离度 |
| 排名 | `momentum_20d_rank_pct` | 20 日动量分位数 |
| | `volatility_20d_rank_pct` | 20 日波动率分位数 |
| 估值 | `pe_ttm` | 市盈率 TTM |
| | `pb` | 市净率 |
| | `total_mv_yuan` | 总市值 |
| | `circ_mv_yuan` | 流通市值 |

### 1.4 策略逻辑 (strategy/)

**股票池过滤规则**:
1. 交易所过滤（仅沪深 A 股）
2. ST 股票过滤
3. 停牌股票过滤
4. 上市时间过滤（默认 60 天）
5. 价格过滤（默认 ≥3 元）
6. 流动性过滤（20 日日均成交额）

**买入候选打分** (权重):
- `momentum_20d_rank_pct`: 40%
- `avg_amount_20d_yuan`: 20%
- `close_to_ma_20`: 15%
- `close_to_ma_10`: 10%
- `volatility_20d`: 10%
- `volume_ratio`: 5%

**卖出候选打分** (权重):
- `momentum_20d`: 45%
- `close_to_ma_10`: 25%
- `close_to_ma_20`: 20%
- `volatility_20d`: 10%

### 1.5 回测引擎 (backtest/)

**核心功能**:
- ✅ T+1 交易约束
- ✅ 限价委托撮合
- ✅ 交易成本（佣金 + 印花税）
- ✅ 最小持仓天数约束
- ✅ 固定 5 持仓管理
- ✅ 等权重分配

**输出指标**:
- 总收益率、年化收益率
- 最大回撤
- 夏普比率
- 换手率
- 胜率
- 交易统计

### 1.6 CLI 命令

```bash
ashare-signal sync-tushare      # 数据同步
ashare-signal build-universe    # 构建股票池
ashare-signal generate-signal   # 生成信号
ashare-signal backtest          # 运行回测
```

---

## 2. 架构分析

### 2.1 架构设计

**分层架构**:
```
CLI 层 (cli.py)
    ↓
业务逻辑层 (strategy/, portfolio/, backtest/)
    ↓
数据访问层 (data/repository.py)
    ↓
外部数据源 (Tushare API)
```

**设计模式**:
- 领域驱动设计（DDD）
- 仓储模式（Repository Pattern）
- 策略模式（Strategy Pattern）
- 配置驱动

### 2.2 架构优势

#### 1. 清晰的职责分离
- 数据层只负责数据获取和缓存
- 特征层只负责特征计算
- 策略层只负责信号生成
- 回测层只负责模拟交易

#### 2. 领域模型清晰
```python
@dataclass
class Candidate:
    symbol: str
    name: str
    score: float
    reason: Dict[str, Any]

@dataclass
class Position:
    symbol: str
    name: str
    entry_date: str
    entry_price: float
    quantity: int

@dataclass
class SignalBoard:
    signal_date: str
    calc_trade_date: str
    effective_date: str
    buy_signal: Optional[Candidate]
    sell_signal: Optional[Candidate]
```

#### 3. 配置驱动
```toml
[market]
max_positions = 5

[filters]
min_list_days = 60
min_price = 3.0
min_avg_turnover = 50000000

[strategy]
buy_top_n = 1
sell_top_n = 1

[backtest]
initial_cash = 1000000
commission_rate = 0.0003
stamp_duty_rate = 0.001
```

#### 4. 数据缓存策略
- 按交易日分文件存储
- 支持增量同步
- 减少 API 调用

---

## 3. 问题与风险

### 3.1 高优先级问题

#### 问题 1: 涨跌停风控缺失 ⚠️

**现状**: 买入信号没有检查涨停板，卖出信号没有检查跌停板

**风险**:
- 推荐买入涨停股，次日无法成交
- 推荐卖出跌停股，次日无法成交

**影响**: 信号质量下降，实际执行率低

**建议**:
```python
# src/ashare_signal/strategy/risk_control.py

def is_limit_up(row) -> bool:
    """检查是否涨停"""
    return abs(row['close'] - row['high']) < 0.01 and row['pct_chg'] > 9.5

def is_limit_down(row) -> bool:
    """检查是否跌停"""
    return abs(row['close'] - row['low']) < 0.01 and row['pct_chg'] < -9.5

def filter_limit_stocks(candidates: pd.DataFrame, action: str) -> pd.DataFrame:
    """过滤涨跌停股票"""
    if action == 'buy':
        return candidates[~candidates.apply(is_limit_up, axis=1)]
    elif action == 'sell':
        return candidates[~candidates.apply(is_limit_down, axis=1)]
    return candidates
```

#### 问题 2: 数据完整性校验不足 ⚠️

**现状**: 数据同步后缺少完整性检查，没有重试机制

**风险**:
- 数据缺失导致信号生成失败
- Tushare API 限流导致数据不完整

**影响**: 系统稳定性差

**建议**:
```python
# src/ashare_signal/data/validation.py

@dataclass
class ValidationResult:
    is_valid: bool
    missing_files: List[str]
    empty_files: List[str]
    error_message: str

def validate_daily_data(trade_date: str, data_dir: str) -> ValidationResult:
    """检查指定交易日的数据完整性"""
    required_files = [
        f"daily/{trade_date}.csv",
        f"daily_basic/{trade_date}.csv"
    ]
    
    missing = []
    empty = []
    
    for file in required_files:
        path = os.path.join(data_dir, file)
        if not os.path.exists(path):
            missing.append(file)
        elif os.path.getsize(path) == 0:
            empty.append(file)
    
    is_valid = len(missing) == 0 and len(empty) == 0
    error_msg = ""
    if missing:
        error_msg += f"缺失文件: {missing}. "
    if empty:
        error_msg += f"空文件: {empty}."
    
    return ValidationResult(is_valid, missing, empty, error_msg)

def sync_with_retry(sync_func, max_retries=3, backoff=5):
    """带重试的数据同步"""
    for attempt in range(max_retries):
        try:
            return sync_func()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(backoff * (attempt + 1))
                logger.warning(f"同步失败，重试 {attempt + 1}/{max_retries}: {e}")
            else:
                logger.error(f"同步失败，已达最大重试次数: {e}")
                raise
```

#### 问题 3: 持仓管理不完整 ⚠️

**现状**: 需要手动维护 `current_positions.csv`

**风险**:
- 手动维护容易出错
- 无法追踪持仓历史
- 无法计算持仓收益

**影响**: 运维成本高，容易出错

**建议**:
```python
# src/ashare_signal/portfolio/manager.py

class PortfolioManager:
    def update_positions_after_trade(self, trades: List[Trade]):
        """交易后自动更新持仓文件"""
        positions = self.load_current_positions()
        
        for trade in trades:
            if trade.action == 'buy':
                positions.append(Position(
                    symbol=trade.symbol,
                    name=trade.name,
                    entry_date=trade.trade_date,
                    entry_price=trade.price,
                    quantity=trade.quantity
                ))
            elif trade.action == 'sell':
                positions = [p for p in positions if p.symbol != trade.symbol]
        
        self.save_positions(positions)
        self.save_position_snapshot(trade.trade_date, positions)
    
    def calculate_position_pnl(self, current_prices: Dict[str, float]) -> pd.DataFrame:
        """计算持仓盈亏"""
        positions = self.load_current_positions()
        pnl_data = []
        
        for pos in positions:
            current_price = current_prices.get(pos.symbol, pos.entry_price)
            pnl = (current_price - pos.entry_price) * pos.quantity
            pnl_pct = (current_price / pos.entry_price - 1) * 100
            
            pnl_data.append({
                'symbol': pos.symbol,
                'name': pos.name,
                'entry_price': pos.entry_price,
                'current_price': current_price,
                'quantity': pos.quantity,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
        
        return pd.DataFrame(pnl_data)
    
    def save_position_snapshot(self, date: str, positions: List[Position]):
        """保存持仓快照"""
        snapshot_dir = "data/positions/snapshots"
        os.makedirs(snapshot_dir, exist_ok=True)
        
        df = pd.DataFrame([asdict(p) for p in positions])
        df.to_csv(f"{snapshot_dir}/{date}.csv", index=False)
```

### 3.2 中优先级问题

#### 问题 4: 日志和监控缺失 ⚠️

**现状**: 关键操作缺少日志记录

**风险**: 问题排查困难

**建议**:
```python
# src/ashare_signal/utils/logging.py

import logging
import sys

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """配置日志系统"""
    logger = logging.getLogger("ashare_signal")
    logger.setLevel(log_level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# 在关键操作中使用
logger = logging.getLogger("ashare_signal")

def sync_tushare(start_date, end_date):
    logger.info(f"开始同步数据: {start_date} -> {end_date}")
    try:
        result = sync()
        logger.info(f"同步完成: 开市日={result.open_days}, 股票数={result.stock_count}")
        return result
    except Exception as e:
        logger.error(f"同步失败: {e}", exc_info=True)
        raise
```

#### 问题 5: 测试覆盖不足 ⚠️

**现状**: 只有 4 个测试文件，缺少集成测试

**建议**:
```python
# tests/test_backtest_engine.py
def test_backtest_with_t1_constraint():
    """测试 T+1 约束"""
    
def test_backtest_transaction_cost():
    """测试交易成本计算"""

# tests/integration/test_signal_pipeline.py
def test_full_signal_generation_pipeline():
    """端到端测试：数据同步 -> 特征计算 -> 信号生成"""
```

#### 问题 6: 性能优化空间 ⚠️

**现状**: 使用 CSV 格式，特征计算串行

**建议**:
```python
# 1. 使用 Parquet 格式
df.to_parquet('daily/20260407.parquet')

# 2. 并行计算
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor() as executor:
    results = executor.map(compute_features, stock_groups)
```

---

## 4. 与计划文档对比

### 4.1 计划符合度

| 计划要求 | 实现状态 | 备注 |
|---------|---------|------|
| 只输出信号，不自动下单 | ✅ 符合 | 输出 Markdown 信号板 |
| 每日收盘后生成信号 | ✅ 符合 | `scheduler/jobs.py` |
| 组合维持 5 只持仓 | ✅ 符合 | `max_positions = 5` |
| 每天只输出 1 买 1 卖 | ✅ 符合 | `buy_top_n = 1, sell_top_n = 1` |
| 输出建议委托价 | ✅ 符合 | `buy_markup = 0.003` |
| 输出信号理由 | ✅ 符合 | `Candidate.format_reason()` |
| T+1 约束 | ✅ 符合 | 回测引擎实现 |
| 股票池过滤 | ✅ 符合 | 6 项过滤规则 |
| 使用 Tushare 数据源 | ✅ 符合 | `tushare_client.py` |
| 回测框架 | ✅ 符合 | 完整的回测引擎 |
| 日报生成 | ✅ 符合 | Markdown 渲染 |
| 定时执行 | ⚠️ 部分符合 | 代码已实现，未配置 cron |

---

## 5. 改进建议

### 5.1 短期任务（1-2 周）

#### 任务 1: 完善风控机制
- [ ] 添加涨跌停过滤
- [ ] 添加流动性异常检查
- [ ] 添加行业集中度检查

#### 任务 2: 完善数据层
- [ ] 添加数据完整性校验
- [ ] 添加重试机制
- [ ] 生成数据质量报告

#### 任务 3: 自动化持仓管理
- [ ] 实现持仓自动更新
- [ ] 实现持仓历史追踪
- [ ] 实现持仓收益计算

### 5.2 中期任务（1-2 月）

#### 任务 4: 完善日志和监控
- [ ] 添加结构化日志
- [ ] 添加性能监控
- [ ] 添加告警机制（邮件/钉钉）

#### 任务 5: 增加测试覆盖
- [ ] 回测引擎测试
- [ ] 集成测试
- [ ] 性能测试

#### 任务 6: 性能优化
- [ ] 使用 Parquet 格式
- [ ] 特征计算并行化
- [ ] 缓存优化

### 5.3 长期任务（3-6 月）

#### 任务 7: 策略优化
- [ ] 参数网格搜索
- [ ] 因子有效性分析
- [ ] 机器学习模型集成

#### 任务 8: 部署自动化
- [ ] 配置 cron 定时任务
- [ ] Docker 容器化
- [ ] CI/CD 流水线

#### 任务 9: 功能扩展
- [ ] Web 界面
- [ ] 实时监控面板
- [ ] 多策略组合

---

## 6. 最终结论

### 6.1 项目评价

**该项目架构合理，实现质量高，已具备生产可用性。**

**核心优势**:
1. ✅ 模块划分清晰，职责分离良好
2. ✅ 回测引擎完整，包含 T+1、交易成本等实盘细节
3. ✅ 配置驱动，策略参数可调
4. ✅ 领域模型清晰，代码可读性高
5. ✅ CLI 设计合理，易于自动化

**主要不足**:
1. ⚠️ 缺少涨跌停风控
2. ⚠️ 数据完整性校验不足
3. ⚠️ 持仓管理需要手动维护
4. ⚠️ 日志和监控缺失
5. ⚠️ 测试覆盖不足

### 6.2 实施建议

**建议按以下优先级推进**:

1. **立即执行** (1-2 周):
   - 添加涨跌停风控
   - 实现持仓自动更新
   - 添加数据完整性检查

2. **短期执行** (1 个月):
   - 完善日志和监控
   - 增加测试覆盖
   - 配置定时任务

3. **中期执行** (2-3 个月):
   - 性能优化
   - 策略参数优化
   - Web 界面开发

### 6.3 试运行建议

**项目可以开始试运行，建议先纸面跟踪 1-2 个月，验证信号质量后再考虑实盘。**

**试运行检查清单**:
- [ ] 每日信号生成稳定
- [ ] 信号质量符合预期
- [ ] 建议委托价合理
- [ ] 涨跌停风控有效
- [ ] 数据同步稳定
- [ ] 持仓管理准确

**风险提示**:
- A 股存在涨跌停，信号不代表一定能成交
- 日线策略对开盘跳空敏感
- 建议委托价不是精确预测价
- 需要持续监控信号质量和执行率

---

## 附录

### A. 关键文件路径

| 功能 | 文件路径 |
|------|---------|
| 数据同步 | `src/ashare_signal/data/sync.py` |
| Tushare 客户端 | `src/ashare_signal/data/tushare_client.py` |
| 数据仓库 | `src/ashare_signal/data/repository.py` |
| 特征计算 | `src/ashare_signal/features/pipeline.py` |
| 股票池过滤 | `src/ashare_signal/strategy/universe.py` |
| 信号选择 | `src/ashare_signal/strategy/selector.py` |
| 信号板生成 | `src/ashare_signal/strategy/signal_board.py` |
| 回测引擎 | `src/ashare_signal/backtest/engine.py` |
| 报告渲染 | `src/ashare_signal/report/render.py` |
| 日报任务 | `src/ashare_signal/scheduler/jobs.py` |
| 配置系统 | `src/ashare_signal/config.py` |
| CLI 入口 | `src/ashare_signal/cli.py` |

### B. 配置文件

| 文件 | 说明 |
|------|------|
| `configs/strategy.toml.example` | 策略配置模板 |
| `configs/current_positions.csv` | 当前持仓 |
| `.env` | 环境变量（Tushare Token） |

### C. 数据目录

| 目录 | 说明 |
|------|------|
| `data/raw/tushare/` | Tushare 原始数据缓存 |
| `data/processed/universe/` | 处理后的股票池快照 |
| `reports/generated/` | 生成的信号板和回测报告 |
| `logs/` | 日志文件 |

---

**报告结束**
