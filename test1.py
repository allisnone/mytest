#encoding:gbk
# v20230113
# v20250105
# v20250205
# v20250323

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import defaultdict, deque

# -------------------- 配置日志系统 --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TradingSystem')


# -------------------- 数据类定义 --------------------
@dataclass
class StrategyConfig:
    """策略核心参数配置"""
    # 信号参数
    daily_n_days: int = 3          # 日线回溯窗口
    hourly_n_days: int = 5         # 小时线回溯窗口（交易日）
    min_trade_units: int = 100     # 最小交易单位
    # TODO：reserve_volume 用法
    """ 
    reserve_volume：卖出股票时，根据一定的规则预留标的股票的股份数；拟订规则：基于现有持仓的股票数，标识标的的特定交易状态，比如确定是否日后买入和卖出的操作类型，默认0，全仓卖出；100-日后触发策略自动买入
    # a = reserve_volume%1000=xyz
    # x=reserve_volume%1000//100，用于标识交易类型（暂不启用），x=0，默认交易类型；x=1，25%的仓位T+0滚动交易; x=2，50%的仓位 T+0滚动交易; x=4，100%的仓位 T+0滚动交易; x=其他保，留
    # y=reserve_volume%100//10，用于标识个股分仓的持仓比例，仓位控制：y*10%为实际分仓仓位。仓位保障优先级：z=9的标的仓位最最优先保障，z为其他值，z越小越优先；z值相同，y越大越优先，优先保留标的到特定仓位；
    # z=reserve_volume%10，用于标识股票阶段类型，不同阶段不同策略风格；z=9，表示不区分阶段，长期持有；z=1~5对应阶段划分的阶段，即当z=1时，表示当前处于阶段1；z=0和6~8保留
    # 举例1：长期持有的股票，保持10%的仓位，则预留特定reserve_volume=19（长期持有，保持10%的仓位：如果当天有卖出，尾盘买回仓位；如当天买入超过10%的仓位，第二天卖出）；实际策略运行卖出股票时，如计划卖出数量volume（如500股），则实际下单卖出的数量的=volume-reserve_volume=481股
    # 举例2：reserve_volume=31，阶段1的股票，后续根据策略的买卖点，仓位控制最高在30%的仓位
    # 举例3：reserve_volume=52，阶段2的股票，后续根据策略的买卖点，仓位控制最高在50%的仓位
    # 举例4：reserve_volume=23，阶段3的股票，后续根据策略的买卖点，仓位控制最高在20%的仓位
    # 举例5：reserve_volume=15，阶段5的股票，后续根据策略的买卖点，仓位控制最高在10%的仓位
    """
    reserve_volume: int = 0

    # 仓位管理
    max_account_position: float = 0.75   # 账户最大持仓比例
    min_stock_num: int = 3               # 最小持仓个股数
    single_stock_cap: float = None       # 单股最大仓位比例（动态计算）

    # 风控参数
    stop_loss_stock: float = -3.0        # 个股止损比例
    stop_profit_stock: float = 6.0       # 个股止盈比例
    partial_sell_ratio: float = 0.5      # 部分卖出比例

    def __post_init__(self):
        self.single_stock_cap = round(self.max_account_position / self.min_stock_num, 2)


@dataclass
class RiskConfig:
    """风险控制参数配置"""
    # 大盘参数
    # 国证A指：399317.SZ
    index_symbol: str = '399317.SZ'          # 基准指数
    index_daily_n: int = 3                   # 指数日线窗口
    index_hourly_n: int = 5                  # 指数小时线窗口

    # 熔断机制
    circuit_breaker_threshold: float = -0.05  # 指数暴跌，熔断触发阈值
    circuit_breaker_days: int = 3             # 指数触发熔断时，清仓，并禁止circuit_breaker_days个交易日买入操作

    # 账户风控
    stop_loss_account: float = -2.0          # 账户级止损
    stop_profit_account: float = 5.0         # 账户级止盈
    stop_loss_account: float = 0.5          # 账户级止损时，仓位减少的乘数比例
    consecutive_loss_days: int = 3           # 账户连续续consecutive_loss_days天数，清仓，并禁止circuit_breaker_days个交易日买入操作

    # 市场状态
    volatility_window: int = 14              # 波动率计算窗口
    high_volatility_threshold: float = 0.03   # 高波动阈值
    low_liquidity_threshold: int = 1000000    # 低流动性阈值
    """
    核心风控原则，坚持长期主义，尊重和敬畏市场，严守交易纪律，慎独慎思慎行，化繁为简，活下活着活好！
    
    仓位管理：
    分仓操作：至少4个分仓，按投资偏好和操作风格分仓（成长、价值、周期）；不同行业分仓；个股仓位不超过总仓位的50%；
    仓位操作原则：
    1、开仓1432，避免倒金字塔开仓；
    2、平仓532，避免金字塔式获利了结。
    
    风险分类和处置：
    1、市场风险：指数单日大幅下跌超过-5%，触发熔断，清仓；一周内不开仓；  （指数取沪深300）
    2、市场风险：指数连续三日触发日线级别卖出信号，清仓；一周内不开仓；
    3、指数风险：指数单日跌幅超过2%或者日线级别卖出信号，总体仓位不超过30%
    4、指数风险：指数当日出现小时级别卖出信号，总体仓位不超过60%
    5、系统风险：账户连续3个交易日亏损均超过3%，清仓；
    6、系统风险：账户连续单日亏损超过3%，总体仓位不超过30%；
    7、个股风险：单一个股跌幅超过总市值的2%，个股清仓）
    8、个股风险：单一股票出现小时级别卖出信号，个股仓位减少至原来的50%；当日连续出现2个小时级别的K线信号，卖出全部；
    9、个股风险：单一股票出现日线级别卖出信号，个股全仓卖出；
    
    风险规避原则：
    1、ST股不参与、财务造假不参与、潜在ST不参与
    5、人民币大幅贬值，减仓
    6、港股大跌，跟随减仓
    7、美股大跌，早盘减仓
    7、黄金暴跌，规避有色
    8、石油暴跌，规避资源
    9、疫情，规避消费
    10、流感，偏好医药
    
    去掉不好的交易习惯：
    1、交易不设置止损
    2、干预系统止损，臆想意淫
    3、不止损，不认错，不悔改
    4、无交易计划，无交易目标
    5、急功近利，满仓梭哈————快速来的财富，大概率快速离你而去；
    6、个股加速冲高时，追高买入————加速拉高，是为找接盘侠，主力永远看着你口袋的前
    7、大幅杀跌刚开始时，过早买入————市场大幅下跌是合力的结果，是群体行为，群体行为未发生改变前，正式每一份本金；
    8、道听图，找消息，随意买入————让容易得到的消息，是别人想让你知道的消息，世界没有无缘无故的爱
    9、时刻盯盘，夜不能寐，有损健康————长期主义，化繁为简，自动化交易；锻炼身体，每周至少跑步一次
    10、计划短线，且犹豫不决————杀伐果断，短线即一夜情
    11、喜欢好看的，漂亮————漂亮是广告效应，为吸引眼球
    12、不够重视既得盈利————正视股市赚到的每一笔钱
   
    
    
    坚守原则：
    1、尊重市场，跟随市场
    2、敢于认错，及时止损
    3、赚取认知内利润，系统内交易
    4、逆向思维，独立思考，提高认知
    5、坚持长期主义，耐心等待
    6、性格偏好：稳重、顿感、知足长乐的性格和心态，龙回头后抄底更适合保守的性格，前期二板以上的龙头股，下跌超过30%后，长期盘整一个月以上，下跌衰竭时买入；
    """


# -------------------- 核心模块实现 --------------------
class DataManager:
    """统一数据管理"""
    def __init__(self, config: StrategyConfig):
        self.config = config
        # 数据缓存结构优化
        self.market_data = {
            'daily': defaultdict(pd.DataFrame),
            'hourly': defaultdict(pd.DataFrame),
            'ticks': defaultdict(list)
        }
        self.last_update = defaultdict(datetime)

    def update_market_data(self, symbol: str, period: str, context):
        """统一数据更新入口"""
        try:
            if self._need_update(symbol, period):
                if period == '1d':
                    self._update_daily_data(symbol, context)
                elif period == '1h':
                    self._update_hourly_data(symbol, context)
                logger.info(f"{period}数据更新完成: {symbol}")
        except Exception as e:
            logger.error(f"数据更新失败 {symbol} {period}: {str(e)}")

    def _update_daily_data(self, symbol: str, context):
        """日线数据更新"""
        data = context.get_market_data(
            ['open','high','low','close','volume'],
            stock_code=symbol,
            period='1d',
            dividend_type='front_ratio',
            count=self.config.daily_n_days + 5  # 多取5日防止边界问题
        )
        self.market_data['daily'][symbol] = data

    def _update_hourly_data(self, symbol: str, context):
        """小时数据更新"""
        data = context.get_market_data(
            ['open','high','low','close','volume'],
            stock_code=symbol,
            period='1h',
            dividend_type='front_ratio',
            count=self.config.hourly_n_days * 6  # 按每日6小时计算
        )
        self.market_data['hourly'][symbol] = data

    def _need_update(self, symbol: str, period: str) -> bool:
        """智能更新检测"""
        last = self.last_update.get((symbol, period))
        delta = datetime.now() - last if last else timedelta.max
        return delta > timedelta(minutes=5)  # 5分钟更新间隔


class SignalGenerator:
    """统一信号生成"""
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.signal_cache = defaultdict(dict)

    def generate_signal(self, symbol: str, data: Dict[str, pd.DataFrame], k_multiple: int) -> Dict:
        """多周期信号生成"""
        signals = {'buy': False, 'sell': False, 'strength': 0}

        try:
            # 日线信号
            daily_data = data['daily'].get(symbol, pd.DataFrame())
            if len(daily_data) >= self.config.daily_n_days:
                ref_high = daily_data['high'][-self.config.daily_n_days:-1].max()
                ref_low = daily_data['low'][-self.config.daily_n_days:-1].min()
                current = daily_data.iloc[-1]
                signals['daily_buy'] = current['high'] > ref_high
                signals['daily_sell'] = current['low'] < ref_low

            # 小时线信号
            k_multiple = 6
            hourly_data = data['hourly'].get(symbol, pd.DataFrame())
            if len(hourly_data) >= self.config.hourly_n_days * k_multiple:
                ref_high = hourly_data['high'][-self.config.hourly_n_days*k_multiple:-1].max()
                ref_low = hourly_data['low'][-self.config.hourly_n_days*k_multiple:-1].min()
                current = hourly_data.iloc[-1]
                signals['hourly_buy'] = current['high'] > ref_high
                signals['hourly_sell'] = current['low'] < ref_low

            # 信号强度计算
            signals['strength'] = self._calculate_strength(signals)
            return signals
        except Exception as e:
            logger.error(f"信号生成失败 {symbol}: {str(e)}")
            return signals

    def _calculate_strength(self, signals: Dict) -> int:
        """信号强度量化"""
        strength = 0
        if signals['daily_buy']: strength += 2
        if signals['hourly_buy']: strength += 1
        if signals['daily_sell']: strength -= 2
        if signals['hourly_sell']: strength -= 1
        return strength


class RiskController:
    """统一风险控制"""
    def __init__(self, strat_cfg: StrategyConfig, risk_cfg: RiskConfig):
        self.strat_cfg = strat_cfg
        self.risk_cfg = risk_cfg
        self.risk_status = {
            'circuit_breaker': False,
            'consecutive_loss': 0,
            'position_limits': defaultdict(float)
        }

    def check_market_risk(self, data: Dict) -> Dict:
        """市场级风险检测"""
        risk = {}
        # 熔断检查
        index_data = data['daily'].get(self.risk_cfg.index_symbol)
        if index_data is not None:
            current = index_data.iloc[-1]
            prev_close = index_data['close'].values.tolist()[-2]
            drawdown = (current['low'] - prev_close) / prev_close
            if drawdown <= self.risk_cfg.circuit_breaker_threshold:
                risk['circuit_breaker'] = True
        # 波动率检查
        for symbol in data['daily'].keys():
            vols = self._calculate_volatility(data['daily'][symbol])
            if vols[-1] > self.risk_cfg.high_volatility_threshold:
                self.risk_status['position_limits'][symbol] = 0.5  # 限仓50%

        return risk

    def check_account_risk(self, positions: List, history: pd.DataFrame) -> Dict:
        """账户级风险检测"""
        risk = {}
        # 连续亏损检测
        recent_pnl = history['pnl'].tail(5)
        if all(recent_pnl < 0):
            self.risk_status['consecutive_loss'] += 1

        # 当前持仓风险
        total_value = sum(p.market_value for p in positions)
        max_drawdown = min(p.drawdown for p in positions)
        if max_drawdown <= self.risk_cfg.stop_loss_account:
            risk['position_adjust'] = 'reduce'

        return risk

    def _calculate_volatility(self, data: pd.DataFrame) -> List[float]:
        """计算波动率（ATR）"""
        high, low, close = data['high'], data['low'], data['close']
        tr = np.maximum(high - low,
                        abs(high - close.shift(1)),
                        abs(low - close.shift(1)))
        return tr.rolling(self.risk_cfg.volatility_window).mean().tolist()


class PortfolioManager:
    """投资组合管理"""
    def __init__(self, strat_cfg: StrategyConfig):
        self.cfg = strat_cfg

    def adjust_position(self, signals: Dict, current_pos: Dict) -> Dict:
        """生成调仓指令"""
        target_pos = {}
        # 个股仓位控制
        for symbol, pos in current_pos.items():
            print('signals=', signals)
            if symbol in list(signals.keys()):
                target = self._calculate_single_adjust(signals[symbol], pos)
                target_pos[symbol] = target
            else:
                print(symbol + ' is not exist')
        # 组合级风控
        total = sum(target_pos.values())
        if total > self.cfg.max_account_position:
            scale = self.cfg.max_account_position / total
            target_pos = {k: v*scale for k, v in target_pos.items()}

        return target_pos

    def _calculate_single_adjust(self, symbol: str, signal: Dict, position: float) -> float:
        """单股仓位计算"""
        # 基础信号驱动
        if signal['strength'] >= 1:  # 强势买入
            target = min(position + self.cfg.single_stock_cap, self.cfg.single_stock_cap)
        elif signal['strength'] <= -1:  # 强制卖出
            target = 0.0
        else:  # 保持
            target = position

        # 应用风控限制
        if target > self.risk_status['position_limits'].get(symbol, 1.0):
            target = self.risk_status['position_limits'][symbol]

        return round(target, 4)


# -------------------- 交易引擎 --------------------
class TradingEngine:
    """统一交易引擎"""
    def __init__(self, strat_cfg: StrategyConfig, risk_cfg: RiskConfig):
        self.strat_cfg = strat_cfg
        self.risk_cfg = risk_cfg
        self.data_mgr = DataManager(strat_cfg)
        self.signal_gen = SignalGenerator(strat_cfg)
        self.risk_ctrl = RiskController(strat_cfg, risk_cfg)
        self.portfolio_mgr = PortfolioManager(strat_cfg)

        self.current_positions = {}
        self.trade_enabled = True

    def run_cycle(self, context):
        """运行交易周期"""
        try:
            # 1. 数据更新
            self._update_market_data(context)

            # 2. 信号生成
            signals = self._generate_signals()

            # 3. 风险检测
            market_risk = self.risk_ctrl.check_market_risk(self.data_mgr.market_data)
            account_risk = self.risk_ctrl.check_account_risk(self.current_positions, context.history)

            # 4. 组合管理
            target_pos = self.portfolio_mgr.adjust_position(signals, self.current_positions)

            # 5. 执行交易
            self._execute_trades(target_pos, context)

        except Exception as e:
            logger.error(f"交易周期执行失败: {str(e)}")
            self.trade_enabled = False

    def _update_market_data(self, context):
        """更新市场数据"""
        for symbol in context.universe + [self.risk_cfg.index_symbol]:
            self.data_mgr.update_market_data(symbol, '1d', context)
            self.data_mgr.update_market_data(symbol, '1h', context)

    def _generate_signals(self) -> Dict:
        """生成全市场信号"""
        signals = {}
        for symbol in self.data_mgr.market_data['daily'].keys():
            signals[symbol] = self.signal_gen.generate_signal(
                symbol, self.data_mgr.market_data)
        return signals

    def _execute_trades(self, target_pos: Dict, context):
        """执行调仓操作"""
        for symbol, target in target_pos.items():
            current = self.current_positions.get(symbol, 0.0)
            if abs(target - current) < 0.01:  # 忽略微小调整
                continue

            # 计算调整量
            adj_amount = round(target - current, 4)
            if adj_amount > 0:
                self._place_order(symbol, adj_amount, 'buy', context)
            else:
                self._place_order(symbol, abs(adj_amount), 'sell', context)

    def _place_order(self, symbol: str, amount: float, side: str, context):
        """委托下单"""
        try:
            # 获取实时价格
            tick = context.get_full_tick([symbol])[symbol]
            price = tick.ask1 if side == 'buy' else tick.bid1

            # 转换比例为实际股数
            units = max(int(amount * 1e4 / price), self.strat_cfg.min_trade_units)

            # 调用交易接口
            order_code = 23 if side == 'buy' else 24
            passorder(order_code, 1101, context.account_id, symbol,
                      11, price, units, 'Systematic', 1, self._gen_order_id(symbol, order_code), context)

            logger.info(f"已提交订单 {symbol} {side} {units}股 @{price}")
        except Exception as e:
            logger.error(f"下单失败 {symbol}: {str(e)}")

    def _gen_order_id(self, symbol: str, order_code: str) -> str:
        """生成唯一订单ID"""
        # order_code = 23 if side == 'buy' else 24
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{symbol}_{order_code}_{timestamp}_{random.randint(1000,9999)}"
    """
    def _gen_order_id(self) -> str:
        
        return f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}"
    """


# -------------------- 初始化及主循环 --------------------
def init(ContextInfo):
    """策略初始化"""
    try:
        # 初始化配置
        strat_cfg = StrategyConfig()
        risk_cfg = RiskConfig()

        # 创建交易引擎
        ContextInfo.engine = TradingEngine(strat_cfg, risk_cfg)

        # 设置初始持仓
        positions = get_trade_detail_data(ContextInfo.account_id,
                                          ContextInfo.account_type,
                                          'POSITION')
        position_stocks = []
        for p in positions:
            symbol = f"{p.m_strInstrumentID}.{'SH' if p.m_strExchangeID == 'SSE' else 'SZ'}"
            position_stocks.append(symbol)
            ContextInfo.engine.current_positions[symbol] = p.m_dMarketValue
        index_list = ['399317.SZ']
        # 国证A指：399317.SZ
        # 沪深300：000300.SH
        # 上证50：000016.SH
        # 上证银行：000134.SH
        # index_list = ['399317.SZ']# ['399381.SZ', '399382.SZ', '399383.SZ', '399384.SZ', '399385.SZ', '399386.SZ']
        exit_stocks = ['300742.SZ', '000918.SZ', '002002.SZ', '600393.SH', '600466.SH', '002087.SZ', '000961.SZ', '000413.SZ', '002288.SZ', '000666.SZ', '600077.SH', '600297.SH', '600836.SH', '000540.SZ', '000836.SZ', '002435.SZ', '000996.SZ', '002013.SZ', '600277.SH', '600213.SH', '002621.SZ', '600565.SH', '000023.SZ', '000671.SZ', '300116.SZ', '000046.SZ', '600321.SH', '002280.SZ', '300262.SZ', '600766.SH', '002502.SZ', '000667.SZ', '000416.SZ', '000861.SZ', '601258.SH', '600220.SH', '002341.SZ', '002505.SZ', '002610.SZ', '002665.SZ', '000976.SZ', '603133.SH', '002325.SZ', '002503.SZ', '300799.SZ', '600647.SH', '600823.SH', '002118.SZ', '000982.SZ', '603555.SH', '002308.SZ']
        universe_stocks = list(set(position_stocks+index_list).difference(set(exit_stocks)))
        print('universe_stocks=', universe_stocks)
        ContextInfo.set_universe(universe_stocks)
        logger.info("交易系统初始化完成")
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}")
        ContextInfo.engine.trade_enabled = False


def handlebar(ContextInfo):
    """主处理循环"""
    if not ContextInfo.engine.trade_enabled:
        return

    try:
        ContextInfo.engine.run_cycle(ContextInfo)
    except Exception as e:
        logger.error(f"主循环异常: {str(e)}")
        ContextInfo.engine.trade_enabled = False

"""
import unittest
from unittest.mock import Mock


class TestTradingSystem(unittest.TestCase):
    def setUp(self):
        self.engine = TradingEngine(StrategyConfig(), RiskConfig())
        self.context = Mock()
        self.context.universe = ['600519.SH', '000001.SZ']

    def test_market_risk(self):
        # 构造市场暴跌场景
        index_data = pd.DataFrame({
            'high': [3550, 3500, 3400, 3300],
            'low': [3470, 3400, 3300, 3200],
            'close': [3495, 3450, 3350, 3250]
        })
        self.engine.data_mgr.market_data['daily']['000001.SH'] = index_data
        risk = self.engine.risk_ctrl.check_market_risk(self.engine.data_mgr.market_data)
        print('risk=', risk)
        if risk:
            self.assertTrue(risk['circuit_breaker'])

    def test_portfolio_rebalance(self):
        # 初始持仓
        self.engine.current_positions = {
            '600519.SH': 0.3,
            '000001.SZ': 0.5
        }
        # 生成买入信号
        signals = {
            '600519.SH': {'strength': 2},
            '000001.SZ': {'strength': -2}
        }
        position_datas= {'002917.SZ': {'m_nCanUseVolume': 3900, 'm_dMarketValue': 61191.0, 'm_dOpenPrice': 15.17, 'm_dFloatProfit': 0.0, 'm_dPositionProfit': 2023.46, 'm_dProfitRate': 0.03, 'm_bIsToday': True, 'm_strOpenDate': '', 'm_strTradingDay': '20250224', 'm_nVolume': 3900, 'm_dLastPrice': 15.69, 'm_nFrozenVolume': 0}, '300231.SZ': {'m_nCanUseVolume': 6600, 'm_dMarketValue': 345054.0, 'm_dOpenPrice': 12.44, 'm_dFloatProfit': -10480.0, 'm_dPositionProfit': 19110.04, 'm_dProfitRate': 0.06, 'm_bIsToday': True, 'm_strOpenDate': '', 'm_strTradingDay': '20250224', 'm_nVolume': 26200, 'm_dLastPrice': 13.17, 'm_nFrozenVolume': 19600}, '300842.SZ': {'m_nCanUseVolume': 1200, 'm_dMarketValue': 59844.0, 'm_dOpenPrice': 46.63, 'm_dFloatProfit': 3936.0, 'm_dPositionProfit': 3883.35, 'm_dProfitRate': 0.07, 'm_bIsToday': True, 'm_strOpenDate': '', 'm_strTradingDay': '20250224', 'm_nVolume': 1200, 'm_dLastPrice': 49.870000000000005, 'm_nFrozenVolume': 0}}
        target = self.engine.portfolio_mgr.adjust_position(signals, position_datas)
        print('target=', target)
        if target:
            self.assertLess(target['000001.SZ'], 0.1)
            self.assertAlmostEqual(sum(target.values()), 0.75, delta=0.01)

if __name__ == '__main__':
    unittest.main()

"""