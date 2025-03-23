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

# -------------------- ������־ϵͳ --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TradingSystem')


# -------------------- �����ඨ�� --------------------
@dataclass
class StrategyConfig:
    """���Ժ��Ĳ�������"""
    # �źŲ���
    daily_n_days: int = 3          # ���߻��ݴ���
    hourly_n_days: int = 5         # Сʱ�߻��ݴ��ڣ������գ�
    min_trade_units: int = 100     # ��С���׵�λ
    # TODO��reserve_volume �÷�
    """ 
    reserve_volume��������Ʊʱ������һ���Ĺ���Ԥ����Ĺ�Ʊ�Ĺɷ������ⶩ���򣺻������гֲֵĹ�Ʊ������ʶ��ĵ��ض�����״̬������ȷ���Ƿ��պ�����������Ĳ������ͣ�Ĭ��0��ȫ��������100-�պ󴥷������Զ�����
    # a = reserve_volume%1000=xyz
    # x=reserve_volume%1000//100�����ڱ�ʶ�������ͣ��ݲ����ã���x=0��Ĭ�Ͻ������ͣ�x=1��25%�Ĳ�λT+0��������; x=2��50%�Ĳ�λ T+0��������; x=4��100%�Ĳ�λ T+0��������; x=����������
    # y=reserve_volume%100//10�����ڱ�ʶ���ɷֲֵĳֱֲ�������λ���ƣ�y*10%Ϊʵ�ʷֲֲ�λ����λ�������ȼ���z=9�ı�Ĳ�λ�������ȱ��ϣ�zΪ����ֵ��zԽСԽ���ȣ�zֵ��ͬ��yԽ��Խ���ȣ����ȱ�����ĵ��ض���λ��
    # z=reserve_volume%10�����ڱ�ʶ��Ʊ�׶����ͣ���ͬ�׶β�ͬ���Է��z=9����ʾ�����ֽ׶Σ����ڳ��У�z=1~5��Ӧ�׶λ��ֵĽ׶Σ�����z=1ʱ����ʾ��ǰ���ڽ׶�1��z=0��6~8����
    # ����1�����ڳ��еĹ�Ʊ������10%�Ĳ�λ����Ԥ���ض�reserve_volume=19�����ڳ��У�����10%�Ĳ�λ�����������������β����ز�λ���統�����볬��10%�Ĳ�λ���ڶ�����������ʵ�ʲ�������������Ʊʱ����ƻ���������volume����500�ɣ�����ʵ���µ�������������=volume-reserve_volume=481��
    # ����2��reserve_volume=31���׶�1�Ĺ�Ʊ���������ݲ��Ե������㣬��λ���������30%�Ĳ�λ
    # ����3��reserve_volume=52���׶�2�Ĺ�Ʊ���������ݲ��Ե������㣬��λ���������50%�Ĳ�λ
    # ����4��reserve_volume=23���׶�3�Ĺ�Ʊ���������ݲ��Ե������㣬��λ���������20%�Ĳ�λ
    # ����5��reserve_volume=15���׶�5�Ĺ�Ʊ���������ݲ��Ե������㣬��λ���������10%�Ĳ�λ
    """
    reserve_volume: int = 0

    # ��λ����
    max_account_position: float = 0.75   # �˻����ֱֲ���
    min_stock_num: int = 3               # ��С�ֲָ�����
    single_stock_cap: float = None       # ��������λ��������̬���㣩

    # ��ز���
    stop_loss_stock: float = -3.0        # ����ֹ�����
    stop_profit_stock: float = 6.0       # ����ֹӯ����
    partial_sell_ratio: float = 0.5      # ������������

    def __post_init__(self):
        self.single_stock_cap = round(self.max_account_position / self.min_stock_num, 2)


@dataclass
class RiskConfig:
    """���տ��Ʋ�������"""
    # ���̲���
    # ��֤Aָ��399317.SZ
    index_symbol: str = '399317.SZ'          # ��׼ָ��
    index_daily_n: int = 3                   # ָ�����ߴ���
    index_hourly_n: int = 5                  # ָ��Сʱ�ߴ���

    # �۶ϻ���
    circuit_breaker_threshold: float = -0.05  # ָ���������۶ϴ�����ֵ
    circuit_breaker_days: int = 3             # ָ�������۶�ʱ����֣�����ֹcircuit_breaker_days���������������

    # �˻����
    stop_loss_account: float = -2.0          # �˻���ֹ��
    stop_profit_account: float = 5.0         # �˻���ֹӯ
    stop_loss_account: float = 0.5          # �˻���ֹ��ʱ����λ���ٵĳ�������
    consecutive_loss_days: int = 3           # �˻�������consecutive_loss_days��������֣�����ֹcircuit_breaker_days���������������

    # �г�״̬
    volatility_window: int = 14              # �����ʼ��㴰��
    high_volatility_threshold: float = 0.03   # �߲�����ֵ
    low_liquidity_threshold: int = 1000000    # ����������ֵ
    """
    ���ķ��ԭ�򣬼�ֳ������壬���غ;�η�г������ؽ��׼��ɣ�������˼���У�����Ϊ�򣬻��»��Ż�ã�
    
    ��λ����
    �ֲֲ���������4���ֲ֣���Ͷ��ƫ�úͲ������ֲ֣��ɳ�����ֵ�����ڣ�����ͬ��ҵ�ֲ֣����ɲ�λ�������ܲ�λ��50%��
    ��λ����ԭ��
    1������1432�����⵹���������֣�
    2��ƽ��532�����������ʽ�����˽ᡣ
    
    ���շ���ʹ��ã�
    1���г����գ�ָ�����մ���µ�����-5%�������۶ϣ���֣�һ���ڲ����֣�  ��ָ��ȡ����300��
    2���г����գ�ָ���������մ������߼��������źţ���֣�һ���ڲ����֣�
    3��ָ�����գ�ָ�����յ�������2%�������߼��������źţ������λ������30%
    4��ָ�����գ�ָ�����ճ���Сʱ���������źţ������λ������60%
    5��ϵͳ���գ��˻�����3�������տ��������3%����֣�
    6��ϵͳ���գ��˻��������տ��𳬹�3%�������λ������30%��
    7�����ɷ��գ���һ���ɵ�����������ֵ��2%��������֣�
    8�����ɷ��գ���һ��Ʊ����Сʱ���������źţ����ɲ�λ������ԭ����50%��������������2��Сʱ�����K���źţ�����ȫ����
    9�����ɷ��գ���һ��Ʊ�������߼��������źţ�����ȫ��������
    
    ���չ��ԭ��
    1��ST�ɲ����롢������ٲ����롢Ǳ��ST������
    5������Ҵ����ֵ������
    6���۹ɴ�����������
    7�����ɴ�������̼���
    7���ƽ𱩵��������ɫ
    8��ʯ�ͱ����������Դ
    9�����飬�������
    10�����У�ƫ��ҽҩ
    
    ȥ�����õĽ���ϰ�ߣ�
    1�����ײ�����ֹ��
    2����Ԥϵͳֹ����������
    3����ֹ�𣬲��ϴ����ڸ�
    4���޽��׼ƻ����޽���Ŀ��
    5����������������������������������ĲƸ�������ʿ��������ȥ��
    6�����ɼ��ٳ��ʱ��׷�����롪�������������ߣ���Ϊ�ҽ�������������Զ������ڴ���ǰ
    7�����ɱ���տ�ʼʱ���������롪�������г�����µ��Ǻ����Ľ������Ⱥ����Ϊ��Ⱥ����Ϊδ�����ı�ǰ����ʽÿһ�ݱ���
    8������ͼ������Ϣ���������롪�����������׵õ�����Ϣ���Ǳ���������֪������Ϣ������û����Ե�޹ʵİ�
    9��ʱ�̶��̣�ҹ�����£����𽡿����������������壬����Ϊ���Զ������ף��������壬ÿ�������ܲ�һ��
    10���ƻ����ߣ�����ԥ������������ɱ�����ϣ����߼�һҹ��
    11��ϲ���ÿ��ģ�Ư����������Ư���ǹ��ЧӦ��Ϊ��������
    12���������Ӽȵ�ӯ�������������ӹ���׬����ÿһ��Ǯ
   
    
    
    ����ԭ��
    1�������г��������г�
    2�������ϴ���ʱֹ��
    3��׬ȡ��֪������ϵͳ�ڽ���
    4������˼ά������˼���������֪
    5����ֳ������壬���ĵȴ�
    6���Ը�ƫ�ã����ء��ٸС�֪�㳤�ֵ��Ը����̬������ͷ�󳭵׸��ʺϱ��ص��Ը�ǰ�ڶ������ϵ���ͷ�ɣ��µ�����30%�󣬳�������һ�������ϣ��µ�˥��ʱ���룻
    """


# -------------------- ����ģ��ʵ�� --------------------
class DataManager:
    """ͳһ���ݹ���"""
    def __init__(self, config: StrategyConfig):
        self.config = config
        # ���ݻ���ṹ�Ż�
        self.market_data = {
            'daily': defaultdict(pd.DataFrame),
            'hourly': defaultdict(pd.DataFrame),
            'ticks': defaultdict(list)
        }
        self.last_update = defaultdict(datetime)

    def update_market_data(self, symbol: str, period: str, context):
        """ͳһ���ݸ������"""
        try:
            if self._need_update(symbol, period):
                if period == '1d':
                    self._update_daily_data(symbol, context)
                elif period == '1h':
                    self._update_hourly_data(symbol, context)
                logger.info(f"{period}���ݸ������: {symbol}")
        except Exception as e:
            logger.error(f"���ݸ���ʧ�� {symbol} {period}: {str(e)}")

    def _update_daily_data(self, symbol: str, context):
        """�������ݸ���"""
        data = context.get_market_data(
            ['open','high','low','close','volume'],
            stock_code=symbol,
            period='1d',
            dividend_type='front_ratio',
            count=self.config.daily_n_days + 5  # ��ȡ5�շ�ֹ�߽�����
        )
        self.market_data['daily'][symbol] = data

    def _update_hourly_data(self, symbol: str, context):
        """Сʱ���ݸ���"""
        data = context.get_market_data(
            ['open','high','low','close','volume'],
            stock_code=symbol,
            period='1h',
            dividend_type='front_ratio',
            count=self.config.hourly_n_days * 6  # ��ÿ��6Сʱ����
        )
        self.market_data['hourly'][symbol] = data

    def _need_update(self, symbol: str, period: str) -> bool:
        """���ܸ��¼��"""
        last = self.last_update.get((symbol, period))
        delta = datetime.now() - last if last else timedelta.max
        return delta > timedelta(minutes=5)  # 5���Ӹ��¼��


class SignalGenerator:
    """ͳһ�ź�����"""
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.signal_cache = defaultdict(dict)

    def generate_signal(self, symbol: str, data: Dict[str, pd.DataFrame], k_multiple: int) -> Dict:
        """�������ź�����"""
        signals = {'buy': False, 'sell': False, 'strength': 0}

        try:
            # �����ź�
            daily_data = data['daily'].get(symbol, pd.DataFrame())
            if len(daily_data) >= self.config.daily_n_days:
                ref_high = daily_data['high'][-self.config.daily_n_days:-1].max()
                ref_low = daily_data['low'][-self.config.daily_n_days:-1].min()
                current = daily_data.iloc[-1]
                signals['daily_buy'] = current['high'] > ref_high
                signals['daily_sell'] = current['low'] < ref_low

            # Сʱ���ź�
            k_multiple = 6
            hourly_data = data['hourly'].get(symbol, pd.DataFrame())
            if len(hourly_data) >= self.config.hourly_n_days * k_multiple:
                ref_high = hourly_data['high'][-self.config.hourly_n_days*k_multiple:-1].max()
                ref_low = hourly_data['low'][-self.config.hourly_n_days*k_multiple:-1].min()
                current = hourly_data.iloc[-1]
                signals['hourly_buy'] = current['high'] > ref_high
                signals['hourly_sell'] = current['low'] < ref_low

            # �ź�ǿ�ȼ���
            signals['strength'] = self._calculate_strength(signals)
            return signals
        except Exception as e:
            logger.error(f"�ź�����ʧ�� {symbol}: {str(e)}")
            return signals

    def _calculate_strength(self, signals: Dict) -> int:
        """�ź�ǿ������"""
        strength = 0
        if signals['daily_buy']: strength += 2
        if signals['hourly_buy']: strength += 1
        if signals['daily_sell']: strength -= 2
        if signals['hourly_sell']: strength -= 1
        return strength


class RiskController:
    """ͳһ���տ���"""
    def __init__(self, strat_cfg: StrategyConfig, risk_cfg: RiskConfig):
        self.strat_cfg = strat_cfg
        self.risk_cfg = risk_cfg
        self.risk_status = {
            'circuit_breaker': False,
            'consecutive_loss': 0,
            'position_limits': defaultdict(float)
        }

    def check_market_risk(self, data: Dict) -> Dict:
        """�г������ռ��"""
        risk = {}
        # �۶ϼ��
        index_data = data['daily'].get(self.risk_cfg.index_symbol)
        if index_data is not None:
            current = index_data.iloc[-1]
            prev_close = index_data['close'].values.tolist()[-2]
            drawdown = (current['low'] - prev_close) / prev_close
            if drawdown <= self.risk_cfg.circuit_breaker_threshold:
                risk['circuit_breaker'] = True
        # �����ʼ��
        for symbol in data['daily'].keys():
            vols = self._calculate_volatility(data['daily'][symbol])
            if vols[-1] > self.risk_cfg.high_volatility_threshold:
                self.risk_status['position_limits'][symbol] = 0.5  # �޲�50%

        return risk

    def check_account_risk(self, positions: List, history: pd.DataFrame) -> Dict:
        """�˻������ռ��"""
        risk = {}
        # ����������
        recent_pnl = history['pnl'].tail(5)
        if all(recent_pnl < 0):
            self.risk_status['consecutive_loss'] += 1

        # ��ǰ�ֲַ���
        total_value = sum(p.market_value for p in positions)
        max_drawdown = min(p.drawdown for p in positions)
        if max_drawdown <= self.risk_cfg.stop_loss_account:
            risk['position_adjust'] = 'reduce'

        return risk

    def _calculate_volatility(self, data: pd.DataFrame) -> List[float]:
        """���㲨���ʣ�ATR��"""
        high, low, close = data['high'], data['low'], data['close']
        tr = np.maximum(high - low,
                        abs(high - close.shift(1)),
                        abs(low - close.shift(1)))
        return tr.rolling(self.risk_cfg.volatility_window).mean().tolist()


class PortfolioManager:
    """Ͷ����Ϲ���"""
    def __init__(self, strat_cfg: StrategyConfig):
        self.cfg = strat_cfg

    def adjust_position(self, signals: Dict, current_pos: Dict) -> Dict:
        """���ɵ���ָ��"""
        target_pos = {}
        # ���ɲ�λ����
        for symbol, pos in current_pos.items():
            print('signals=', signals)
            if symbol in list(signals.keys()):
                target = self._calculate_single_adjust(signals[symbol], pos)
                target_pos[symbol] = target
            else:
                print(symbol + ' is not exist')
        # ��ϼ����
        total = sum(target_pos.values())
        if total > self.cfg.max_account_position:
            scale = self.cfg.max_account_position / total
            target_pos = {k: v*scale for k, v in target_pos.items()}

        return target_pos

    def _calculate_single_adjust(self, symbol: str, signal: Dict, position: float) -> float:
        """���ɲ�λ����"""
        # �����ź�����
        if signal['strength'] >= 1:  # ǿ������
            target = min(position + self.cfg.single_stock_cap, self.cfg.single_stock_cap)
        elif signal['strength'] <= -1:  # ǿ������
            target = 0.0
        else:  # ����
            target = position

        # Ӧ�÷������
        if target > self.risk_status['position_limits'].get(symbol, 1.0):
            target = self.risk_status['position_limits'][symbol]

        return round(target, 4)


# -------------------- �������� --------------------
class TradingEngine:
    """ͳһ��������"""
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
        """���н�������"""
        try:
            # 1. ���ݸ���
            self._update_market_data(context)

            # 2. �ź�����
            signals = self._generate_signals()

            # 3. ���ռ��
            market_risk = self.risk_ctrl.check_market_risk(self.data_mgr.market_data)
            account_risk = self.risk_ctrl.check_account_risk(self.current_positions, context.history)

            # 4. ��Ϲ���
            target_pos = self.portfolio_mgr.adjust_position(signals, self.current_positions)

            # 5. ִ�н���
            self._execute_trades(target_pos, context)

        except Exception as e:
            logger.error(f"��������ִ��ʧ��: {str(e)}")
            self.trade_enabled = False

    def _update_market_data(self, context):
        """�����г�����"""
        for symbol in context.universe + [self.risk_cfg.index_symbol]:
            self.data_mgr.update_market_data(symbol, '1d', context)
            self.data_mgr.update_market_data(symbol, '1h', context)

    def _generate_signals(self) -> Dict:
        """����ȫ�г��ź�"""
        signals = {}
        for symbol in self.data_mgr.market_data['daily'].keys():
            signals[symbol] = self.signal_gen.generate_signal(
                symbol, self.data_mgr.market_data)
        return signals

    def _execute_trades(self, target_pos: Dict, context):
        """ִ�е��ֲ���"""
        for symbol, target in target_pos.items():
            current = self.current_positions.get(symbol, 0.0)
            if abs(target - current) < 0.01:  # ����΢С����
                continue

            # ���������
            adj_amount = round(target - current, 4)
            if adj_amount > 0:
                self._place_order(symbol, adj_amount, 'buy', context)
            else:
                self._place_order(symbol, abs(adj_amount), 'sell', context)

    def _place_order(self, symbol: str, amount: float, side: str, context):
        """ί���µ�"""
        try:
            # ��ȡʵʱ�۸�
            tick = context.get_full_tick([symbol])[symbol]
            price = tick.ask1 if side == 'buy' else tick.bid1

            # ת������Ϊʵ�ʹ���
            units = max(int(amount * 1e4 / price), self.strat_cfg.min_trade_units)

            # ���ý��׽ӿ�
            order_code = 23 if side == 'buy' else 24
            passorder(order_code, 1101, context.account_id, symbol,
                      11, price, units, 'Systematic', 1, self._gen_order_id(symbol, order_code), context)

            logger.info(f"���ύ���� {symbol} {side} {units}�� @{price}")
        except Exception as e:
            logger.error(f"�µ�ʧ�� {symbol}: {str(e)}")

    def _gen_order_id(self, symbol: str, order_code: str) -> str:
        """����Ψһ����ID"""
        # order_code = 23 if side == 'buy' else 24
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{symbol}_{order_code}_{timestamp}_{random.randint(1000,9999)}"
    """
    def _gen_order_id(self) -> str:
        
        return f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}"
    """


# -------------------- ��ʼ������ѭ�� --------------------
def init(ContextInfo):
    """���Գ�ʼ��"""
    try:
        # ��ʼ������
        strat_cfg = StrategyConfig()
        risk_cfg = RiskConfig()

        # ������������
        ContextInfo.engine = TradingEngine(strat_cfg, risk_cfg)

        # ���ó�ʼ�ֲ�
        positions = get_trade_detail_data(ContextInfo.account_id,
                                          ContextInfo.account_type,
                                          'POSITION')
        position_stocks = []
        for p in positions:
            symbol = f"{p.m_strInstrumentID}.{'SH' if p.m_strExchangeID == 'SSE' else 'SZ'}"
            position_stocks.append(symbol)
            ContextInfo.engine.current_positions[symbol] = p.m_dMarketValue
        index_list = ['399317.SZ']
        # ��֤Aָ��399317.SZ
        # ����300��000300.SH
        # ��֤50��000016.SH
        # ��֤���У�000134.SH
        # index_list = ['399317.SZ']# ['399381.SZ', '399382.SZ', '399383.SZ', '399384.SZ', '399385.SZ', '399386.SZ']
        exit_stocks = ['300742.SZ', '000918.SZ', '002002.SZ', '600393.SH', '600466.SH', '002087.SZ', '000961.SZ', '000413.SZ', '002288.SZ', '000666.SZ', '600077.SH', '600297.SH', '600836.SH', '000540.SZ', '000836.SZ', '002435.SZ', '000996.SZ', '002013.SZ', '600277.SH', '600213.SH', '002621.SZ', '600565.SH', '000023.SZ', '000671.SZ', '300116.SZ', '000046.SZ', '600321.SH', '002280.SZ', '300262.SZ', '600766.SH', '002502.SZ', '000667.SZ', '000416.SZ', '000861.SZ', '601258.SH', '600220.SH', '002341.SZ', '002505.SZ', '002610.SZ', '002665.SZ', '000976.SZ', '603133.SH', '002325.SZ', '002503.SZ', '300799.SZ', '600647.SH', '600823.SH', '002118.SZ', '000982.SZ', '603555.SH', '002308.SZ']
        universe_stocks = list(set(position_stocks+index_list).difference(set(exit_stocks)))
        print('universe_stocks=', universe_stocks)
        ContextInfo.set_universe(universe_stocks)
        logger.info("����ϵͳ��ʼ�����")
    except Exception as e:
        logger.error(f"��ʼ��ʧ��: {str(e)}")
        ContextInfo.engine.trade_enabled = False


def handlebar(ContextInfo):
    """������ѭ��"""
    if not ContextInfo.engine.trade_enabled:
        return

    try:
        ContextInfo.engine.run_cycle(ContextInfo)
    except Exception as e:
        logger.error(f"��ѭ���쳣: {str(e)}")
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
        # �����г���������
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
        # ��ʼ�ֲ�
        self.engine.current_positions = {
            '600519.SH': 0.3,
            '000001.SZ': 0.5
        }
        # ���������ź�
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