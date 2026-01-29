"""
Contrarian Short Bot (Simulation 3) for Bitcoin 15min markets.
*Restored Full Architecture*

Strategy: ABSORPTION TRAP (Short/DOWN)
1. Target: DOWN/NO Tokens.
2. Trigger: High Buying Volume (MFI >= 90) + Price Stalled at Lows (Stoch K <= 10).
3. Sizing: 5% Risk. 100% size if Stoch <= 5 (Deep Trap), 75% if Stoch <= 10.
4. Exit: T+4 (Early Proof), T+6 (Hard Stop), or MFI Normalization (< 50).

*Includes full indicator suite and UP logic infrastructure.*
"""

import asyncio
import logging
import re
import time
import math
import csv
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Dict
from dotenv import load_dotenv
load_dotenv()

import httpx

from .config import load_settings
from .config_validator import ConfigValidator
from .logger import setup_logging, print_header, print_success, print_error
from .lookup import fetch_market_from_slug
from .risk_manager import RiskManager, RiskLimits
from .statistics import StatisticsTracker
from .trading import (
    get_client,
    place_order,
    extract_order_id,
    wait_for_terminal_order,
    cancel_orders,
)
from .utils import GracefulShutdown
from .wss_market import MarketWssClient

logger = logging.getLogger(__name__)

# --- Market Finding Logic ---
def find_current_btc_15min_market() -> str:
    """Find the current active BTC 15min market on Polymarket via HTML scraping."""
    logger.info("Searching for current BTC 15min market...")
    try:
        page_url = "https://polymarket.com/crypto/15M"
        resp = httpx.get(page_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()
        
        pattern = r'btc-updown-15m-(\d+)'
        matches = re.findall(pattern, resp.text)
        if not matches:
            raise RuntimeError("No BTC 15min market slugs found")
        
        now_ts = int(datetime.now().timestamp())
        all_ts = sorted(int(ts) for ts in matches)
        open_ts = [ts for ts in all_ts if ts <= now_ts < ts + 900]  # Active: start <= now < end
        
        if not open_ts:
            raise RuntimeError("No active BTC 15min market (retry later)")
        
        chosen_ts = open_ts[0]  # Earliest active
        slug = f"btc-updown-15m-{chosen_ts}"
        logger.info(f"✅ Active market: {slug} (now={now_ts}, ts={chosen_ts})")
        return slug
    except Exception as e:
        logger.error(f"Market search failed: {e}")
        raise

def normalize_size(size: float, min_size: float = 1.0) -> float:
    """Rounds size down to 3 decimals and ensures min_size."""
    if size < min_size:
        return 0.0
    # Quantize to 3 decimal places (standard for many CLOBs)
    d = Decimal(str(size)).quantize(Decimal("0.001"), rounding=ROUND_DOWN)
    return float(d)

# --- Strategy State Tracker ---
class PositionState:
    """Tracks the state of a single side (UP or DOWN)."""
    def __init__(self, token_id, side_name):
        self.token_id = token_id
        self.side_name = side_name
        self.total_shares = 0.0
        self.total_cost = 0.0
        self.avg_price = 0.0
        self.btc_entry_price = 0.0      # BTC price at moment of entry
        self.btc_stop_price = 0.0       # The BTC price that triggers the exit

        # --- New Management Fields ---
        self.initial_size = 0.0         # Track starting size for scaling math
        self.size_remaining_pct = 0.0   # Track % remaining (1.0 -> 0.75 -> 0.50 -> 0.25)
        self.stop_loss = 0.0            # Dynamic Stop Price
        self.is_breakeven_set = False   # Breakeven flag
        self.btc_shadow_stop = 0.0
        
        # Strategy Fields
        self.entry_time = 0.0
        self.entry_price = 0.0
        self.entry_stoch_k = 0.0
        self.high_price_seen = 0.0
        
        # New Checkpoint Flags
        self.checked_t4 = False       # Has T+4 Early Proof been checked?
        self.checked_t6 = False       # Has T+6 Hard Stop been checked?
        self.taken_profit = False     # Has 25% TP been triggered?
        self.realized_pnl = 0.0

    def add_buy(self, price, size, stoch_k, btc_price=0.0):
        cost = price * size
        self.total_cost += cost
        self.total_shares += size
        self.avg_price = self.total_cost / self.total_shares if self.total_shares > 0 else 0.0
        
        if self.entry_time == 0.0:
            self.entry_time = time.time()
            self.entry_price = price
            self.entry_stoch_k = stoch_k
            self.high_price_seen = price
            self.btc_entry_price = btc_price 
            self.btc_stop_price = 0.0

            # Initialize New Management Fields
            self.initial_size = size
            self.size_remaining_pct = 1.0
            self.is_breakeven_set = False
            self.stop_loss = 0.0 # Will be set by ATR logic immediately after

            # Reset flags on new entry
            self.checked_t4 = False
            self.checked_t6 = False
            self.taken_profit = False

    def sell_shares(self, size, price):
        if size > self.total_shares:
            size = self.total_shares
        
        revenue = size * price
        cost_basis = size * self.avg_price
        pnl = revenue - cost_basis
        
        self.total_shares -= size
        self.total_cost -= cost_basis
        self.realized_pnl += pnl
        
        if self.total_shares < 0.1: 
            self.reset()
            
    def reset(self):
        self.total_shares = 0.0
        self.total_cost = 0.0
        self.avg_price = 0.0
        self.entry_time = 0.0
        self.entry_price = 0.0
        self.entry_stoch_k = 0.0
        self.high_price_seen = 0.0
        self.checked_t4 = False
        self.checked_t6 = False
        self.taken_profit = False

        # Reset New Fields
        self.initial_size = 0.0
        self.size_remaining_pct = 0.0
        self.stop_loss = 0.0
        self.btc_shadow_stop = 0.0
        self.is_breakeven_set = False
        self.checked_t4 = False
        self.checked_t6 = False
        self.btc_entry_price = 0.0
        self.btc_stop_price = 0.0

# --- Main Bot Class ---
class ContrarianShortBot:
    """Bot implementing the Contrarian Short Strategy (with Full Architecture)."""
    
    def __init__(self, settings):
        self.settings = settings
        self.client = get_client(settings)
        
        # --- STRATEGY PARAMETERS (Simulation 3) ---
        # Logic: ABSORPTION TRAP
        # High MFI (Buying Pressure) + Low Stoch (Price Stalled) = DOWN Signal
        self.mfi_entry_min = float(os.getenv("SHORT_MFI_ENTRY_MIN", 90.0))
        self.stoch_entry_max = float(os.getenv("SHORT_STOCH_ENTRY_MAX", 10.0))
        
        # Optional: UP Logic (Panic Exhaustion) if needed
        self.long_mfi_entry_max = float(os.getenv("LONG_MFI_ENTRY_MAX", 8.0))
        self.long_stoch_entry_max = float(os.getenv("LONG_STOCH_ENTRY_MAX", 20.0))
        
        self.min_time_remaining = float(os.getenv("STRATEGY_MIN_TIME_REMAINING", 5))
        self.risk_per_trade = float(os.getenv("STRATEGY_RISK_PER_TRADE_PERCENT", 0.05))
        
        # Sizing: Full size if Stoch is extremely low (<= 5)
        self.stoch_full_size_limit = 5.0 
        self.size_reduction_mult = 0.75
        
        # --- Exit Settings ---
        self.atr_multiplier = float(os.getenv("STRATEGY_ATR_MULTIPLIER", 2.0))
        self.rsi_exit_long = float(os.getenv("STRATEGY_RSI_EXIT_LONG", 70.0))
        self.rsi_exit_short = float(os.getenv("STRATEGY_RSI_EXIT_SHORT", 30.0))

        self.t4_checkpoint_min = float(os.getenv("STRATEGY_T4_MINUTES", 4))
        self.t4_min_pnl_pct = float(os.getenv("STRATEGY_T4_PNL_PERCENT", 0.15))
        self.t4_price_bump = float(os.getenv("STRATEGY_T4_PRICE_BUMP", 0.02))
        
        self.t6_exit_min = float(os.getenv("STRATEGY_T6_MINUTES", 6))
        self.t6_min_pnl_pct = float(os.getenv("STRATEGY_T6_PNL_PERCENT", 0.25))
        self.t6_price_bump = float(os.getenv("STRATEGY_T6_PRICE_BUMP", 0.03))
        
        self.tp_trigger_pct = float(os.getenv("STRATEGY_TP_PERCENT", 0.25))

        # Emergency Exit: MFI Normalization
        self.mfi_normalization_exit = float(os.getenv("STRATEGY_MFI_PANIC_EXIT", 50.0))

        # --- Statistics & Risk ---
        self.stats_tracker = None
        if settings.enable_stats:
            try:
                self.stats_tracker = StatisticsTracker(log_file=settings.trade_log_file)
            except Exception as e:
                logger.warning(f"Failed to initialize statistics tracker: {e}")
        
        self.risk_manager = None
        if settings.max_daily_loss > 0:
            risk_limits = RiskLimits(
                max_daily_loss=settings.max_daily_loss,
                max_position_size=settings.max_position_size,
                max_trades_per_day=settings.max_trades_per_day,
                min_balance_required=settings.min_balance_required,
                max_balance_utilization=settings.max_balance_utilization,
            )
            self.risk_manager = RiskManager(risk_limits)
        
        # --- Market Discovery ---
        try:
            market_slug = find_current_btc_15min_market()
        except Exception:
            if settings.market_slug:
                market_slug = settings.market_slug
            else:
                raise RuntimeError("Could not find market and no slug configured.")
        
        self.market_slug = market_slug
        market_info = fetch_market_from_slug(market_slug)
        self.min_order_size = float(market_info.get("min_order_size", 0.1))
        
        self.market_id = str(market_info["market_id"])
        self.yes_token_id = str(market_info["yes_token_id"])
        self.no_token_id = str(market_info["no_token_id"])
        
        match = re.search(r'btc-updown-15m-(\d+)', market_slug)
        market_start = int(match.group(1)) if match else None
        self.market_end_timestamp = market_start + 900 if market_start else None

        # --- Global Filters ---
        should_skip_first = os.getenv("SKIP_FIRST_MARKET", "true").lower() == "true"
        if not hasattr(self.__class__, "first_market_seen"):
            self.__class__.first_market_seen = self.market_id
        else:
            self.is_first_market = (self.__class__.first_market_seen == self.market_id)
        in_first_market = (self.__class__.first_market_seen == self.market_id)
        self.is_first_market = (in_first_market and should_skip_first)

        # --- State Tracking (Full) ---
        self.strategies: Dict[str, PositionState] = {
            "UP": PositionState(self.yes_token_id, "UP"),
            "DOWN": PositionState(self.no_token_id, "DOWN")
        }
        
        self.has_traded_this_market = False
        self.cached_balance = None
        self.sim_balance = float(os.getenv("SIM_BALANCE", 100.0))
        self.sim_start_balance = self.sim_balance
        self.trades_executed = 0
        self._last_log_ts = 0.0
        self.last_csv_log = 0.0

        self.transaction_file = "transactions_full.csv"
        self.strike_price = 0.0
        self.is_searching = False
        self.latest_up_bid = 0.0
        self.latest_up_ask = 0.0
        self.latest_down_bid = 0.0
        self.latest_down_ask = 0.0
        self.latest_btc_price = 0.0
        self.last_valid_indicators = None
        self.indicator_lock = asyncio.Lock()
        self.last_indicator_update = 0.0
        self.shared_client = httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"})
        self.price_lock = False
        self.summary_shown = False
        
    async def close(self):
        """Gracefully close network connections."""
        if hasattr(self, 'shared_client') and self.shared_client:
            await self.shared_client.aclose()
            logger.info("🔌 Network connections closed.")

    def get_time_remaining(self) -> str:
        if not self.market_end_timestamp: return "Unknown"
        now = time.time()
        remaining = self.market_end_timestamp - now
        if remaining <= 0: return "CLOSED"
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        return f"{minutes}m {seconds}s"

    def get_time_remaining_minutes(self) -> float:
        if not self.market_end_timestamp: return 0
        rem = self.market_end_timestamp - time.time()
        return rem / 60.0

    def get_balance(self) -> float:
        if self.settings.dry_run: return self.sim_balance
        from .trading import get_balance
        return get_balance(self.settings)
    
    def log_transaction(self, side, action, price, size, value, pnl, start_bal, curr_bal, indicators=None):
        """Append transaction details + FULL indicators to CSV."""
        file_exists = os.path.isfile(self.transaction_file)
        if indicators is None: indicators = {}
        
        try:
            with open(self.transaction_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                
                # Full Indicator Set
                ind_keys = [
                    "BTC_Price", "Strike_Price",
                    "UP_Bid", "UP_Ask", "DOWN_Bid", "DOWN_Ask",
                    "RSI", "BB_Upper", "BB_Lower", "BB_Middle", "BB_Width", "BB_PctB", 
                    "ATR", "Stoch_K", "VWAP", "MACD", "CCI", "MFI", "ROC", "Pivot"
                ]
                
                if not file_exists:
                    writer.writerow([
                        "Timestamp", "Market", "Side", "Action", 
                        "Price", "Size", "Total_Value", "PnL", 
                        "Start_Balance", "Current_Balance"
                    ] + ind_keys) 
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                row = [
                    timestamp, self.market_slug, side, action,
                    f"{price:.4f}", f"{size:.2f}", f"{value:.2f}", 
                    f"{pnl:.2f}", f"{start_bal:.2f}", f"{curr_bal:.2f}"
                ]
                for k in ind_keys:
                    row.append(indicators.get(k, ""))
                    
                writer.writerow(row)

                if action == "MARKET_CLOSE":
                    writer.writerow([]) 
                    writer.writerow([]) 

        except Exception as e:
            logger.error(f"Failed to write log: {e}")

    async def get_indicators(self, use_shared_client=True, force_update=False) -> dict:
        """Fetch data and calculate FULL indicators (cached)."""
        keys = ["RSI", "BB_Upper", "BB_Lower", "BB_Middle", "BB_Width", "BB_PctB", 
                "ATR", "Stoch_K", "VWAP", "MACD", "CCI", "MFI", "ROC", "Pivot"]
        defaults = {k: 0.0 for k in keys}

        async with self.indicator_lock:
            now = time.time()
            if not force_update and self.last_valid_indicators and (now - self.last_indicator_update < 5.0):
                return self.last_valid_indicators

            try:
                endpoints = ["https://api.binance.com", "https://api.binance.us"]
                resp_1m = None
                resp_1d = None
                
                client_to_use = self.shared_client if use_shared_client else httpx.AsyncClient()
                
                if use_shared_client:
                     for base_url in endpoints:
                        try:
                            r1 = await client_to_use.get(f"{base_url}/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=100", timeout=2.0)
                            r2 = await client_to_use.get(f"{base_url}/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=2", timeout=2.0)
                            if r1.status_code == 200:
                                resp_1m = r1
                                resp_1d = r2
                                break
                        except Exception:
                            continue
                
                if not resp_1m or resp_1m.status_code != 200:
                    return self.last_valid_indicators if self.last_valid_indicators else defaults

                data = resp_1m.json()
                if not isinstance(data, list): return defaults

                df = pd.DataFrame(data, columns=['Time','Open','High','Low','Close','Vol','x','y','z','a','b','c'])
                cols = ['Close', 'High', 'Low', 'Vol']
                for c in cols:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                
                # === LIVE INJECTION ===
                if self.latest_btc_price > 0:
                    df.iloc[-1, df.columns.get_loc('Close')] = self.latest_btc_price
                    if self.latest_btc_price > df.iloc[-1, df.columns.get_loc('High')]:
                        df.iloc[-1, df.columns.get_loc('High')] = self.latest_btc_price
                    if self.latest_btc_price < df.iloc[-1, df.columns.get_loc('Low')]:
                        df.iloc[-1, df.columns.get_loc('Low')] = self.latest_btc_price
                
                close = df['Close']
                high = df['High']
                low = df['Low']
                
                # --- CALCULATIONS (RESTORED ALL) ---
                delta = close.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

                bb_middle = close.rolling(20).mean()
                std20 = close.rolling(20).std()
                bb_upper = bb_middle + (std20 * 2)
                bb_lower = bb_middle - (std20 * 2)
                bb_width = (bb_upper - bb_lower) / bb_middle
                bb_pct_b = (close - bb_lower) / (bb_upper - bb_lower)

                tr1 = high - low
                tr2 = (high - close.shift()).abs()
                tr3 = (low - close.shift()).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

                lowest_low = low.rolling(14).min()
                highest_high = high.rolling(14).max()
                stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
                
                vwap = (df['Vol'] * (high + low + close) / 3).rolling(20).sum() / df['Vol'].rolling(20).sum()

                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26

                tp = (high + low + close) / 3
                tp_sma = tp.rolling(20).mean()
                mad = (tp - tp_sma).abs().rolling(20).mean()
                cci = (tp - tp_sma) / (0.015 * mad)

                typical_price = (high + low + close) / 3
                money_flow = typical_price * df['Vol']
                pos_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
                neg_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
                mfi = 100 - (100 / (1 + (pos_flow / neg_flow)))

                roc = ((close - close.shift(9)) / close.shift(9)) * 100

                pivot = 0.0
                if resp_1d and resp_1d.status_code == 200:
                    d_data = resp_1d.json()
                    if len(d_data) >= 2:
                        prev = d_data[-2]
                        p_high, p_low, p_close = float(prev[2]), float(prev[3]), float(prev[4])
                        pivot = (p_high + p_low + p_close) / 3

                def get_val(series):
                    v = series.iloc[-1]
                    return round(float(v), 4) if not pd.isna(v) else 0.0

                result = {
                    "BTC_Price": get_val(close),
                    "RSI": get_val(rsi),
                    "BB_Upper": get_val(bb_upper),
                    "BB_Lower": get_val(bb_lower),
                    "BB_Middle": get_val(bb_middle), 
                    "BB_Width": get_val(bb_width),  
                    "BB_PctB": get_val(bb_pct_b),   
                    "ATR": get_val(atr),
                    "Stoch_K": get_val(stoch_k),
                    "VWAP": get_val(vwap),
                    "MACD": get_val(macd),
                    "CCI": get_val(cci),
                    "MFI": get_val(mfi),
                    "ROC": get_val(roc),
                    "Pivot": round(pivot, 4)
                }
                
                self.last_valid_indicators = result
                self.last_indicator_update = time.time()
                return result
                
            except Exception as e:
                logger.error(f"❌ Indicator Calc Failed: {e}")
                if self.last_valid_indicators:
                     return self.last_valid_indicators
                return defaults
        
    async def execute_order(self, side_key, token_id, price, size, is_buy: bool, stoch_k=0.0):
        cost = price * size
        action = "BUY" if is_buy else "SELL"
        pnl = 0.0
        
        # Balance & Risk Checks (Real Only)
        if not self.settings.dry_run:
            bal = self.cached_balance if self.cached_balance else self.get_balance()
            if is_buy and bal < cost:
                logger.error(f"❌ Insufficient balance: Need ${cost:.2f}, Have ${bal:.2f}")
                return False
            
            if is_buy and self.risk_manager:
                can_trade, reason = self.risk_manager.can_trade(cost, bal)
                if not can_trade:
                    logger.warning(f"⚠️ Risk Blocked: {reason}")
                    return False

        # --- LOGGING ---
        logger.info("=" * 70)
        logger.info(f"⚡ EXECUTING {action} ({side_key})")
        logger.info("-" * 70)
        logger.info(f"Price:            ${price:.4f}")
        logger.info(f"Size:             {size:.2f} shares")
        logger.info(f"Total Value:      ${cost:.2f}")
        
        st = self.strategies[side_key]
        if not is_buy:
             pnl = (price - st.avg_price) * size
             logger.info(f"Avg Cost Basis:   ${st.avg_price:.4f}")
             logger.info(f"Est. PnL:         ${pnl:.2f}")

        # Balance Variables
        if self.settings.dry_run:
            s_bal = self.sim_start_balance
            c_bal_before = self.sim_balance - cost if is_buy else self.sim_balance + cost
        else:
            s_bal = 0.0 
            c_bal_before = self.cached_balance if self.cached_balance else self.get_balance()

        logger.info("-" * 70)
        logger.info(f"Start Balance:    ${s_bal:.2f}")
        logger.info(f"New Balance:      ${c_bal_before:.2f}")
        logger.info("=" * 70)

        inds = await self.get_indicators()

        # --- SIMULATION ---
        if self.settings.dry_run:
            logger.info("🔸 SIMULATION MODE - Executing virtually")
            if is_buy: self.sim_balance -= cost
            else: self.sim_balance += cost
            
            self.log_transaction(side_key, action, price, size, cost, pnl, s_bal, self.sim_balance, indicators=inds)

            if self.stats_tracker:
                self.stats_tracker.record_trade(
                    market_slug=self.market_slug,
                    price_up=price if side_key == "UP" else 0,
                    price_down=price if side_key == "DOWN" else 0,
                    total_cost=cost,
                    order_size=size,
                    filled=True
                )
            return True

        # --- REAL ORDER ---
        try:
            res = place_order(
                self.settings, 
                side=action, 
                token_id=token_id, 
                price=price, 
                size=size, 
                tif=self.settings.order_type
            )
            order_id = extract_order_id(res)
            
            if not order_id:
                logger.error(f"❌ Order submission failed: {res}")
                return False

            logger.info(f"⏳ Order {order_id} submitted. Waiting for fill...")
            state = wait_for_terminal_order(self.settings, order_id)
            
            if state.get("filled"):
                fill_price = float(state.get("avg_price", price))
                logger.info(f"✅ ORDER FILLED @ ${fill_price:.4f}")
                
                self.cached_balance = self.get_balance()
                self.trades_executed += 1
                
                real_cost = fill_price * size
                real_pnl = 0.0
                if not is_buy:
                    real_pnl = (fill_price - st.avg_price) * size
                
                self.log_transaction(side_key, action, fill_price, size, real_cost, real_pnl, s_bal, self.cached_balance, indicators=inds)

                if self.stats_tracker:
                    self.stats_tracker.record_trade(
                        market_slug=self.market_slug,
                        price_up=fill_price if side_key == "UP" else 0,
                        price_down=fill_price if side_key == "DOWN" else 0,
                        total_cost=real_cost,
                        order_size=size,
                        order_ids=[order_id],
                        filled=True
                    )
                if not is_buy and self.risk_manager:
                    self.risk_manager.record_trade_result(real_pnl)
                return True
            else:
                logger.warning(f"⚠️ Order not filled ({state.get('status')}). Cancelling...")
                cancel_orders(self.settings, [order_id])
                return False

        except Exception as e:
            logger.error(f"❌ Execution Error: {e}")
            return False
        
    async def check_exits(self, side_key: str, st: PositionState, current_price: float, current_rsi: float, pnl_pct: float) -> bool:
        """
        New Consolidated Exit Logic:
        1. Scaling: Bank safety profit (25% size) at +15% PnL.
        2. Profit Taking: Decoupled from PnL. Only exit if Technical Mean Reversion occurs.
        Returns: True if position was fully closed, False otherwise.
        """
        
        # 1. RETRIEVE THRESHOLDS
        rsi_target = self.rsi_exit_long if side_key == 'UP' else self.rsi_exit_short
        mean_reversion_midpoint = 50.0
        
        # 2. SCALING LOGIC (Bank "Safety Profit" only)
        # Trigger: > 15% PnL. Action: Sell 25% of INITIAL size.
        # Condition: Must have > 75% remaining (meaning we haven't scaled yet).
        if pnl_pct >= 0.15 and st.size_remaining_pct > 0.75:
            scale_amount = 0.25
            
            # Calculate size
            scale_size = st.initial_size * scale_amount
            final_sell_size = normalize_size(scale_size)
            
            # Safety check to never sell more than we hold
            if final_sell_size > st.total_shares:
                final_sell_size = st.total_shares

            if final_sell_size > 0:
                logger.info(f"💰 SCALING ({side_key}): +15% PnL hit. Banking 25% safety profit.")
                await self.execute_order(side_key, st.token_id, current_price, final_sell_size, False)
                st.sell_shares(final_sell_size, current_price)
                st.size_remaining_pct = 0.75 # Update tracker
            
            return False # Position still open

        # 3. PROFIT TAKING LOGIC (Technical Exit)
        technical_exit_signal = False
        reason = ""

        if side_key == 'UP':
            # A. Lotto Hit: RSI goes parabolic (e.g. > 70)
            if current_rsi >= rsi_target:
                technical_exit_signal = True
                reason = f"RSI Target Hit ({current_rsi:.1f} >= {rsi_target})"
            
            # B. Mean Reversion: We have profit (>25%) AND momentum has reset to neutral (RSI 50)
            elif pnl_pct > 0.25 and current_rsi >= mean_reversion_midpoint:
                technical_exit_signal = True
                reason = f"Mean Reversion (PnL {pnl_pct*100:.1f}% & RSI {current_rsi:.1f})"

        elif side_key == 'DOWN':
            # A. Lotto Hit: RSI crashes (e.g. < 30) - Underlying dumped, our NO shares are up.
            if current_rsi <= rsi_target:
                technical_exit_signal = True
                reason = f"RSI Target Hit ({current_rsi:.1f} <= {rsi_target})"
            
            # B. Mean Reversion: Profit (>25%) AND momentum reset (RSI 50)
            elif pnl_pct > 0.25 and current_rsi <= mean_reversion_midpoint:
                technical_exit_signal = True
                reason = f"Mean Reversion (PnL {pnl_pct*100:.1f}% & RSI {current_rsi:.1f})"

        # 4. EXECUTE FULL EXIT
        if technical_exit_signal:
            logger.info(f"💎 TECHNICAL EXIT ({side_key}): {reason}. Closing Position.")
            await self.execute_order(side_key, st.token_id, current_price, st.total_shares, False)
            st.sell_shares(st.total_shares, current_price)
            return True # Position Closed

        return False
        
    async def process_strategy(self, side_key: str, book: dict):
        """
        DUAL STRATEGY PROCESSING:
        1. DOWN (Short): Absorption Trap (MFI >= 90, Stoch <= 10)
        2. UP (Long): Panic Exhaustion (MFI <= 8, Stoch <= 20)
        """
        st = self.strategies[side_key]
        minutes_remaining = self.get_time_remaining_minutes()
        
        # --- 0. FETCH INDICATORS ---
        inds = await self.get_indicators()
        mfi = inds.get("MFI", 50.0)
        stoch_k = inds.get("Stoch_K", 50.0)
        atr_value = inds.get("ATR", 0.0)
        
        best_ask = book.get("best_ask")
        best_bid = book.get("best_bid")
        
        if best_ask is None or best_bid is None: return

        # ==========================================
        # 1. ENTRY LOGIC (Only if no position)
        # ==========================================
        can_buy = (
            not self.has_traded_this_market and
            self.strategies["UP"].total_shares == 0 and 
            self.strategies["DOWN"].total_shares == 0 and
            not self.is_first_market and 
            minutes_remaining >= self.min_time_remaining
        )

        if can_buy:
            bal = self.cached_balance if not self.settings.dry_run else self.sim_balance
            risk_amt = bal * self.risk_per_trade

            current_btc_price = inds.get("BTC_Price", 0.0)
            atr_value = inds.get("ATR", 0.0)

            # --- A. CONTRARIAN SHORT (DOWN) ---
            if side_key == "DOWN":
                if mfi >= self.mfi_entry_min and stoch_k <= self.stoch_entry_max:
                    # Sizing: Full size if Stoch <= 5 (Deep Absorption)
                    if stoch_k <= self.stoch_full_size_limit:
                        final_value = risk_amt
                        desc = "100% Size (Stoch <= 5)"
                    else:
                        final_value = risk_amt * self.size_reduction_mult
                        desc = "75% Size (Stoch 5-10)"
                    
                    raw_size = final_value / best_ask
                    buy_size = normalize_size(raw_size, self.min_order_size)
                    
                    if buy_size > 0:
                        logger.info(f"🚨 SHORT SIGNAL (DOWN): MFI {mfi:.1f}, Stoch {stoch_k:.1f} | {desc}")
                        success = await self.execute_order(side_key, st.token_id, best_ask, buy_size, True, stoch_k=stoch_k)
                        if success:
                            st.add_buy(best_ask, buy_size, stoch_k, btc_price=current_btc_price)
                            self.has_traded_this_market = True

                            if atr_value > 0:
                                st.btc_stop_price = current_btc_price + (atr_value * self.atr_multiplier)
                                logger.info(f"🛡️ BTC SHADOW STOP SET ({side_key}): ${st.btc_stop_price:,.2f} (BTC: ${current_btc_price:,.2f} + 2xATR)")



            # --- B. PANIC LONG (UP) ---
            elif side_key == "UP":
                if mfi <= self.long_mfi_entry_max and stoch_k <= self.long_stoch_entry_max:
                    if stoch_k <= 10.0:
                        final_value = risk_amt
                        desc = "100% Size (Panic Stoch <= 10)"
                    else:
                        final_value = risk_amt * self.size_reduction_mult
                        desc = "75% Size (Panic Stoch 10-20)"

                    raw_size = final_value / best_ask
                    buy_size = normalize_size(raw_size, self.min_order_size)

                    if buy_size > 0:
                        logger.info(f"🟢 LONG SIGNAL (UP): MFI {mfi:.1f}, Stoch {stoch_k:.1f} | {desc}")
                        success = await self.execute_order(side_key, st.token_id, best_ask, buy_size, True, stoch_k=stoch_k)
                        if success:
                            st.add_buy(best_ask, buy_size, stoch_k, btc_price=current_btc_price)
                            self.has_traded_this_market = True
                            if atr_value > 0:
                                st.btc_stop_price = current_btc_price - (atr_value * self.atr_multiplier)
                                logger.info(f"🛡️ BTC SHADOW STOP SET ({side_key}): ${st.btc_stop_price:,.2f} (BTC: ${current_btc_price:,.2f} - 2xATR)")

        # ==========================================
        # 2. EXIT LOGIC (If position exists)
        # ==========================================
        if st.total_shares > 0:
            
            # Update High Water Mark
            if best_bid > st.high_price_seen:
                st.high_price_seen = best_bid

            # Data for Calculations
            current_price = best_bid
            atr_value = inds.get("ATR", 0.0)
            rsi_value = inds.get("RSI", 50.0)
            
            # Calculate Profit %
            price_gain = current_price - st.entry_price
            pnl_pct = (price_gain / st.entry_price) if st.entry_price > 0 else 0.0

            # ------------------------------------------------------------------
            # A. INITIALIZE STOP LOSS (If not set)
            # ------------------------------------------------------------------
            if st.stop_loss == 0.0:
                stop_pct = 0.30 
                st.stop_loss = st.entry_price * (1.0 - stop_pct)
                logger.info(f"🛡️ INITIAL STOP SET ({side_key}): ${st.stop_loss:.3f} (Entry ${st.entry_price:.3f} - 30%)")

            # ------------------------------------------------------------------
            # B. CHECK HARD STOP LOSS (Execution Priority #1)
            # ------------------------------------------------------------------
            if st.stop_loss > 0 and current_price <= st.stop_loss:
                logger.warning(f"🛑 STOP LOSS HIT ({side_key}): Price ${current_price:.3f} <= Stop ${st.stop_loss:.3f}")
                await self.execute_order(side_key, st.token_id, current_price, st.total_shares, False)
                st.sell_shares(st.total_shares, current_price)
                return

            # ------------------------------------------------------------------
            # C. NEW CHECK EXITS (Scaling + Technical Mean Reversion)
            # ------------------------------------------------------------------
            is_closed = await self.check_exits(side_key, st, current_price, rsi_value, pnl_pct)
            if is_closed:
                return

            # ------------------------------------------------------------------
            # D. BREAKEVEN DEFENSE (+15% Gain)
            # ------------------------------------------------------------------
            if pnl_pct >= 0.15 and not st.is_breakeven_set:
                st.stop_loss = max(st.stop_loss, st.entry_price)
                st.is_breakeven_set = True
                logger.info(f"🛡️ BREAKEVEN ACTIVATED ({side_key}): Stop moved to ${st.stop_loss:.3f}")

# ------------------------------------------------------------------
            # E. BTC-BASED ATR TRAILING STOP (The "Shadow" Stop)
            # ------------------------------------------------------------------
            # We monitor BTC Price. If BTC moves against us by 2xATR, we dump the options.
            current_btc = inds.get("BTC_Price", 0.0)
            
            if st.btc_stop_price > 0 and current_btc > 0 and atr_value > 0:
                
                # --- 1. RATCHET LOGIC (Move Stop in our favor only) ---
                if side_key == "UP":
                    # We want BTC to go UP. Move Stop UP.
                    potential_new_stop = current_btc - (atr_value * self.atr_multiplier)
                    if potential_new_stop > st.btc_stop_price:
                        st.btc_stop_price = potential_new_stop
                        # logger.debug(f"⛓️ RATCHET UP: Stop moved to ${st.btc_stop_price:.2f}")

                elif side_key == "DOWN":
                    # We want BTC to go DOWN. Move Stop DOWN.
                    potential_new_stop = current_btc + (atr_value * self.atr_multiplier)
                    if potential_new_stop < st.btc_stop_price:
                        st.btc_stop_price = potential_new_stop
                        # logger.debug(f"⛓️ RATCHET DOWN: Stop moved to ${st.btc_stop_price:.2f}")

                # --- 2. TRIGGER CHECK ---
                stop_hit = False
                if side_key == "UP" and current_btc <= st.btc_stop_price:
                    stop_hit = True
                    reason = f"BTC Hit Stop (Est. ${st.btc_stop_price:.2f})"
                elif side_key == "DOWN" and current_btc >= st.btc_stop_price:
                    stop_hit = True
                    reason = f"BTC Hit Stop (Est. ${st.btc_stop_price:.2f})"

                if stop_hit:
                    logger.warning(f"🛑 SHADOW STOP TRIGGERED ({side_key}): BTC ${current_btc:.2f} crossed ${st.btc_stop_price:.2f}")
                    # EXECUTE EXIT ON THE OPTION (Sell shares at market)
                    await self.execute_order(side_key, st.token_id, best_bid, st.total_shares, False)
                    st.sell_shares(st.total_shares, best_bid)
                    return

            # ------------------------------------------------------------------
            # F. TIME CHECKPOINTS (Safety Net)
            # ------------------------------------------------------------------
            time_held = time.time() - st.entry_time
            time_held_mins = time_held / 60.0

            # --- CHECKPOINT #1: T + 4 Minutes ---
            if time_held_mins >= self.t4_checkpoint_min and not st.checked_t4:
                has_proof = (
                    price_gain >= self.t4_price_bump or
                    pnl_pct >= self.t4_min_pnl_pct
                )
                
                if has_proof:
                    logger.info(f"✅ T+4 CHECK ({side_key}): Early proof confirmed (PnL {pnl_pct*100:.1f}%). Holding.")
                else:
                    logger.warning(f"⚠️ T+4 CHECK ({side_key}): No early proof. De-risking 50%.")
                    sell_size = normalize_size(st.total_shares * 0.5)
                    if sell_size > 0:
                        await self.execute_order(side_key, st.token_id, best_bid, sell_size, False)
                        st.sell_shares(sell_size, best_bid)
                
                st.checked_t4 = True

            # --- CHECKPOINT #2: T + 6 Minutes ---
            if time_held_mins >= self.t6_exit_min and not st.checked_t6:
                is_working = (
                    price_gain >= self.t6_price_bump or
                    pnl_pct >= self.t6_min_pnl_pct
                )

                if is_working:
                    logger.info(f"🏃 T+6 CHECK ({side_key}): Trade working. Holding runner to expiry.")
                else:
                    logger.warning(f"🛑 T+6 CHECK ({side_key}): Trade failing. Selling remaining.")
                    await self.execute_order(side_key, st.token_id, best_bid, st.total_shares, False)
                    st.sell_shares(st.total_shares, best_bid)
                
                st.checked_t6 = True

    def show_final_summary(self, final_indicators=None):
        """Show final summary when market closes and settle open positions."""
        logger.info("\n" + "=" * 70)
        logger.info("🏁 MARKET CLOSED - FINAL SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Market: {self.market_slug}")
        
        # --- 1. SETTLEMENT LOGIC ---
        if self.strike_price > 0 and self.latest_btc_price > 0:
            if self.latest_btc_price >= self.strike_price:
                payouts = {"UP": 1.0, "DOWN": 0.0}
                winner_desc = "UP (BTC >= Strike)"
            else:
                payouts = {"UP": 0.0, "DOWN": 1.0}
                winner_desc = "DOWN (BTC < Strike)"
            
            logger.info(f"⚖️ SETTLEMENT: BTC ${self.latest_btc_price:,.2f} vs Strike ${self.strike_price:,.2f}")
            logger.info(f"🏆 WINNER: {winner_desc}")
            
            for side in ["UP", "DOWN"]:
                st = self.strategies[side]
                if st.total_shares > 0:
                    payout_price = payouts[side]
                    revenue = st.total_shares * payout_price
                    cost_basis = st.total_shares * st.avg_price
                    pnl = revenue - cost_basis
                    
                    if self.settings.dry_run:
                        self.sim_balance += revenue
                    
                    st.realized_pnl += pnl
                    
                    curr_bal = self.sim_balance if self.settings.dry_run else (self.cached_balance or 0.0)
                    action = "SETTLE_WIN" if payout_price == 1.0 else "SETTLE_LOSS"
                    
                    self.log_transaction(
                        side, action, payout_price, st.total_shares, 
                        revenue, pnl, 0, curr_bal, 
                        indicators=final_indicators
                    )
                    
                    logger.info(f"   🔓 Settled {side}: {st.total_shares:.2f} shares @ ${payout_price:.2f} | PnL: ${pnl:.2f}")
                    st.total_shares = 0.0
        else:
            logger.warning("⚠️ Skipping settlement: Missing Strike or BTC Price.")

        # --- 2. PRINT & LOG TOTALS ---
        up_pnl = self.strategies["UP"].realized_pnl
        down_pnl = self.strategies["DOWN"].realized_pnl
        total_pnl = up_pnl + down_pnl
        
        logger.info("-" * 70)
        logger.info(f"Realized PnL (UP):   ${up_pnl:.2f}")
        logger.info(f"Realized PnL (DOWN): ${down_pnl:.2f}")
        logger.info("-" * 70)
        logger.info(f"TOTAL PnL:           ${total_pnl:.2f}")
        
        if self.settings.dry_run:
            s_bal = self.sim_start_balance
            e_bal = self.sim_balance
            logger.info("-" * 70)
            logger.info(f"Sim Start:           ${s_bal:.2f}")
            logger.info(f"Sim End:             ${e_bal:.2f}")
        else:
            s_bal = 0.0
            e_bal = self.cached_balance if self.cached_balance else 0.0

        logger.info("=" * 70)

        self.log_transaction("N/A", "MARKET_CLOSE", 0, 0, 0, total_pnl, s_bal, e_bal, indicators=final_indicators)

    # --- Helper to parse WSS book objects ---
    def _book_from_state(self, bid_levels, ask_levels) -> dict:
        best_bid = max((float(p) for p, _ in bid_levels), default=None) if bid_levels else None
        best_ask = min((float(p) for p, _ in ask_levels), default=None) if ask_levels else None
        return {"best_bid": best_bid, "best_ask": best_ask}
    

    async def resolve_strike_price(self):
        """Fetch EXACT OPEN price from Coinbase in the background."""
        if self.is_searching: return
        self.is_searching = True
        
        if self.strike_price == 0.0:
            logger.debug(f"🕵️ Background Search: Fetching Strike Price for {self.market_slug}...")

        try:
            parts = self.market_slug.split("-")
            if len(parts) < 4: 
                self.is_searching = False
                return
            
            target_ts = int(parts[-1])
            
            dt_start = datetime.fromtimestamp(target_ts, timezone.utc).isoformat()
            dt_end = datetime.fromtimestamp(target_ts + 60, timezone.utc).isoformat()
            url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
            params = {"start": dt_start, "end": dt_end, "granularity": 60}
            headers = {"User-Agent": "Mozilla/5.0"}

            async with httpx.AsyncClient() as client:
                try:
                    resp = await client.get(url, params=params, headers=headers, timeout=5)
                    if resp.status_code == 200:
                        data = resp.json()
                        for candle in data:
                            if candle[0] == target_ts:
                                self.strike_price = float(candle[3]) # Index 3 is OPEN
                                logger.info(f"✅ STRIKE PRICE FOUND: ${self.strike_price:,.2f}")
                                self.is_searching = False
                                return
                        logger.debug("⏳ Coinbase data not ready yet.")
                    else:
                        logger.warning(f"❌ Coinbase Error: {resp.status_code}")
                except Exception as e:
                    logger.warning(f"❌ Connection Failed: {e}")

        except Exception as e:
            logger.error(f"❌ Error logic: {e}")
        
        self.is_searching = False

    async def heartbeat_loop(self):
        """Runs periodic tasks (Logging, Indicators, Console Ticker) independently."""
        while True:
            try:
                now = time.time()

                if self.get_time_remaining() == "CLOSED":
                    break

                if self.strike_price == 0.0 and not self.is_searching:
                    asyncio.create_task(self.resolve_strike_price())

                asyncio.create_task(self.get_btc_price_fast())

                if (now - self.last_csv_log) > 10:
                    try:
                        inds = await self.get_indicators(use_shared_client=True)
                        inds["Strike_Price"] = self.strike_price
                        inds["UP_Bid"] = self.latest_up_bid
                        inds["UP_Ask"] = self.latest_up_ask
                        inds["DOWN_Bid"] = self.latest_down_bid
                        inds["DOWN_Ask"] = self.latest_down_ask

                        if self.latest_btc_price > 0:
                             inds["BTC_Price"] = self.latest_btc_price

                        curr_bal = self.sim_balance if self.settings.dry_run else (self.cached_balance or 0.0)
                        self.log_transaction("N/A", "STATUS", 0, 0, 0, 0, curr_bal, curr_bal, indicators=inds)
                        self.last_csv_log = now
                    except Exception as e:
                        logger.warning(f"⚠️ Status log failed: {e}")

                if (now - self._last_log_ts) >= 1.0:
                    self._last_log_ts = now
                    ub, ua = self.latest_up_bid, self.latest_up_ask
                    db, da = self.latest_down_bid, self.latest_down_ask
                    
                    btc_str = f"BTC: ${self.latest_btc_price:,.2f}"
                    strike_str = f" | 🎯 Strike: ${self.strike_price:,.2f}" if self.strike_price > 0 else " | 🎯 Strike: ⏳"
                    
                    holding_str = ""
                    u_shares = self.strategies["UP"].total_shares
                    d_shares = self.strategies["DOWN"].total_shares
                    if u_shares > 0 or d_shares > 0:
                        parts = []
                        if u_shares > 0: parts.append(f"UP:{u_shares:.1f}")
                        if d_shares > 0: parts.append(f"DN:{d_shares:.1f}")
                        holding_str = f" | 💼 {' '.join(parts)}"

                    logger.info(f"⏱️ {self.get_time_remaining()} | ₿ {btc_str}{strike_str} | 🏷️ UP: ${ub:.3f}/{ua:.3f} | DOWN: ${db:.3f}/{da:.3f}{holding_str}")

                await asyncio.sleep(1.0) 

            except Exception as e:
                logger.error(f"Heartbeat Error: {e}")
                await asyncio.sleep(1)

    async def get_btc_price_fast(self):
        """Fast fetch BTC price with redundancy."""
        try:
            url = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
            resp = await self.shared_client.get(url, timeout=1.0)
            if resp.status_code == 200:
                self.latest_btc_price = float(resp.json()['price'])
                return
        except Exception:
            pass 

        try:
            url = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
            resp = await self.shared_client.get(url, timeout=1.0)
            if resp.status_code == 200:
                self.latest_btc_price = float(resp.json()['data']['amount'])
                return
        except Exception:
            pass

        try:
            url = "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"
            resp = await self.shared_client.get(url, timeout=1.0)
            if resp.status_code == 200:
                data = resp.json()
                if 'result' in data and 'XXBTZUSD' in data['result']:
                    self.latest_btc_price = float(data['result']['XXBTZUSD']['c'][0])
        except Exception:
            pass

    async def monitor_wss(self):
        """Monitor using Polymarket CLOB Market WebSocket."""
        if self.strike_price == 0.0:
             asyncio.create_task(self.resolve_strike_price())

        while True:
            if self.get_time_remaining() == "CLOSED":
                if not self.summary_shown:
                    logger.info("\n🚨 Market has closed.")
                    try:
                        final_inds = await self.get_indicators()
                        self.show_final_summary(final_indicators=final_inds)
                    except Exception as e:
                        logger.error(f"Error showing summary: {e}")
                    self.summary_shown = True 
                
                logger.info("🔄 Searching for next BTC 15min market...")
                try:
                    new_market = find_current_btc_15min_market()
                    if new_market != self.market_slug:
                        await self.close()
                        self.__init__(self.settings)
                        self.summary_shown = False
                        
                        # New Market Setup (keep your code)
                        asyncio.create_task(self.resolve_strike_price())
                        try:
                            inds = await self.get_indicators()
                            inds["Strike_Price"] = self.strike_price
                            self.log_transaction("N/A", "MARKET_START", 0, 0, 0, 0, self.sim_balance, self.sim_balance, indicators=inds)
                        except Exception as e:
                            logger.error(f"Failed to log market start: {e}")
                        
                        logger.info(f"✅ Switching to: {new_market}")
                        continue  # Exit retry, proceed to WSS/loop
                    else:
                        continue  # Same market, no switch needed
                except RuntimeError as e:
                    if "retry" in str(e).lower():
                        logger.info("No active market yet. Waiting 10s...")
                        await asyncio.sleep(10)
                    else:
                        raise
                except Exception as e:
                    logger.error(f"Search error: {e}")
                    await asyncio.sleep(10)

            logger.info("=" * 70)
            logger.info("🚀 CONTRARIAN SHORT BOT STARTED (PARALLEL MODE)")
            if self.is_first_market:
                logger.warning("⚠️ FIRST MARKET DETECTED - TRADING DISABLED FOR THIS SESSION")
            logger.info("=" * 70)
            logger.info(f"Market: {self.market_slug}")
            
            heartbeat_task = asyncio.create_task(self.heartbeat_loop())
            
            client = MarketWssClient(
                ws_base_url=self.settings.ws_url,
                asset_ids=[self.yes_token_id, self.no_token_id],
            )

            last_eval = 0.0
            
            try:
                async for asset_id, event_type in client.run():
                    if self.get_time_remaining() == "CLOSED":
                        heartbeat_task.cancel()
                        break 

                    now = time.time()
                    if (now - last_eval) < 0.1: 
                        continue
                    last_eval = now

                    yes_state = client.get_book(self.yes_token_id)
                    no_state = client.get_book(self.no_token_id)
                    
                    if not yes_state or not no_state: continue

                    yes_bids, yes_asks = yes_state.to_levels()
                    no_bids, no_asks = no_state.to_levels()
                    
                    up_book = self._book_from_state(yes_bids, yes_asks)
                    down_book = self._book_from_state(no_bids, no_asks)
                    
                    # Update shared prices
                    up_ask = up_book.get("best_ask")
                    up_bid = up_book.get("best_bid")
                    if up_ask is None and up_bid and up_bid > 0.90: self.latest_up_ask = 1.00
                    else: self.latest_up_ask = up_ask or 0.0
                    self.latest_up_bid = up_bid or 0.0

                    down_ask = down_book.get("best_ask")
                    down_bid = down_book.get("best_bid")
                    if down_ask is None and down_bid and down_bid > 0.90: self.latest_down_ask = 1.00
                    else: self.latest_down_ask = down_ask or 0.0
                    self.latest_down_bid = down_bid or 0.0

                    await self.process_strategy("UP", up_book)
                    await self.process_strategy("DOWN", down_book)

            except Exception as e:
                logger.warning(f"WSS monitor error: {e}. Reconnecting...")
                heartbeat_task.cancel()
                await asyncio.sleep(2)
                continue
            finally:
                heartbeat_task.cancel()

async def main():
    shutdown = GracefulShutdown()
    bot = None
    try:
        settings = load_settings()
        setup_logging(verbose=settings.verbose, use_rich=settings.use_rich_output)
        
        if not ConfigValidator.validate_and_print(settings):
            print_error("Configuration invalid.")
            return
        
        print_header("🚀 BTC 15-Minute Contrarian Short Bot")
        # --- RETRY LOOP: Keep trying until a market is found ---
        while True:
            try:
                bot = ContrarianShortBot(settings)
                break # Success! Market found.
            except RuntimeError as e:
                # If the error is about finding a market, wait and retry
                if "Could not find market" in str(e) or "No active BTC" in str(e):
                    logger.warning("⏳ No active market found. Retrying in 10s...")
                    await asyncio.sleep(10)
                else:
                    raise e # Real error, let it crash
        
        def on_shutdown():
            if bot: bot.show_final_summary()
        shutdown.register_callback(on_shutdown)
        
        await bot.monitor_wss()
        
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.exception(f"Fatal Error: {e}")
    finally:
        if bot:
            logger.info("Shutting down bot resources...")
            await bot.close()

if __name__ == "__main__":
    asyncio.run(main())