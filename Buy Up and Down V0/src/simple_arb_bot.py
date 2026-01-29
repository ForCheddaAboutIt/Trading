"""
Average Down Bot for Bitcoin 15min markets.

Strategy: 
1. Monitor UP and DOWN tokens independently.
2. Enter if price <= ENTRY_PRICE.
3. If price drops by INTERVAL, buy more (size * multiplier) to average down.
4. Exit: Sell 50% at Profit Target, then Trail 5%.
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
    get_positions,
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
            raise RuntimeError("No active BTC 15min market found")
        
        now_ts = int(datetime.now().timestamp())
        all_ts = sorted((int(ts) for ts in matches), reverse=True)
        open_ts = [ts for ts in all_ts if now_ts < (ts + 900)]
        chosen_ts = open_ts[0] if open_ts else all_ts[0]
        slug = f"btc-updown-15m-{chosen_ts}"
        
        logger.info(f"✅ Market found: {slug}")
        return slug
        
    except Exception as e:
        logger.error(f"Error searching for BTC 15min market: {e}")
        logger.warning("Using default market from configuration...")
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
        self.side_name = side_name  # "UP" or "DOWN"
        self.total_shares = 0.0
        self.total_cost = 0.0
        self.avg_price = 0.0
        self.last_buy_price = None
        self.last_buy_size = 0.0
        self.is_trailing = False
        self.trailing_high_water_mark = 0.0
        self.realized_pnl = 0.0
        self.stop_loss_triggered = False

    def add_buy(self, price, size):
        cost = price * size
        self.total_cost += cost
        self.total_shares += size
        self.avg_price = self.total_cost / self.total_shares if self.total_shares > 0 else 0.0
        self.last_buy_price = price
        self.last_buy_size = size
        self.is_trailing = False # Reset trailing on new buy

    def sell_shares(self, size, price):
        if size > self.total_shares:
            size = self.total_shares
        
        revenue = size * price
        cost_basis = size * self.avg_price
        pnl = revenue - cost_basis
        
        self.total_shares -= size
        self.total_cost -= cost_basis
        self.realized_pnl += pnl
        
        if self.total_shares < 0.1: # Practically empty
            self.reset()
            
    def reset(self):
        self.total_shares = 0.0
        self.total_cost = 0.0
        self.avg_price = 0.0
        self.last_buy_price = None
        self.last_buy_size = 0.0
        self.is_trailing = False
        self.trailing_high_water_mark = 0.0
        self.stop_loss_triggered = False

# --- Main Bot Class ---
class AverageDownBot:
    """Bot implementing the Average Down Strategy."""
    
    def __init__(self, settings):
        self.settings = settings
        self.client = get_client(settings)
        
        # --- Strategy Parameters ---
        self.entry_price = float(settings.entry_price)
        self.min_price_thresh = float(settings.min_price_threshold)
        self.initial_size = float(settings.initial_order_size)
        self.size_multiplier = float(settings.size_multiplier)
        self.drop_interval = float(settings.price_drop_interval)
        self.profit_target = float(settings.profit_target)
        self.sell_pct = float(settings.sell_percentage)
        self.trailing_pct = float(settings.trailing_stop_percent)
        self.stop_buy_mins = float(settings.stop_buying_time_minutes)

        # --- Statistics & Risk ---
        self.stats_tracker = None
        if settings.enable_stats:
            try:
                self.stats_tracker = StatisticsTracker(log_file=settings.trade_log_file)
            except Exception as e:
                logger.warning(f"Failed to initialize statistics tracker: {e}")
        
        self.risk_manager = None
        if settings.max_daily_loss > 0: # Simple check for risk usage
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

        # --- State Tracking ---
        self.strategies: Dict[str, PositionState] = {
            "UP": PositionState(self.yes_token_id, "UP"),
            "DOWN": PositionState(self.no_token_id, "DOWN")
        }
        
        self.cached_balance = None
        self.sim_balance = settings.sim_balance if settings.sim_balance > 0 else 100.0
        self.sim_start_balance = self.sim_balance
        self.trades_executed = 0
        self._last_log_ts = 0.0

        self.transaction_file = "transactions.csv"
        self.btc_rsi = 50.0
        self.last_rsi_update = 0.0
        self.last_csv_log = 0.0
        self.summary_shown = False
        self.strike_price = 0.0
        self.is_searching = False
        self.latest_up_bid = 0.0
        self.latest_up_ask = 0.0
        self.latest_down_bid = 0.0
        self.latest_down_ask = 0.0
        self.latest_btc_price = 0.0
        self.last_valid_indicators = None
        self.indicator_lock = asyncio.Lock() # Prevent race conditions
        self.last_indicator_update = 0.0     # For throttling
        self.shared_client = httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) # Persistent connection
        self.price_lock = False # Prevents request flooding

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
        """Append transaction details + indicators to CSV."""
        file_exists = os.path.isfile(self.transaction_file)
        if indicators is None: indicators = {}
        
        try:
            with open(self.transaction_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                
                ind_keys = [
                    "UP_Bid", "UP_Ask", "DOWN_Bid", "DOWN_Ask",
                    "BTC_Price",
                    "Strike_Price",
                    "RSI", "BB_Upper", "BB_Lower", "BB_Middle", "BB_Width", "BB_PctB", 
                    "ATR", "Stoch_K", "VWAP", "MACD", "CCI", "MFI", "ROC", "Pivot"
                ]
                
                # Write Header if new file
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

                # === ADD WHITESPACE ON CLOSE ===
                if action == "MARKET_CLOSE":
                    writer.writerow([]) # Empty row 1
                    writer.writerow([]) # Empty row 2
                # ===============================

        except Exception as e:
            logger.error(f"Failed to write log: {e}")

    def log_session_summary(self, start_bal, end_bal, total_pnl):
        """Log the final session summary and a blank separator row."""
        try:
            with open(self.transaction_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Write the Summary Row
                writer.writerow([
                    timestamp, self.market_slug, "MARKET_CLOSE", "SUMMARY",
                    "", "", "", f"{total_pnl:.2f}", 
                    f"{start_bal:.2f}", f"{end_bal:.2f}"
                ])
                
                # Write an empty row for spacing between markets
                writer.writerow([]) 
                
        except Exception as e:
            logger.error(f"Failed to write session summary: {e}")

    async def get_indicators(self, use_shared_client=True, force_update=False) -> dict:
        """
        Fetch data and calculate indicators.
        - Internal Throttling (Max 1 request per 5s).
        - Thread-safe & Cached.
        """
        keys = ["RSI", "BB_Upper", "BB_Lower", "BB_Middle", "BB_Width", "BB_PctB", 
                "ATR", "Stoch_K", "VWAP", "MACD", "CCI", "MFI", "ROC", "Pivot"]
        defaults = {k: 0.0 for k in keys}

        async with self.indicator_lock:
            now = time.time()
            
            # === INTERNAL THROTTLE ===
            # If we fetched less than 5 seconds ago, return CACHE immediately.
            # This protects Binance from being hammered by order executions.
            if not force_update and self.last_valid_indicators and (now - self.last_indicator_update < 5.0):
                return self.last_valid_indicators
            # =========================

            try:
                endpoints = ["https://api.binance.com", "https://api.binance.us"]
                resp_1m = None
                resp_1d = None
                
                client_to_use = self.shared_client if use_shared_client else httpx.AsyncClient()
                
                if use_shared_client:
                     for base_url in endpoints:
                        try:
                            # 2.0s timeout is plenty for a persistent connection
                            r1 = await client_to_use.get(f"{base_url}/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=120", timeout=2.0)
                            r2 = await client_to_use.get(f"{base_url}/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=2", timeout=2.0)
                            if r1.status_code == 200:
                                resp_1m = r1
                                resp_1d = r2
                                break
                        except Exception:
                            continue
                else:
                    async with client_to_use as client:
                        for base_url in endpoints:
                            try:
                                r1 = await client.get(f"{base_url}/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=120", timeout=2.0)
                                r2 = await client.get(f"{base_url}/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=2", timeout=2.0)
                                if r1.status_code == 200:
                                    resp_1m = r1
                                    resp_1d = r2
                                    break
                            except Exception:
                                continue

                if not resp_1m or resp_1m.status_code != 200:
                    if self.last_valid_indicators:
                        return self.last_valid_indicators
                    return defaults

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
                
                # --- CALCULATIONS (Wilder's Smoothing) ---
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
                cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

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
        
    async def execute_order(self, side_key, token_id, price, size, is_buy: bool):
        cost = price * size
        action = "BUY" if is_buy else "SELL"
        pnl = 0.0  # <--- Initialize PnL
        
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

        # --- PRE-CALC PnL for Logs ---
        if not is_buy:
            st = self.strategies[side_key]
            pnl = (price - st.avg_price) * size

        # --- LOGGING ---
        logger.info("=" * 70)
        logger.info(f"⚡ EXECUTING {action} ({side_key})")
        logger.info("-" * 70)
        logger.info(f"Price:            ${price:.4f}")
        logger.info(f"Size:             {size:.2f} shares")
        logger.info(f"Total Value:      ${cost:.2f}")
        
        st = self.strategies[side_key]
        if not is_buy:
             logger.info(f"Avg Cost Basis:   ${st.avg_price:.4f}")
             logger.info(f"Est. PnL:         ${pnl:.2f}")
        else:
             # Calculate Projected New Average & Target
             new_shares = st.total_shares + size
             new_cost = st.total_cost + cost
             if new_shares > 0:
                 new_avg = new_cost / new_shares
                 raw_target = new_avg + self.profit_target
                 
                 # Round UP to nearest 1 cent (0.01)
                 # Example: 0.371 -> 0.38
                 target_price = math.ceil(raw_target * 100) / 100
                 
                 logger.info(f"New Sell Target:  ${target_price:.4f}")

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
            
            # --- ADD THIS: Log to CSV ---
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
                
                # Recalculate real metrics
                real_cost = fill_price * size
                real_pnl = 0.0
                if not is_buy:
                    st = self.strategies[side_key]
                    real_pnl = (fill_price - st.avg_price) * size
                
                # --- ADD THIS: Log to CSV ---
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
        
    async def process_strategy(self, side_key: str, book: dict):
        """Core Average Down Strategy Logic."""
        st = self.strategies[side_key]
        minutes_remaining = self.get_time_remaining_minutes()
        
        # Extract Best Prices
        # We BUY at the Ask (best_ask) and SELL at the Bid (best_bid)
        best_ask = book.get("best_ask")
        best_bid = book.get("best_bid")
        
        if best_ask is None or best_bid is None:
            return

        # --- 1. ENTRY LOGIC (BUYING) ---
        can_buy_time = minutes_remaining > self.stop_buy_mins
        can_buy_price_floor = best_ask >= self.min_price_thresh
        
        buy_signal = False
        buy_size = 0.0
        reason = ""

        if can_buy_time and can_buy_price_floor:
            # Case A: Initial Entry
# Case A: Initial Entry
            if st.total_shares == 0:
                if best_ask <= self.entry_price:
                    raw_size = self.initial_size
                    buy_size = normalize_size(raw_size, self.min_order_size)
                    
                    if buy_size > 0:
                        buy_signal = True
                        reason = f"Initial Entry (${best_ask:.3f} <= ${self.entry_price})"
                    else:
                        logger.warning(f"⚠️ {side_key} Initial entry size {raw_size} < min {self.min_order_size}")

            # Case B: Average Down
            else:
                if st.last_buy_price and best_ask <= (st.last_buy_price - self.drop_interval):
                    raw_size = st.last_buy_size * self.size_multiplier
                    buy_size = normalize_size(raw_size, self.min_order_size)
                    
                    if buy_size > 0:
                        buy_signal = True
                        reason = f"Avg Down (Dropped {self.drop_interval} from {st.last_buy_price:.3f})"
                    else:
                        logger.warning(f"⚠️ {side_key} Avg down size {raw_size} < min {self.min_order_size}")

        if buy_signal:
            logger.info(f"🔵 {side_key} BUY SIGNAL: {reason}")
            success = await self.execute_order(side_key, st.token_id, best_ask, buy_size, True)
            if success:
                st.add_buy(best_ask, buy_size)

        # --- 2. EXIT LOGIC (SELLING) ---
        if st.total_shares > 0:
            # --- NEW: 10 Cent Panic Stop (Sell 50%) ---
            # If price hits $0.10 and we haven't panic-sold yet:
            if best_bid <= 0.10 and not st.stop_loss_triggered:
                logger.info(f"🔻 {side_key} HIT 10c PANIC STOP (${best_bid:.3f} <= $0.10)")
                
                # Calculate 50% of current holding
                sell_amt = normalize_size(st.total_shares, self.min_order_size)
                
                # Execute Sell
                success = await self.execute_order(side_key, st.token_id, best_bid, sell_amt, False)
                if success:
                    st.sell_shares(sell_amt, best_bid)
                    st.stop_loss_triggered = True
                    logger.info(f"   ✂️ Sold {sell_amt:.2f} shares (50%) to limit losses.")
            target_price = st.avg_price + self.profit_target
            
            # Sub-Strategy: Profit Target -> Trailing Stop
            if not st.is_trailing:
                if best_bid >= target_price:
                    logger.info(f"🟢 {side_key} HIT PROFIT TARGET (${best_bid:.3f} >= ${target_price:.3f})")
                    sell_amt = st.total_shares * self.sell_pct
                    success = await self.execute_order(side_key, st.token_id, best_bid, sell_amt, False)
                    if success:
                        st.sell_shares(sell_amt, best_bid)
                        st.is_trailing = True
                        st.trailing_high_water_mark = best_bid
                        logger.info(f"   ⚓ Trailing Stop Activated at ${best_bid:.3f}")

            # Sub-Strategy: Managing Trailing Stop
            else:
                if best_bid > st.trailing_high_water_mark:
                    st.trailing_high_water_mark = best_bid
                
                # stop_price = st.trailing_high_water_mark * (1 - self.trailing_pct)
                hard_stop = st.avg_price # Break even hard stop
                
                should_sell = False
                exit_reason = ""
                
                # if best_bid <= stop_price:
                #     should_sell = True
                #     exit_reason = f"Trailing Stop Hit (${best_bid:.3f} <= ${stop_price:.3f})"
                if best_bid <= hard_stop:
                    should_sell = True
                    exit_reason = f"Hard Stop Hit (${best_bid:.3f} <= ${hard_stop:.3f})"
                
                if should_sell:
                    logger.info(f"🔴 {side_key} EXIT SIGNAL: {exit_reason}")
                    success = await self.execute_order(side_key, st.token_id, best_bid, st.total_shares, False)
                    if success:
                        st.sell_shares(st.total_shares, best_bid)

    def show_final_summary(self, final_indicators=None):
        """Show final summary when market closes and settle open positions."""
        logger.info("\n" + "=" * 70)
        logger.info("🏁 MARKET CLOSED - FINAL SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Market: {self.market_slug}")
        
        # --- 1. SETTLEMENT LOGIC (Fixing the PnL Issue) ---
        if self.strike_price > 0 and self.latest_btc_price > 0:
            # Determine Winner: UP Wins if BTC >= Strike
            if self.latest_btc_price >= self.strike_price:
                payouts = {"UP": 1.0, "DOWN": 0.0}
                winner_desc = "UP (BTC >= Strike)"
            else:
                payouts = {"UP": 0.0, "DOWN": 1.0}
                winner_desc = "DOWN (BTC < Strike)"
            
            logger.info(f"⚖️ SETTLEMENT: BTC ${self.latest_btc_price:,.2f} vs Strike ${self.strike_price:,.2f}")
            logger.info(f"🏆 WINNER: {winner_desc}")
            
            # Process Settlements for both sides
            for side in ["UP", "DOWN"]:
                st = self.strategies[side]
                if st.total_shares > 0:
                    payout_price = payouts[side]
                    
                    # Calculate Final PnL for held shares
                    revenue = st.total_shares * payout_price
                    cost_basis = st.total_shares * st.avg_price
                    pnl = revenue - cost_basis
                    
                    # Update Balance (Simulated)
                    if self.settings.dry_run:
                        self.sim_balance += revenue
                    
                    # Update Strategy PnL State
                    st.realized_pnl += pnl
                    
                    # Log to CSV as a transaction
                    curr_bal = self.sim_balance if self.settings.dry_run else (self.cached_balance or 0.0)
                    action = "SETTLE_WIN" if payout_price == 1.0 else "SETTLE_LOSS"
                    
                    self.log_transaction(
                        side, action, payout_price, st.total_shares, 
                        revenue, pnl, 0, curr_bal, 
                        indicators=final_indicators
                    )
                    
                    logger.info(f"   🔓 Settled {side}: {st.total_shares:.2f} shares @ ${payout_price:.2f} | PnL: ${pnl:.2f}")
                    
                    # Clear position
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

        # Write final summary row to CSV
        self.log_transaction("N/A", "MARKET_CLOSE", 0, 0, 0, total_pnl, s_bal, e_bal, indicators=final_indicators)

    # --- Helper to parse WSS book objects ---
    def _book_from_state(self, bid_levels, ask_levels) -> dict:
        best_bid = max((float(p) for p, _ in bid_levels), default=None) if bid_levels else None
        best_ask = min((float(p) for p, _ in ask_levels), default=None) if ask_levels else None
        return {"best_bid": best_bid, "best_ask": best_ask}
    

    async def resolve_strike_price(self):
        """
        Fetch EXACT OPEN price from Coinbase in the background.
        """
        # Guard: If already searching, don't start another one.
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
            
            # Coinbase API setup
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
                                self.is_searching = False # Done!
                                return
                        
                        logger.debug("⏳ Coinbase data not ready yet.")
                    else:
                        logger.warning(f"❌ Coinbase Error: {resp.status_code}")
                except Exception as e:
                    logger.warning(f"❌ Connection Failed: {e}")

        except Exception as e:
            logger.error(f"❌ Error logic: {e}")
        
        # Always reset flag at the end so we can try again later
        self.is_searching = False

    async def heartbeat_loop(self):
        """Runs periodic tasks (Logging, Indicators, Console Ticker) independently."""
        while True:
            try:
                now = time.time()

                if self.get_time_remaining() == "CLOSED":
                    break

                # Background Retry for Strike Price
                if self.strike_price == 0.0 and not self.is_searching:
                    asyncio.create_task(self.resolve_strike_price())

                # 1. FAST BTC PRICE FETCH (Runs every second)
                asyncio.create_task(self.get_btc_price_fast())

                # 2. CSV LOGGING (Every 10s)
                if (now - self.last_csv_log) > 10:
                    try:
                        # === FIX: CALL THE ENGINE ===
                        # Don't read a static variable. Call the function.
                        # This runs the math INSTANTLY using self.latest_btc_price
                        # It also handles fetching new candles if the cache is >15s old.
                        inds = await self.get_indicators(use_shared_client=True)
                        
                        # Add Strike Price metadata
                        # Add Strike Price metadata
                        inds["Strike_Price"] = self.strike_price
                        
                        # --- NEW: Add Market Prices to CSV ---
                        inds["UP_Bid"] = self.latest_up_bid
                        inds["UP_Ask"] = self.latest_up_ask
                        inds["DOWN_Bid"] = self.latest_down_bid
                        inds["DOWN_Ask"] = self.latest_down_ask
                        # -------------------------------------

                        # (Optional) Ensure the BTC_Price column matches exactly what we see
                        if self.latest_btc_price > 0:
                             inds["BTC_Price"] = self.latest_btc_price

                        curr_bal = self.sim_balance if self.settings.dry_run else (self.cached_balance or 0.0)
                        self.log_transaction("N/A", "STATUS", 0, 0, 0, 0, curr_bal, curr_bal, indicators=inds)
                        self.last_csv_log = now
                    except Exception as e:
                        logger.warning(f"⚠️ Status log failed: {e}")

                # 3. Console Ticker (Every 1s)
                if (now - self._last_log_ts) >= 1.0:
                    self._last_log_ts = now
                    
                    # Grab local copies
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

                    # Display as "Bid/Ask"
                    logger.info(f"⏱️ {self.get_time_remaining()} | ₿ {btc_str}{strike_str} | 🏷️ UP: ${ub:.3f}/{ua:.3f} | DOWN: ${db:.3f}/{da:.3f}{holding_str}")

                await asyncio.sleep(1.0) # Wait 1s

            except Exception as e:
                logger.error(f"Heartbeat Error: {e}")
                await asyncio.sleep(1)

    async def get_btc_price_fast(self):
        """
        Fast fetch BTC price with 3 layers of redundancy.
        1. Coinbase Exchange (Pro)
        2. Coinbase Spot (Standard)
        3. Kraken (Backup)
        """
        # Layer 1: Coinbase Exchange (Best)
        try:
            url = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
            resp = await self.shared_client.get(url, timeout=1.0)
            if resp.status_code == 200:
                self.latest_btc_price = float(resp.json()['price'])
                return
        except Exception:
            pass 

        # Layer 2: Coinbase Spot (v2) - If Layer 1 fails
        try:
            url = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
            resp = await self.shared_client.get(url, timeout=1.0)
            if resp.status_code == 200:
                self.latest_btc_price = float(resp.json()['data']['amount'])
                return
        except Exception:
            pass

        # Layer 3: Kraken (Safety Net) - If Coinbase is fully down
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
        # 0. Initial Kickoff
        if self.strike_price == 0.0:
             asyncio.create_task(self.resolve_strike_price())

        while True:
            # 1. Market Rollover Check
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
                        self.__init__(self.settings)
                        self.summary_shown = False
                        
                        # New Market Setup
                        asyncio.create_task(self.resolve_strike_price())
                        
                        try:
                            inds = await self.get_indicators()
                            inds["Strike_Price"] = self.strike_price
                            self.log_transaction("N/A", "MARKET_START", 0, 0, 0, 0, self.sim_balance, self.sim_balance, indicators=inds)
                        except Exception as e:
                            logger.error(f"Failed to log market start: {e}")
                        
                        logger.info(f"✅ Switching to: {new_market}")
                        continue 

                    await asyncio.sleep(10)
                    continue

                except Exception as e:
                    logger.error(f"Search error: {e}")
                    await asyncio.sleep(10)
                    continue

            # 2. Start WSS & Heartbeat
            logger.info("=" * 70)
            logger.info("🚀 AVERAGE DOWN BOT STARTED (PARALLEL MODE)")
            logger.info("=" * 70)
            logger.info(f"Market: {self.market_slug}")
            
            # === LAUNCH THE HEARTBEAT (Track A) ===
            heartbeat_task = asyncio.create_task(self.heartbeat_loop())
            # ======================================
            
            client = MarketWssClient(
                ws_base_url=self.settings.ws_url,
                asset_ids=[self.yes_token_id, self.no_token_id],
            )

            last_eval = 0.0
            
            try:
                # 3. Listen for Trades (Track B)
                async for asset_id, event_type in client.run():
                    if self.get_time_remaining() == "CLOSED":
                        heartbeat_task.cancel() # Stop the heartbeat
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
                    
                    # === UPDATE SHARED PRICES ===
                    # This allows the Heartbeat loop to see the prices!
                    # 1. UP SIDE
                    up_ask = up_book.get("best_ask")
                    up_bid = up_book.get("best_bid")
                    
                    # If Ask is missing but Bid is high (>0.90), visually default Ask to 1.00
                    if up_ask is None and up_bid and up_bid > 0.90:
                        self.latest_up_ask = 1.00
                    else:
                        self.latest_up_ask = up_ask or 0.0
                    self.latest_up_bid = up_bid or 0.0

                    # 2. DOWN SIDE
                    down_ask = down_book.get("best_ask")
                    down_bid = down_book.get("best_bid")
                    
                    if down_ask is None and down_bid and down_bid > 0.90:
                        self.latest_down_ask = 1.00
                    else:
                        self.latest_down_ask = down_ask or 0.0
                    self.latest_down_bid = down_bid or 0.0
                    # ============================

                    await self.process_strategy("UP", up_book)
                    await self.process_strategy("DOWN", down_book)

            except Exception as e:
                logger.warning(f"WSS monitor error: {e}. Reconnecting...")
                heartbeat_task.cancel() # Ensure we don't duplicate tasks
                await asyncio.sleep(2)
                continue
            finally:
                heartbeat_task.cancel()

async def main():
    shutdown = GracefulShutdown()
    bot = None # Init ref
    try:
        settings = load_settings()
        setup_logging(verbose=settings.verbose, use_rich=settings.use_rich_output)
        
        if not ConfigValidator.validate_and_print(settings):
            print_error("Configuration invalid.")
            return
        
        print_header("🚀 BTC 15-Minute Average Down Bot")
        bot = AverageDownBot(settings)
        
        # Shutdown hook: Only handles the summary print
        def on_shutdown():
            if bot: bot.show_final_summary()
        shutdown.register_callback(on_shutdown)
        
        # Start Monitor
        await bot.monitor_wss()
        
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.exception(f"Fatal Error: {e}")
    finally:
        # === VITAL: CLEANUP ON EXIT ===
        if bot:
            logger.info("Shutting down bot resources...")
            await bot.close()

if __name__ == "__main__":
    asyncio.run(main())