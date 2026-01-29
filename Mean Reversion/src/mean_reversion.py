"""
Average Down Bot for Bitcoin 15min markets.

Strategy Update (Research Implemented):
1. Environment Filter: Blocks trading if Volatility (BB Width) is too low or Spread is too high.
2. Confluence Signal: Requires RSI, Bollinger Band %B, Stoch_K, and CCI alignment.
3. Dynamic Risk: Uses ATR for Stop Losses and Grid Spacing instead of fixed dollars.
4. Exit: Technical exits based on RSI reversion or hard stops.
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
    get_balance
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
        self.side_name = side_name
        self.total_shares = 0.0
        self.total_cost = 0.0
        self.avg_price = 0.0
        self.last_buy_price = None
        self.last_buy_ts = 0.0 # Added for debounce
        self.stop_loss_price = 0.0
        self.entry_atr = 0.0 # Store ATR at entry for dynamic exit
        self.realized_pnl = 0.0



    def add_buy(self, price, size, atr):
        cost = price * size
        self.total_cost += cost
        self.total_shares += size
        self.avg_price = self.total_cost / self.total_shares if self.total_shares > 0 else 0.0
        self.last_buy_price = price
        self.last_buy_ts = time.time()
        self.entry_atr = atr
        # Set initial Stop Loss at 2.5x ATR below entry (for long)
        # Note: For binary options, we simulate 'price' stops via PnL or underlying
        pass

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
        self.last_buy_ts = 0.0
        self.stop_loss_price = 0.0

# --- Main Bot Class ---
class AverageDownBot:
    """
    Updated Bot implementing the 'Confluence Engine' Strategy.
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.client = get_client(settings)
        
        # --- Load New Logic Parameters from ENV ---
        # Layer 1: Environment Filters
        self.env_min_volatility = float(os.getenv("ENV_MIN_VOLATILITY", "0.003")) # BB Width floor
        self.env_max_spread = float(os.getenv("ENV_MAX_SPREAD", "0.05")) # 5% max spread cost
        
        # Layer 2: Signal Triggers
        self.rsi_buy_up = float(os.getenv("ENV_RSI_BUY_UP", "30"))
        self.rsi_buy_down = float(os.getenv("ENV_RSI_BUY_DOWN", "70"))
        
        self.stoch_buy_up = float(os.getenv("ENV_STOCH_BUY_UP", "20"))
        self.stoch_buy_down = float(os.getenv("ENV_STOCH_BUY_DOWN", "80"))
        
        self.cci_buy_up = float(os.getenv("ENV_CCI_BUY_UP", "-100"))
        self.cci_buy_down = float(os.getenv("ENV_CCI_BUY_DOWN", "100"))

        self.mfi_abort_threshold = float(os.getenv("MFI_ABORT_THRESHOLD", "90"))
        self.mfi_buy_up_max = float(os.getenv("MFI_BUY_UP_MAX", "10"))
        self.mfi_buy_down_min = float(os.getenv("MFI_BUY_DOWN_MIN", "80"))
        self.mfi_panic_block_max = float(os.getenv("MFI_PANIC_BLOCK_MAX", "20"))
        
        # Layer 3: Risk Management
        self.atr_stop_mult = float(os.getenv("ENV_ATR_STOP_MULT", "2.5")) # Stop loss multiplier
        self.atr_grid_mult = float(os.getenv("ENV_ATR_GRID_MULT", "1.5")) # Grid spacing multiplier
        self.max_drawdown_pct = float(os.getenv("ENV_MAX_DRAWDOWN", "0.15")) # 15% Hard Stop per trade

        # Layer 4: Panic Override Settings
        self.panic_mfi = float(os.getenv("PANIC_MFI_THRESHOLD", "15"))
        self.panic_spread = float(os.getenv("PANIC_SPREAD_TOLERANCE", "0.15"))
        self.panic_bb_pct = float(os.getenv("PANIC_BB_PCT_B", "-0.1"))

        # Time Management
        self.stop_buying_minutes = int(os.getenv("STOP_BUYING_TIME_MINUTES", "5"))

        # Standard settings
        self.initial_size = float(settings.initial_order_size)
        self.profit_target = float(settings.profit_target)
        
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
        self.closing_warning_shown = False
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
                    "BTC_Price",
                    "Strike_Price",
                    "UP_Bid", "UP_Ask", "DOWN_Bid", "DOWN_Ask",
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
        Fetch data and calculate indicators. Robust version that doesn't crash if one fails.
        """
        keys = [
            "BTC_Price", "Strike_Price",
            "UP_Bid", "UP_Ask", "DOWN_Bid", "DOWN_Ask",
            "RSI", "BB_Upper", "BB_Lower", "BB_Middle", "BB_Width", "BB_PctB", 
            "ATR", "Stoch_K", "VWAP", "MACD", "CCI", "MFI", "ROC", "Pivot" 
        ]
        
        # Default all to 0.0
        result = {k: 0.0 for k in keys}

        async with self.indicator_lock:
            now = time.time()
            if not force_update and self.last_valid_indicators and (now - self.last_indicator_update < 5.0):
                return self.last_valid_indicators

            try:
                # 1. FETCH DATA
                base_url = "https://api.binance.us"
                client_to_use = self.shared_client if use_shared_client else httpx.AsyncClient()
                
                url = f"{base_url}/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=100"
                r1 = await client_to_use.get(url, timeout=3.0)
                
                if r1.status_code != 200:
                    logger.warning(f"⚠️ Binance API Error: {r1.status_code}")
                    return self.last_valid_indicators if self.last_valid_indicators else result

                data = r1.json()
                if not data or len(data) < 20:
                    logger.warning("⚠️ Not enough history data from Binance.")
                    return self.last_valid_indicators if self.last_valid_indicators else result

                # 2. PREPARE DATAFRAME
                df = pd.DataFrame(data, columns=['Time','Open','High','Low','Close','Vol','x','y','z','a','b','c'])
                cols = ['Close', 'High', 'Low', 'Vol']
                for c in cols: 
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                
                # Live Price Injection
                if self.latest_btc_price > 0:
                    df.iloc[-1, df.columns.get_loc('Close')] = self.latest_btc_price

                close = df['Close']
                high = df['High']
                low = df['Low']
                vol = df['Vol']
                
                # Helper to safely get the last value
                def get_val(series):
                    if series is None or series.empty: return 0.0
                    v = series.iloc[-1]
                    return round(float(v), 4) if not pd.isna(v) else 0.0

                # 3. CALCULATE BASIC INDICATORS
                # RSI
                delta = close.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

                # BB
                bb_middle = close.rolling(20).mean()
                std20 = close.rolling(20).std()
                bb_upper = bb_middle + (std20 * 2)
                bb_lower = bb_middle - (std20 * 2)
                bb_width = (bb_upper - bb_lower) / bb_middle
                bb_pct_b = (close - bb_lower) / (bb_upper - bb_lower)

                # ATR
                tr1 = high - low
                tr2 = (high - close.shift()).abs()
                tr3 = (low - close.shift()).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

                # Stoch & CCI
                lowest_low = low.rolling(14).min()
                highest_high = high.rolling(14).max()
                stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
                
                tp = (high + low + close) / 3
                cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
                
                # MFI & ROC
                raw_flow = tp * vol
                pos_flow = raw_flow.where(tp > tp.shift(), 0).rolling(14).sum()
                neg_flow = raw_flow.where(tp < tp.shift(), 0).rolling(14).sum()
                mfi = 100 - (100 / (1 + (pos_flow / neg_flow)))
                roc = ((close - close.shift(9)) / close.shift(9)) * 100

                # Fill Basic Results
                result.update({
                    "BTC_Price": get_val(close),
                    "RSI": get_val(rsi),
                    "BB_Upper": get_val(bb_upper),
                    "BB_Lower": get_val(bb_lower),
                    "BB_Middle": get_val(bb_middle), 
                    "BB_Width": get_val(bb_width),  
                    "BB_PctB": get_val(bb_pct_b),   
                    "ATR": get_val(atr),
                    "Stoch_K": get_val(stoch_k),
                    "CCI": get_val(cci),
                    "MFI": get_val(mfi),
                    "ROC": get_val(roc),
                })

                # 4. CALCULATE NEW INDICATORS (Safely)
                try:
                    # VWAP
                    vwap_val = (tp * vol).cumsum() / vol.cumsum()
                    result["VWAP"] = get_val(vwap_val)
                except Exception as e:
                    logger.error(f"⚠️ VWAP Calc Failed: {e}")

                try:
                    # MACD
                    ema12 = close.ewm(span=12, adjust=False).mean()
                    ema26 = close.ewm(span=26, adjust=False).mean()
                    macd_line = ema12 - ema26
                    result["MACD"] = get_val(macd_line)
                except Exception as e:
                    logger.error(f"⚠️ MACD Calc Failed: {e}")

                try:
                    # Pivot (Requires at least 2 closed candles)
                    if len(high) >= 2:
                        prev_high = high.iloc[-2]
                        prev_low = low.iloc[-2]
                        prev_close = close.iloc[-2]
                        pivot_val = (prev_high + prev_low + prev_close) / 3
                        result["Pivot"] = round(float(pivot_val), 2)
                except Exception as e:
                    logger.error(f"⚠️ Pivot Calc Failed: {e}")

                self.last_valid_indicators = result
                self.last_indicator_update = time.time()
                return result
                
            except Exception as e:
                logger.error(f"❌ CRITICAL Indicator Failure: {e}")
                # Return partial results or defaults if complete crash
                return self.last_valid_indicators if self.last_valid_indicators else result
            
    async def execute_order(self, side_key, token_id, price, size, is_buy: bool):
        cost = price * size
        action = "BUY" if is_buy else "SELL"
        
        # Risk Check
        if is_buy and not self.settings.dry_run and self.risk_manager:
            bal = self.cached_balance if self.cached_balance else self.get_balance()
            can_trade, reason = self.risk_manager.can_trade(cost, bal)
            if not can_trade:
                logger.warning(f"⚠️ Risk Blocked: {reason}")
                return False

        logger.info(f"⚡ EXECUTING {action} {side_key} | Price: {price} | Size: {size}")
        
        # Simulation
        if self.settings.dry_run:
            # Update Simulated Balance
            if is_buy: self.sim_balance -= cost
            else: self.sim_balance += cost
            
            pnl = 0.0
            st = self.strategies[side_key] # Get the strategy state object
            inds = await self.get_indicators()

            # --- CRITICAL FIX: Update State in Simulation ---
            if is_buy:
                st.add_buy(price, size, inds.get('ATR', 0)) # <--- Update Share Count!
            else:
                pnl = (price - st.avg_price) * size
                st.reset() # <--- Clear Position on Sell
            # -----------------------------------------------

            # Log with indicators
            self.log_transaction(side_key, action, price, size, cost, pnl, 0, self.sim_balance, inds)
            return True

        # Real Order
        res = place_order(self.settings, action, token_id, price, size)
        order_id = extract_order_id(res)
        if not order_id: return False
        
        state = wait_for_terminal_order(self.settings, order_id)
        if state.get("filled"):
            avg_price = float(state.get("avg_price", price))
            real_cost = avg_price * size
            # Update position state
            st = self.strategies[side_key]
            inds = await self.get_indicators()
            
            if is_buy:
                st.add_buy(avg_price, size, inds.get('ATR', 0))
                self.log_transaction(side_key, "BUY", avg_price, size, real_cost, 0, 0, self.get_balance(), inds)
            else:
                real_pnl = (avg_price - st.avg_price) * size
                self.log_transaction(side_key, "SELL", avg_price, size, real_cost, real_pnl, 0, self.get_balance(), inds)
                st.reset()
            return True
        return False

    async def process_strategy(self, side, book_data):
        """
        Routes the WSS data to the main logic engine.
        Now constructs the combined market_data expected by evaluate_and_trade.
        """
        # We need a combined view for the logic to work, 
        # so we grab the latest values we saved in the heartbeat loop.
        market_data = {
            'UP_Bid': self.latest_up_bid,
            'UP_Ask': self.latest_up_ask,
            'DOWN_Bid': self.latest_down_bid,
            'DOWN_Ask': self.latest_down_ask
        }
        
        # Determine the specific price for this side's book
        # (This updates the 'latest' values immediately to be safe)
        if side == "UP":
            market_data['UP_Bid'] = book_data.get('best_bid', 0) or 0
            market_data['UP_Ask'] = book_data.get('best_ask', 0) or 0
        elif side == "DOWN":
            market_data['DOWN_Bid'] = book_data.get('best_bid', 0) or 0
            market_data['DOWN_Ask'] = book_data.get('best_ask', 0) or 0

        # Now call the main logic
        await self.evaluate_and_trade(market_data)

    async def evaluate_and_trade(self, market_data):
        """
        Core Logic: Checks Environment -> MFI -> Signals -> Management.
        """
        # Add this to __init__ first: 
        # self.stop_buying_minutes = int(os.getenv("STOP_BUYING_TIME_MINUTES", "5"))

        # Then in evaluate_and_trade:
        if self.get_time_remaining_minutes() < self.stop_buying_minutes:
            if not self.closing_warning_shown:
                logger.info("⏳ Market closing soon. Buying disabled.")
                self.closing_warning_shown = True
            return
        
        inds = await self.get_indicators()
        
        # Safety check: if indicators failed, abort
        if not inds or inds.get('BB_Width') is None:
            return

        # --- LAYER 1: Environment & Global Filters ---
        # 1. Volatility Check
        if inds['BB_Width'] < self.env_min_volatility:
            return 
            
        # 2. MFI Abort (Overheated Market Safety)
        # If MFI is extremely high globally, we might pause all new entries
        if inds['MFI'] >= self.mfi_abort_threshold:
            return

        # 3. Dynamic Spread Calculation
        up_ask = market_data.get('UP_Ask', 0) or 0
        up_bid = market_data.get('UP_Bid', 0) or 0
        up_spread = (up_ask - up_bid) / up_ask if up_ask > 0 else 1.0

        down_ask = market_data.get('DOWN_Ask', 0) or 0
        down_bid = market_data.get('DOWN_Bid', 0) or 0
        down_spread = (down_ask - down_bid) / down_ask if down_ask > 0 else 1.0

        # ==============================================================================
        # TIER 1: PANIC OVERRIDE (The "Knife Catcher")
        # Triggers on extreme capitulation, ignoring standard rigid filters.
        # ==============================================================================
# ==============================================================================
        # TIER 1: PANIC OVERRIDE (The "Knife Catcher")
        # ==============================================================================
        is_panic_buy_up = (
            inds['BB_PctB'] < self.panic_bb_pct and    
            inds['MFI'] < self.panic_mfi               
        )

        if is_panic_buy_up:
            st_up = self.strategies["UP"]
            
            # 1. Check Cooldown
            if (time.time() - st_up.last_buy_ts) > 10.0:
                
                # 2. Check Spread (Looser tolerance for panic)
                if up_spread < self.panic_spread:
                    
                    # 3. LADDER LOGIC
                    # - Allow up to 30 shares
                    # - Only buy if price is $0.05 better than last time
                    current_price = up_ask
                    last_price = st_up.last_buy_price if st_up.last_buy_price else 999.0
                    price_is_better = current_price < (last_price - 0.05)

                    if st_up.total_shares < 30.0 and (st_up.total_shares == 0 or price_is_better):
                        st_up.last_buy_ts = time.time() 
                        logger.info(f"🚨 PANIC BUY (UP) | MFI:{inds['MFI']} | PctB:{inds['BB_PctB']} | Spread:{up_spread:.2%}")
                        await self.execute_order("UP", self.yes_token_id, up_ask, self.initial_size * 2, True) # Note: Panic buys are often double size!
                        return

        # ==============================================================================
        # TIER 2: STANDARD CONFLUENCE ENGINE
        # ==============================================================================


        # --- LAYER 2: Entry Signals (Now including MFI) ---
        
        # UP Signal: Market is Oversold (RSI Low, MFI Low, etc.)
        is_up_signal = (
            inds['RSI'] < self.rsi_buy_up and
            inds['MFI'] < self.mfi_buy_up_max and  # <--- NEW MFI CHECK
            inds['BB_PctB'] < 0.0 and
            inds['Stoch_K'] < self.stoch_buy_up and
            inds['CCI'] < self.cci_buy_up and
            up_spread < self.env_max_spread
        )

        # DOWN Signal: Market is Overbought (RSI High, MFI High, etc.)
        is_down_signal = (
            inds['RSI'] > self.rsi_buy_down and
            inds['MFI'] > self.mfi_buy_down_min and # <--- NEW MFI CHECK
            inds['BB_PctB'] > 1.0 and
            inds['Stoch_K'] > self.stoch_buy_down and
            inds['CCI'] > self.cci_buy_down and
            down_spread < self.env_max_spread
        )

        # --- LAYER 3: Execution ---

        # Execute UP
        st_up = self.strategies["UP"]
        if is_up_signal:
            if (time.time() - st_up.last_buy_ts) > 10.0:
                current_price = up_ask
                last_price = st_up.last_buy_price if st_up.last_buy_price else 999.0
                price_is_better = current_price < (last_price - 0.05)

                if st_up.total_shares < 30.0 and (st_up.total_shares == 0 or price_is_better):
                    st_up.last_buy_ts = time.time() 
                    logger.info(f"🚀 CONFLUENCE (UP) | RSI:{inds['RSI']} MFI:{inds['MFI']} Spread:{up_spread:.2%}")
                    await self.execute_order("UP", self.yes_token_id, up_ask, self.initial_size, True)

        # Execute DOWN
# Execute DOWN
        st_down = self.strategies["DOWN"]
        if is_down_signal:
            if (time.time() - st_down.last_buy_ts) > 10.0:
                current_price = down_ask
                last_price = st_down.last_buy_price if st_down.last_buy_price else 999.0
                price_is_better = current_price < (last_price - 0.05)

                # Ladder Logic: Max 30 shares, only if price improved by $0.05
                if st_down.total_shares < 30.0 and (st_down.total_shares == 0 or price_is_better):
                    st_down.last_buy_ts = time.time()
                    logger.info(f"📉 CONFLUENCE (DOWN) | RSI:{inds['RSI']} MFI:{inds['MFI']} Spread:{down_spread:.2%}")
                    await self.execute_order("DOWN", self.no_token_id, down_ask, self.initial_size, True)

        # --- LAYER 4: Position Management (Exits) ---
        
        # Manage UP Position
        # Manage UP Position
        if st_up.total_shares > 0:
            current_roi = (up_bid - st_up.avg_price) / st_up.avg_price if st_up.avg_price > 0 else 0
            current_pnl_per_share = up_bid - st_up.avg_price

            # 1. PANIC TAKE PROFIT (>50% ROI instantly)
            if current_roi > 0.50:  
                logger.info(f"💰 PANIC PROFIT (UP) | ROI: {current_roi:.2%} | Price: {up_bid}")
                await self.execute_order("UP", self.yes_token_id, up_bid, st_up.total_shares, False)
                return

            # 2. HARD PROFIT TARGET (From .env)
            # Checks if we are up by $0.05 (or whatever PROFIT_TARGET is set to)
            # 2. HARD PROFIT TARGET (Conditional)
# Only sell if we hit target AND the market is no longer screaming "BUY"
# (e.g., only sell if RSI has risen above 40. If RSI is still 20, keep holding!)
            if current_pnl_per_share >= self.profit_target and inds['RSI'] > 40:
                logger.info(f"💰 TARGET HIT (UP) | RSI Normalized: {inds['RSI']}...")
                await self.execute_order("UP", self.yes_token_id, up_bid, st_up.total_shares, False)
                return

            # 3. TECHNICAL TAKE PROFIT (RSI / Reversion)
            time_held = time.time() - st_up.last_buy_ts
            is_crash = (up_bid / st_up.avg_price) < 0.85 if st_up.avg_price > 0 else False
            
            if (time_held > 10.0 or is_crash) and (inds['RSI'] > 50 or inds['BB_PctB'] > 0.5):
                st_up.last_buy_ts = time.time()
                logger.info("💰 TAKING PROFIT (UP) - Reversion")
                await self.execute_order("UP", self.yes_token_id, up_bid, st_up.total_shares, False)
                return  # <--- Added return to prevent falling through to Stop Loss

            # 4. ATR STOP LOSS (With MFI Panic Block)
            if st_up.avg_price > 0:
                stop_dist = st_up.entry_atr * self.atr_stop_mult if st_up.entry_atr > 0 else (st_up.avg_price * 0.15)
                stop_price = st_up.avg_price - stop_dist
                
                # Check Stop Trigger
                if up_bid < stop_price:
                    # If MFI is super low (<20), we might be at the bottom, so DON'T sell yet.
                    if inds['MFI'] < self.mfi_panic_block_max:
                        logger.info(f"🛡️ PANIC BLOCK (UP): Stop triggered but MFI {inds['MFI']} is low. Holding...")
                    else:
                        st_up.last_buy_ts = time.time()
                        logger.info(f"🛑 STOP LOSS (UP) - ATR Stop Hit @ {up_bid}")
                        await self.execute_order("UP", self.yes_token_id, up_bid, st_up.total_shares, False)

        # Manage DOWN Position
        if st_down.total_shares > 0:
            time_held = time.time() - st_down.last_buy_ts
            current_price = down_ask if down_ask > 0 else down_bid
            is_crash = (current_price / st_down.avg_price) < 0.85 if st_down.avg_price > 0 else False
            
            # Technical Take Profit
            if (time_held > 10.0 or is_crash) and (inds['RSI'] < 50 or inds['BB_PctB'] < 0.5):
                st_down.last_buy_ts = time.time()
                logger.info("💰 TAKING PROFIT (DOWN) - Reversion")
                await self.execute_order("DOWN", self.no_token_id, down_bid, st_down.total_shares, False)
            
            # ATR Stop Loss
            elif st_down.avg_price > 0:
                stop_dist = st_down.entry_atr * self.atr_stop_mult if st_down.entry_atr > 0 else (st_down.avg_price * 0.15)
                stop_price = st_down.avg_price - stop_dist 
                
                if current_price < stop_price:
                    # Logic note: MFI panic block usually applies to "Oversold" conditions. 
                    # For a DOWN position (Short), "Panic" might mean MFI is too HIGH (Overbought).
                    # However, adhering to your .env "Panic Sell Blocker" strict definition:
                    if inds['MFI'] < self.mfi_panic_block_max:
                         logger.info(f"🛡️ PANIC BLOCK (DOWN): MFI {inds['MFI']} low. Holding...")
                    else:
                        st_down.last_buy_ts = time.time()
                        logger.info(f"🛑 STOP LOSS (DOWN) - ATR Stop Hit @ {current_price}")
                        await self.execute_order("DOWN", self.no_token_id, current_price, st_down.total_shares, False)


    def show_final_summary(self, final_indicators=None):
        """Show final summary when market closes and settle open positions."""
        logger.info("\n" + "=" * 70)
        logger.info("🏁 MARKET CLOSED - FINAL SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Market: {self.market_slug}")
        
        # --- 1. SETTLEMENT LOGIC (Fixing the PnL Issue) ---
        # FIX: If strike is missing but we have price data, try to recover or proceed with caution
        if self.strike_price <= 0 and self.latest_btc_price > 0:
            logger.warning("⚠️ Strike Price missing at close! Attempting to use last known valid price...")
            # Optional: You could implement a fallback here if you store it elsewhere
            # For now, we will just log heavily so you know why it failed.

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
                                new_strike = float(candle[3])  # Index 3 is OPEN
                                # CRITICAL FIX: Only update if valid. Never overwrite a valid strike with 0.
                                if new_strike > 0:
                                    self.strike_price = new_strike
                                    logger.info(f"🎯 Strike Price Resolved: ${self.strike_price:.2f}")
                                    self.is_searching = False
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