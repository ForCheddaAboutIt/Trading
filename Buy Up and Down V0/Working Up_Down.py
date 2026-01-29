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
from datetime import datetime
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
    def log_transaction(self, side, action, price, size, value, pnl, start_bal, curr_bal):
        """Append transaction details to a CSV file."""
        file_exists = os.path.isfile(self.transaction_file)
        try:
            with open(self.transaction_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    # Write Header
                    writer.writerow([
                        "Timestamp", "Market", "Side", "Action", 
                        "Price", "Size", "Total_Value", "PnL", 
                        "Start_Balance", "Current_Balance"
                    ])
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([
                    timestamp, self.market_slug, side, action,
                    f"{price:.4f}", f"{size:.2f}", f"{value:.2f}", 
                    f"{pnl:.2f}", f"{start_bal:.2f}", f"{curr_bal:.2f}"
                ])
        except Exception as e:
            logger.error(f"Failed to write to transaction log: {e}")

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
                 target_price = new_avg + self.profit_target
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

        # --- SIMULATION ---
        if self.settings.dry_run:
            logger.info("🔸 SIMULATION MODE - Executing virtually")
            if is_buy: self.sim_balance -= cost
            else: self.sim_balance += cost
            
            # --- ADD THIS: Log to CSV ---
            self.log_transaction(side_key, action, price, size, cost, pnl, s_bal, self.sim_balance)

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
                self.log_transaction(side_key, action, fill_price, size, real_cost, real_pnl, s_bal, self.cached_balance)

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
                sell_amt = normalize_size(st.total_shares * 0.5, self.min_order_size)
                
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

    def show_final_summary(self):
        """Show final summary when market closes."""
        logger.info("\n" + "=" * 70)
        logger.info("🏁 MARKET CLOSED - FINAL SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Market: {self.market_slug}")
        
        up_pnl = self.strategies["UP"].realized_pnl
        down_pnl = self.strategies["DOWN"].realized_pnl
        total_pnl = up_pnl + down_pnl
        
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
            # For real trading, we use 0.0 or fetch current if needed
            s_bal = 0.0
            e_bal = self.cached_balance if self.cached_balance else 0.0

        logger.info("=" * 70)

        # --- ADD THIS: Write to CSV ---
        self.log_session_summary(s_bal, e_bal, total_pnl)

    # --- Helper to parse WSS book objects ---
    def _book_from_state(self, bid_levels, ask_levels) -> dict:
        best_bid = max((float(p) for p, _ in bid_levels), default=None) if bid_levels else None
        best_ask = min((float(p) for p, _ in ask_levels), default=None) if ask_levels else None
        return {"best_bid": best_bid, "best_ask": best_ask}

    async def monitor_wss(self):
        """Monitor using Polymarket CLOB Market WebSocket."""
        while True:
            # 1. Market Rollover Check
            if self.get_time_remaining() == "CLOSED":
                logger.info("\n🚨 Market has closed.")
                self.show_final_summary()
                logger.info("\n🔄 Searching for next BTC 15min market...")
                try:
                    new_market = find_current_btc_15min_market()
                    if new_market != self.market_slug:
                        current_balance = self.sim_balance
                        self.__init__(self.settings) # Re-init bot
                        # self.sim_balance = current_balance
                        # self.sim_start_balance = current_balance
                        logger.info(f"💰 Carried over balance: ${self.sim_balance:.2f}")
                        continue
                    await asyncio.sleep(10)
                    continue
                except Exception as e:
                    logger.error(f"Search error: {e}")
                    await asyncio.sleep(10)
                    continue

            # 2. Start WSS Logging
            logger.info("=" * 70)
            logger.info("🚀 AVERAGE DOWN BOT STARTED (WSS MODE)")
            logger.info("=" * 70)
            logger.info(f"Market: {self.market_slug}")
            logger.info(f"Time Left: {self.get_time_remaining()}")
            logger.info(f"Mode: {'🔸 SIMULATION' if self.settings.dry_run else '🔴 REAL TRADING'}")
            logger.info("-" * 70)
            logger.info(f"Entry: ${self.entry_price} | Drop: ${self.drop_interval}")
            logger.info(f"Mult: {self.size_multiplier}x | Profit: +${self.profit_target}")
            logger.info("=" * 70)

            client = MarketWssClient(
                ws_base_url=self.settings.ws_url,
                asset_ids=[self.yes_token_id, self.no_token_id],
            )

            last_eval = 0.0
            
            try:
                # 3. WSS Loop
                async for asset_id, event_type in client.run():
                    # Periodic Close Check
                    if self.get_time_remaining() == "CLOSED":
                        break # Break loop to trigger rollover

                    # Rate Limit Logic (Debounce)
                    now = time.time()
                    if (now - last_eval) < 0.1: # 100ms debounce
                        continue
                    last_eval = now

                    # Fetch State
                    yes_state = client.get_book(self.yes_token_id)
                    no_state = client.get_book(self.no_token_id)
                    
                    if not yes_state or not no_state: continue

                    # Safe Conversion
                    yes_bids, yes_asks = yes_state.to_levels()
                    no_bids, no_asks = no_state.to_levels()
                    
                    up_book = self._book_from_state(yes_bids, yes_asks)
                    down_book = self._book_from_state(no_bids, no_asks)

                    # Process Strategy
                    await self.process_strategy("UP", up_book)
                    await self.process_strategy("DOWN", down_book)

                    # Status Log (Every 1s)
                    if (now - self._last_log_ts) >= .3:
                        self._last_log_ts = now
                        
                        p_up = up_book.get("best_ask") or 0.0
                        p_down = down_book.get("best_ask") or 0.0
                        
                        # Holdings String
                        holding_str = ""
                        u_shares = self.strategies["UP"].total_shares
                        d_shares = self.strategies["DOWN"].total_shares
                        if u_shares > 0 or d_shares > 0:
                            parts = []
                            if u_shares > 0: parts.append(f"UP:{u_shares:.1f}")
                            if d_shares > 0: parts.append(f"DN:{d_shares:.1f}")
                            holding_str = f" | 💼 {' '.join(parts)}"

                        logger.info(f"⏱️ {self.get_time_remaining()} | 🏷️ UP: ${p_up:.3f} / DOWN: ${p_down:.3f}{holding_str}")

            except Exception as e:
                logger.warning(f"WSS monitor error: {e}. Reconnecting...")
                await asyncio.sleep(2)
                continue

async def main():
    shutdown = GracefulShutdown()
    try:
        settings = load_settings()
        setup_logging(verbose=settings.verbose, use_rich=settings.use_rich_output)
        
        if not ConfigValidator.validate_and_print(settings):
            print_error("Configuration invalid.")
            return
        
        print_header("🚀 BTC 15-Minute Average Down Bot")
        bot = AverageDownBot(settings)
        
        # Shutdown hook
        def on_shutdown():
            bot.show_final_summary()
        shutdown.register_callback(on_shutdown)
        
        await bot.monitor_wss()
        
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.exception(f"Fatal Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())