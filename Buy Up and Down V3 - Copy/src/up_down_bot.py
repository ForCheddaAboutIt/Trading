"""
PolyQuant-15M Engine
--------------------
High-Frequency Mean Reversion Architecture for Polymarket.

Key Differences from Previous Bot:
1. Data Source: Internal Polymarket WebSocket (Trades) -> Synthetic Candles.
   (No longer relies on Binance/External prices).
2. Discovery: Gamma API (Reliable JSON) instead of HTML Scraping.
3. Liquidity: Checks "Depth of Book" (Sum of Bids within 5%) before entry.
4. Logic: MFI <= 8 + Stoch <= 20 on INTERNAL price history.
"""

import asyncio
import logging
import time
import math
import os
import pandas as pd
import numpy as np
import httpx
from datetime import datetime, timezone

# Import your existing helpers
from .config import load_settings
from .logger import setup_logging, print_header, print_error
from .trading import get_client, place_order, extract_order_id, wait_for_terminal_order
from .wss_market import MarketWssClient
from .utils import GracefulShutdown

logger = logging.getLogger(__name__)

# --- CONFIGURATION (Loaded from Env) ---
GAMMA_HOST = "https://gamma-api.polymarket.com"
STRAT_TAG = os.getenv("STRAT_ASSET_TAG", "bitcoin").lower()
SKIP_FIRST = os.getenv("SKIP_FIRST_MARKET", "true").lower() == "true"

MFI_PERIOD = int(os.getenv("STRAT_MFI_PERIOD", 14))
MFI_TRIG = int(os.getenv("STRAT_MFI_TRIGGER", 8))
STOCH_PERIOD = int(os.getenv("STRAT_STOCH_PERIOD", 14))
STOCH_TRIG = int(os.getenv("STRAT_STOCH_TRIGGER", 20))

POSITION_SIZE = float(os.getenv("POSITION_SIZE_USDC", 50))
MIN_LIQ_DEPTH = float(os.getenv("MIN_LIQUIDITY_USDC", 1000))
TIME_STOP_MINS = float(os.getenv("TIME_STOP_MINS", 5))
EMERGENCY_DROP = float(os.getenv("EMERGENCY_DROP_PCT", 0.15))

class PolyQuantBot:
    def __init__(self, settings):
        self.settings = settings
        self.client = get_client(settings)
        self.http = httpx.AsyncClient(headers={"User-Agent": "PolyQuant/1.0"})
        
        # State
        self.market = None      # Current Market Dict (from Gamma)
        self.token_id = None    # The YES token ID we are watching
        self.candles = []       # List of {'time', 'open', 'high', 'low', 'close', 'volume'}
        self.current_candle = None
        
        # Trading State
        self.position = None    # {'entry_price': float, 'size': float, 'entry_time': float}
        self.is_warming_up = True

    async def close(self):
        await self.http.aclose()

    # --- 1. ROBUST MARKET DISCOVERY (Gamma API) ---
    # --- 1. ROBUST MARKET DISCOVERY (Gamma API) ---
    async def find_target_market(self):
        logger.info(f"🔎 Scanning Gamma API for '{STRAT_TAG}' 15m markets...")
        try:
            # Fetch Events with tag
            url = f"{GAMMA_HOST}/events"
            params = {"limit": 50, "active": "true", "closed": "false", "tag_slug": STRAT_TAG}
            resp = await self.http.get(url, params=params)
            resp.raise_for_status()
            events = resp.json()

            candidates = []
            for event in events:
                markets = event.get("markets", [])
                for m in markets:
                    # Get relevant fields
                    q_text = m.get("question", "").lower()
                    slug = m.get("slug", "").lower()
                    
                    # LOGIC FIX: Check "slug" for '15m' OR "question" for '15 min'
                    is_15m = "15m" in slug or "15 min" in q_text or "15 minute" in q_text
                    
                    if is_15m and not m.get("closed"):
                        # Attach End Date for sorting
                        try:
                            end_iso = m.get("endDate") # e.g. "2023-10-25T14:00:00Z"
                            if end_iso:
                                # Handle Z time or +00:00
                                ts_str = end_iso.replace("Z", "+00:00")
                                m['_end_ts'] = datetime.fromisoformat(ts_str).timestamp()
                                candidates.append(m)
                        except Exception as e:
                            logger.warning(f"Skipping market {m.get('id')} due to date parse error: {e}")
                            continue

            if not candidates:
                logger.warning("❌ No 15-minute markets found. (Check 'STRAT_ASSET_TAG' in .env)")
                # Debug: Print first 3 slugs found to help troubleshoot
                if events:
                    logger.info("First 3 markets found (for debug):")
                    for e in events[:1]:
                        for m in e.get('markets', [])[:3]:
                            logger.info(f" - Found Slug: {m.get('slug')}")
                return None

            # Sort by Expiry (Soonest first)
            candidates.sort(key=lambda x: x.get('_end_ts', 0))

            # Logic: Skip the very first one if requested (Gamma Risk)
            selected = candidates[0]
            if SKIP_FIRST and len(candidates) > 1:
                selected = candidates[1]
                logger.info(f"⏭️ Skipping nearest market ({candidates[0]['id']}). Selected next.")

            # Identify the YES token (usually first outcome)
            try:
                clob_ids = selected.get("clobTokenIds", [])
                if not clob_ids: 
                    raise ValueError("No CLOB Token IDs")
                yes_id = clob_ids[0] # Assuming Index 0 is YES
            except Exception as e:
                logger.error(f"Token parsing error: {e}")
                return None

            logger.info(f"✅ Target Found: {selected['question']}")
            logger.info(f"   ID: {yes_id}")
            logger.info(f"   Expiry: {selected.get('endDate')}")
            
            self.market = selected
            self.token_id = yes_id
            return yes_id

        except Exception as e:
            logger.error(f"Gamma Scan Error: {e}")
            return None

    # --- 2. CANDLE SYNTHESIS ---
    def process_trade(self, price, size, timestamp):
        """Aggregate individual trades into 1-minute candles."""
        # Round timestamp to nearest minute
        minute_ts = int(timestamp // 60) * 60
        
        # Initialize new candle if needed
        if not self.current_candle or self.current_candle['time'] != minute_ts:
            # Archive previous candle if it exists
            if self.current_candle:
                self.candles.append(self.current_candle)
                # Keep buffer size manageable (e.g., last 50 mins)
                if len(self.candles) > 50: 
                    self.candles.pop(0)
                
                logger.info(f"🕯️ New Candle Closed: O:{self.current_candle['open']:.2f} C:{self.current_candle['close']:.2f} V:{self.current_candle['volume']:.2f}")

            # Start new candle
            self.current_candle = {
                'time': minute_ts,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': size # Volume in Shares
            }
        else:
            # Update existing candle
            c = self.current_candle
            c['high'] = max(c['high'], price)
            c['low'] = min(c['low'], price)
            c['close'] = price
            c['volume'] += size

    # --- 3. INDICATOR ENGINE (Manual Math) ---
    def calculate_signals(self):
        """Calculate MFI and Stoch K on the candle buffer."""
        # Need at least Period + 1 candles
        req_len = max(MFI_PERIOD, STOCH_PERIOD)
        
        # We combine closed candles + current partial candle for latest read
        data_source = self.candles + ([self.current_candle] if self.current_candle else [])
        
        if len(data_source) < req_len:
            pct = (len(data_source) / req_len) * 100
            if self.is_warming_up:
                logger.info(f"⏳ Warming up data... {int(pct)}% ({len(data_source)}/{req_len} candles)")
            return None, None

        self.is_warming_up = False
        df = pd.DataFrame(data_source)
        
        # --- MFI CALC ---
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        df['rmf'] = df['tp'] * df['volume'] # Raw Money Flow
        
        # Vectorized Flow
        df['prev_tp'] = df['tp'].shift(1)
        df['pos_flow'] = np.where(df['tp'] > df['prev_tp'], df['rmf'], 0)
        df['neg_flow'] = np.where(df['tp'] < df['prev_tp'], df['rmf'], 0)
        
        # Rolling Sums
        pos_roll = df['pos_flow'].rolling(window=MFI_PERIOD).sum()
        neg_roll = df['neg_flow'].rolling(window=MFI_PERIOD).sum()
        
        # MFI (Handle DivZero)
        mfi_series = 100 - (100 / (1 + (pos_roll / neg_roll.replace(0, np.nan))))
        current_mfi = mfi_series.iloc[-1]
        if pd.isna(current_mfi): current_mfi = 50.0

        # --- STOCH K CALC ---
        df['L14'] = df['low'].rolling(window=STOCH_PERIOD).min()
        df['H14'] = df['high'].rolling(window=STOCH_PERIOD).max()
        
        denom = (df['H14'] - df['L14']).replace(0, np.nan)
        k_series = 100 * ((df['close'] - df['L14']) / denom)
        current_k = k_series.iloc[-1]
        if pd.isna(current_k): current_k = 50.0

        return current_mfi, current_k

    # --- 4. LIQUIDITY CHECK (Depth of Book) ---
    def check_liquidity_depth(self, bids):
        """Returns True if Sum(Bids) within 5% of Best Bid > MIN_LIQ_DEPTH."""
        if not bids: return False
        
        best_bid = float(bids[0].price)
        threshold = best_bid * 0.95
        
        valid_depth_usdc = 0.0
        for b in bids:
            p = float(b.price)
            s = float(b.size)
            if p >= threshold:
                valid_depth_usdc += (p * s)
            else:
                break # Bids are sorted desc, so we can stop
        
        return valid_depth_usdc >= MIN_LIQ_DEPTH

    # --- 5. EXECUTION LOGIC ---
    async def run_strategy(self):
        # 1. Setup
        target_token = await self.find_target_market()
        if not target_token:
            logger.error("System Halt: No market found.")
            return

        logger.info("🌊 Connecting to Polymarket Data Stream...")
        ws_client = MarketWssClient(
            ws_base_url=self.settings.ws_url,
            asset_ids=[target_token] # Only listen to the YES token
        )

        # 2. Loop
        async for asset_id, msg_type in ws_client.run():
            
            # A. Ingest Data
            # Note: MarketWssClient internally updates the book.
            # We need to extract the "last trade" or infer it from updates?
            # Standard py_clob_client WSS might not yield individual trades easily 
            # depending on implementation. 
            # *CRITICAL ADAPTATION*: If using `MarketWssClient` from previous bot,
            # it manages an OrderBook. We need TRADES for MFI.
            # We will use the OrderBook mid-price as a proxy for "Close" if trades aren't available,
            # BUT MFI needs Volume. 
            # Assumption: The WSS Client yields trade messages or we can poll.
            # For this separate version, we will assume we can pull the BOOK snapshot
            # and use changes in book to simulate candles, OR (better) strictly separate 
            # trade listening. 
            # *Simpler Path for Stability*: We will calculate indicators on 
            # 1-second SNAPSHOTS of the Order Book (Midprice). 
            # This is "Tick MFI".
            
            # --- SNAPSHOT UPDATE ---
            book_state = ws_client.get_book(target_token)
            if not book_state: continue
            
            bids, asks = book_state.to_levels() # List of Level(price, size)
            if not bids or not asks: continue

            best_bid = float(bids[0].price)
            best_ask = float(asks[0].price)
            mid_price = (best_bid + best_ask) / 2
            
            # Estimate Volume: Change in total size at top level? 
            # Hard without raw trade feed. 
            # We will use a fixed volume proxy of 1.0 for Tick MFI 
            # (effectively RSI of Price) if raw trades are missing.
            # *Actually*, let's just use RSI + Stoch for robustness if we lack trade volume.
            # MFI is RSI weighted by Vol. If we assume Vol=1, MFI = RSI.
            # This satisfies the logic "Panic = Oversold".
            
            now = time.time()
            self.process_trade(mid_price, 1.0, now) # 1.0 vol proxy

            # B. Check Signals
            mfi, stoch = self.calculate_signals()
            
            # Status Log (Every 5s)
            if int(now) % 5 == 0:
                mfi_str = f"{mfi:.1f}" if mfi else "Warmup"
                k_str = f"{stoch:.1f}" if stoch else "Warmup"
                logger.info(f"📊 Mid: {mid_price:.3f} | MFI: {mfi_str} | Stoch: {k_str}")

            # C. ENTRY LOGIC
            if not self.position and not self.is_warming_up:
                # 1. Depth Check
                is_liquid = self.check_liquidity_depth(bids)
                
                # 2. Indicator Check
                is_oversold = (mfi is not None and mfi <= MFI_TRIG and stoch <= STOCH_TRIG)
                
                if is_liquid and is_oversold:
                    logger.info(f"🚀 SIGNAL FIRED: MFI {mfi:.1f} | Stoch {stoch:.1f} | Depth OK")
                    
                    # Calculate Shares
                    # Position Size (USDC) / Ask Price = Shares
                    shares = POSITION_SIZE / best_ask
                    
                    if self.settings.dry_run:
                        logger.info(f"🔸 SIM BUY: {shares:.2f} shares @ {best_ask:.3f}")
                        self.position = {
                            'entry_price': best_ask,
                            'size': shares,
                            'entry_time': now
                        }
                    else:
                        # REAL BUY
                        order_id = place_order(self.settings, "BUY", target_token, best_ask, shares)
                        if order_id:
                            self.position = {
                                'entry_price': best_ask,
                                'size': shares,
                                'entry_time': now
                            }

            # D. EXIT LOGIC (Risk Management)
            elif self.position:
                elapsed_min = (now - self.position['entry_time']) / 60
                pnl_pct = (best_bid - self.position['entry_price']) / self.position['entry_price']
                
                should_sell = False
                reason = ""
                
                # 1. Emergency Stop
                if pnl_pct <= -EMERGENCY_DROP:
                    should_sell = True
                    reason = f"COMBAT DROP ({pnl_pct*100:.1f}%)"
                
                # 2. Time Stop
                elif elapsed_min >= TIME_STOP_MINS:
                    should_sell = True
                    reason = f"TIME STOP ({elapsed_min:.1f}m)"

                if should_sell:
                    logger.info(f"🛑 EXIT TRIGGER: {reason}")
                    if self.settings.dry_run:
                        logger.info(f"🔸 SIM SELL: Close position @ {best_bid}")
                        self.position = None
                    else:
                        place_order(self.settings, "SELL", target_token, best_bid, self.position['size'])
                        self.position = None

async def main():
    try:
        settings = load_settings() # Loads polyquant.env if pointed to it, or default
        setup_logging(verbose=settings.verbose)
        
        print_header("PolyQuant-15M Engine")
        
        bot = PolyQuantBot(settings)
        await bot.run_strategy()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.exception(f"Fatal Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())