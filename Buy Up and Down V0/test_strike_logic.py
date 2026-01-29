import asyncio
import httpx
import re
import logging
from datetime import datetime, timezone

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("RetryTest")

async def get_market_slug():
    """Find the current live market."""
    print("🔎 Finding live market...")
    page_url = "https://polymarket.com/crypto/15M"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(page_url, headers=headers)
        
    matches = re.findall(r'btc-updown-15m-(\d+)', resp.text)
    if not matches: raise RuntimeError("No market found")

    now_ts = int(datetime.now(timezone.utc).timestamp())
    all_ts = sorted((int(ts) for ts in matches), reverse=False)
    
    # Filter for Open Markets
    open_ts = [ts for ts in all_ts if now_ts < (ts + 900)]
    
    if not open_ts: raise RuntimeError("No open markets found.")
    return open_ts[0]

async def fetch_coinbase_open(target_ts):
    """Try to fetch the price. Returns 0.0 if not ready."""
    try:
        dt_start = datetime.fromtimestamp(target_ts, timezone.utc).isoformat()
        dt_end = datetime.fromtimestamp(target_ts + 60, timezone.utc).isoformat()
        
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        params = {"start": dt_start, "end": dt_end, "granularity": 60}
        headers = {"User-Agent": "Mozilla/5.0"}

        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                for candle in data:
                    # Candle: [time, low, high, open, close, volume]
                    if candle[0] == target_ts:
                        return float(candle[3]) # Index 3 is OPEN
            else:
                logger.warning(f"❌ API Status: {resp.status_code}")
                
    except Exception as e:
        logger.warning(f"❌ Connection Error: {e}")

    return 0.0

async def main():
    print("\n" + "="*50)
    print("🔄 TESTING RETRY LOGIC (Loop until Found)")
    print("="*50)
    
    try:
        ts = await get_market_slug()
        slug = f"btc-updown-15m-{ts}"
        dt = datetime.fromtimestamp(ts, timezone.utc).strftime('%H:%M:%S UTC')
        
        print(f"👉 Market: {slug}")
        print(f"👉 Start Time: {ts} ({dt})")
        print("-" * 50)
        
        attempt = 1
        price = 0.0
        
        # === THE RETRY LOOP ===
        while price == 0.0:
            logger.info(f"Attempt #{attempt}: Checking Coinbase...")
            price = await fetch_coinbase_open(ts)
            
            if price > 0.0:
                print("\n" + "="*50)
                print(f"✅ SUCCESS! Price Found: ${price:,.2f}")
                print("="*50)
                break
            else:
                logger.warning("⏳ Price not ready (0.0). Retrying in 10 seconds...")
                await asyncio.sleep(10)
                attempt += 1
                
    except Exception as e:
        print(f"❌ Critical Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test stopped by user.")