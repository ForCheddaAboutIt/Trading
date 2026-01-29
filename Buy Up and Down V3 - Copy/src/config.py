import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env file from project root if present.
# Do NOT override existing environment variables (so CI/terminal env wins over .env).
load_dotenv(override=False)

@dataclass
class Settings:
    # --- Polymarket API Credentials ---
    api_key: str = os.getenv("POLYMARKET_API_KEY", "")
    api_secret: str = os.getenv("POLYMARKET_API_SECRET", "")
    api_passphrase: str = os.getenv("POLYMARKET_API_PASSPHRASE", "")
    private_key: str = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    signature_type: int = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "1"))
    funder: str = os.getenv("POLYMARKET_FUNDER", "")
    
    # --- Market Configuration ---
    market_slug: str = os.getenv("POLYMARKET_MARKET_SLUG", "")
    
    # --- Strategy: Average Down Parameters ---
    # These match the new variables you added to .env
    entry_price: float = float(os.getenv("ENTRY_PRICE", "0.35"))
    min_price_threshold: float = float(os.getenv("MIN_PRICE_THRESHOLD", "0.20"))
    initial_order_size: float = float(os.getenv("INITIAL_ORDER_SIZE", "5"))
    size_multiplier: float = float(os.getenv("SIZE_MULTIPLIER", "1.5"))
    price_drop_interval: float = float(os.getenv("PRICE_DROP_INTERVAL", "0.02"))
    profit_target: float = float(os.getenv("PROFIT_TARGET", "0.05"))
    sell_percentage: float = float(os.getenv("SELL_PERCENTAGE", "0.5"))
    trailing_stop_percent: float = float(os.getenv("TRAILING_STOP_PERCENT", "0.05"))
    stop_buying_time_minutes: float = float(os.getenv("STOP_BUYING_TIME_MINUTES", "5"))

    # --- Bot & Connection Settings ---
    order_type: str = os.getenv("ORDER_TYPE", "FOK").upper()
    dry_run: bool = os.getenv("DRY_RUN", "false").lower() == "true"
    verbose: bool = os.getenv("VERBOSE", "false").lower() == "true"
    cooldown_seconds: float = float(os.getenv("COOLDOWN_SECONDS", "10"))
    sim_balance: float = float(os.getenv("SIM_BALANCE", "0"))
    
    ws_url: str = os.getenv("POLYMARKET_WS_URL", "wss://ws-subscriptions-clob.polymarket.com")
    use_wss: bool = os.getenv("USE_WSS", "false").lower() == "true"

    # --- Risk Management (Optional/Legacy) ---
    max_daily_loss: float = float(os.getenv("MAX_DAILY_LOSS", "0"))  
    max_position_size: float = float(os.getenv("MAX_POSITION_SIZE", "0"))
    max_trades_per_day: int = int(os.getenv("MAX_TRADES_PER_DAY", "0"))
    min_balance_required: float = float(os.getenv("MIN_BALANCE_REQUIRED", "10.0"))
    max_balance_utilization: float = float(os.getenv("MAX_BALANCE_UTILIZATION", "0.8"))
    
    # --- Logging ---
    enable_stats: bool = os.getenv("ENABLE_STATS", "true").lower() == "true"
    trade_log_file: str = os.getenv("TRADE_LOG_FILE", "trades.json")
    use_rich_output: bool = os.getenv("USE_RICH_OUTPUT", "true").lower() == "true"

def load_settings() -> Settings:
    return Settings()