import logging

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validates the configuration settings."""
    
    @staticmethod
    def validate_and_print(settings) -> bool:
        """
        Validates settings and prints any errors.
        Returns True if valid, False otherwise.
        """
        errors = []
        
        # 1. API Credentials (only needed for real trading)
        if not settings.dry_run:
            if not settings.api_key: errors.append("Missing POLYMARKET_API_KEY")
            if not settings.api_secret: errors.append("Missing POLYMARKET_API_SECRET")
            if not settings.api_passphrase: errors.append("Missing POLYMARKET_API_PASSPHRASE")
            if not settings.private_key: errors.append("Missing POLYMARKET_PRIVATE_KEY")
        
        # 2. Strategy Logic Checks
        if settings.entry_price <= 0 or settings.entry_price >= 1:
            errors.append(f"ENTRY_PRICE ({settings.entry_price}) must be between 0 and 1")
            
        if settings.min_price_threshold >= settings.entry_price:
            errors.append(f"MIN_PRICE_THRESHOLD ({settings.min_price_threshold}) must be lower than ENTRY_PRICE ({settings.entry_price})")
            
        if settings.size_multiplier < 1.0:
            errors.append(f"SIZE_MULTIPLIER ({settings.size_multiplier}) should be >= 1.0 to effectively average down")
            
        if settings.sell_percentage <= 0 or settings.sell_percentage > 1:
            errors.append(f"SELL_PERCENTAGE ({settings.sell_percentage}) must be between 0.0 and 1.0 (e.g. 0.5 for 50%)")
            
        if settings.trailing_stop_percent <= 0 or settings.trailing_stop_percent >= 1:
             errors.append(f"TRAILING_STOP_PERCENT ({settings.trailing_stop_percent}) must be between 0.0 and 1.0")

        # 3. WebSocket Checks
        if settings.use_wss and not settings.ws_url:
            errors.append("USE_WSS is true but POLYMARKET_WS_URL is missing")

        # Print Errors
        if errors:
            logger.error("❌ Configuration Validation Failed:")
            for err in errors:
                logger.error(f"   - {err}")
            return False
            
        return True