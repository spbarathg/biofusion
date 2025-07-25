# Antbot Configuration Guide

## Configuration Consolidation

The Antbot configuration system has been **unified and simplified** to eliminate inconsistencies and redundancy.

### What Changed

**BEFORE** (Deprecated):
- `env.template` - Main configuration template
- `simplified.env.template` - Simplified configuration template
- **Issues**: Duplicated settings, inconsistent naming, conflicting defaults

**NOW** (Unified):
- `unified.env.template` - Single, comprehensive configuration template
- **Benefits**: Consistent naming, no duplication, supports both simplified and advanced modes

### Quick Start

1. **Copy the unified template:**
   ```bash
   cp config/unified.env.template config/env.production
   ```

2. **Edit the configuration:**
   ```bash
   nano config/env.production
   ```

3. **Set your bot mode:**
   - `BOT_MODE=SIMPLIFIED` - For basic trading (recommended for most users)
   - `BOT_MODE=ADVANCED` - For full features with HA, monitoring, etc.

4. **Fill in your API keys:**
   - Replace all `your_*_api_key_here` placeholders with real API keys
   - See the API Requirements section below

## Configuration Modes

### Simplified Mode (`BOT_MODE=SIMPLIFIED`)

**Best for:** Most users, testing, development

**Features:**
- Three-Stage Mathematical Core trading
- Basic monitoring and logging
- SQLite fallback database
- Single wallet management
- Essential safety features

**Ignored Settings:** HA, Redis, advanced monitoring, NATS, complex process pools

### Advanced Mode (`BOT_MODE=ADVANCED`)

**Best for:** Production deployments, enterprise users

**Features:**
- All simplified mode features
- High Availability (HA) with Redis
- Advanced monitoring with Prometheus
- NATS message bus
- Complex process pool management
- Enterprise-grade secrets management

## API Requirements

### Required APIs (All Modes)
```
BIRDEYE_API_KEY=your_birdeye_api_key_here
JUPITER_API_KEY=your_jupiter_api_key_here  
HELIUS_API_KEY=your_helius_api_key_here
SOLANA_TRACKER_API_KEY=your_solana_tracker_api_key_here
RAYDIUM_API_KEY=your_raydium_api_key_here
```

### Recommended APIs
```
DEXSCREENER_API_KEY=your_dexscreener_api_key_here
QUICKNODE_RPC_URL=your_quicknode_rpc_url_here
```

## Key Configuration Sections

### Trading Configuration
- **Initial Capital**: `INITIAL_CAPITAL_SOL=1.5`
- **Three-Stage Core**:
  - Stage 1 (Survival): `ACCEPTABLE_REL_THRESHOLD=0.1`
  - Stage 2 (Win-Rate): `HUNT_THRESHOLD=0.6`
  - Stage 3 (Growth): `KELLY_FRACTION=0.25`

### Safety Settings
- **Kill Switch**: `ENABLE_KILL_SWITCH=true`
- **Emergency Stop**: `EMERGENCY_STOP_ENABLED=true`
- **Daily Loss Limit**: `MAX_DAILY_LOSS_SOL=1.0`

### Database Options
- **Primary**: TimescaleDB (recommended for production)
- **Fallback**: SQLite (development/testing only)

## Migration from Old Templates

If you have an existing configuration:

1. **Backup your current config:**
   ```bash
   cp config/env.production config/env.production.backup
   ```

2. **Use the unified template:**
   ```bash
   cp config/unified.env.template config/env.production
   ```

3. **Transfer your settings:** Copy your API keys and custom settings to the new format

4. **Update variable names:** Some variables have been renamed for consistency:
   - `MAX_TRADE_SIZE_SOL` (old) → `MAX_TRADE_SIZE_SOL` (unchanged)
   - Added: `BOT_MODE`, `INITIAL_CAPITAL_SOL`, trading core thresholds

## Deprecated Files

The following files are **deprecated** and will be removed in future versions:

- ❌ `config/env.template` 
- ❌ `config/simplified.env.template`

Please use `config/unified.env.template` instead.

## Validation

The bot will validate your configuration on startup and report:
- ✅ Missing required API keys
- ✅ Invalid parameter combinations  
- ✅ Suggested fixes for common issues

## Troubleshooting

### Common Issues

**Issue**: Bot fails to start with config error
**Solution**: Check that you've copied `unified.env.template` to `env.production`

**Issue**: API key validation failures
**Solution**: Ensure all required API keys are set and valid

**Issue**: Database connection errors  
**Solution**: Check TimescaleDB settings or enable SQLite fallback

### Support

For configuration help:
1. Check the validation messages on startup
2. Ensure all required API keys are set
3. Verify your `BOT_MODE` setting matches your needs
4. Review the unified template for reference values

## Security Notes

- ✅ **Never** commit `env.production` to version control
- ✅ Use the vault system for sensitive data in production
- ✅ Enable wallet encryption: `WALLET_ENCRYPTION_ENABLED=true`
- ✅ Set appropriate security level: `SECURITY_LEVEL=HIGH` 