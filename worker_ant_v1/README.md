# 🐜 Worker Ant V1 - MVP Memecoin Trading Bot

**One Ant That Makes Money**

A simplified, profitable memecoin trading bot for Solana that focuses on execution speed and consistent small gains.

---

## 🎯 What It Does

Worker Ant V1 is a **money-making MVP** that:

- **Scans** for new token listings on Raydium/Orca
- **Buys** promising tokens within 200ms  
- **Sells** automatically using profit targets, stop losses, and timeouts
- **Tracks** every trade with detailed KPIs
- **Protects** capital with built-in safety limits

**No complex AI, no over-engineering—just profitable trading.**

---

## 🚀 Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r worker_ant_v1/requirements.txt

# Set your wallet (REQUIRED for real trading)
export WALLET_PRIVATE_KEY="your_base58_private_key"

# Optional: Set API keys for better data
export BIRDEYE_API_KEY="your_birdeye_key"
export SOLANA_RPC_URL="your_private_rpc_url"
```

### 2. Fund Your Wallet

Send **at least 0.2 SOL** to your trading wallet for:
- Trade execution (~0.05 SOL per trade)
- Transaction fees
- Safety buffer

### 3. Run

```bash
python worker_ant_v1/run.py
```

---

## ⚙️ Configuration

Edit values in `worker_ant_v1/config.py` or use environment variables:

### Core Trading Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `trade_amount_sol` | 0.05 | SOL amount per trade (~$20-50) |
| `profit_target_percent` | 10.0 | Sell when profit hits 10% |
| `stop_loss_percent` | 10.0 | Sell when loss hits 10% |
| `timeout_exit_seconds` | 45 | Sell if no movement for 45s |
| `max_slippage_percent` | 1.5 | Max allowed slippage |

### Safety Limits

| Setting | Default | Description |
|---------|---------|-------------|
| `max_trades_per_hour` | 20 | Rate limiting |
| `max_daily_loss_sol` | 1.0 | Circuit breaker |
| `min_time_between_trades_seconds` | 30 | Cooldown period |

### Scanner Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `scan_interval_seconds` | 0.5 | How often to scan |
| `min_liquidity_sol` | 5.0 | Minimum pool liquidity |
| `max_pool_age_seconds` | 300 | Focus on tokens <5min old |

---

## 📊 KPIs Tracked

The bot automatically tracks:

- **Win Rate** - % of profitable trades
- **Average P&L** - Per trade profit/loss  
- **Total Profit** - Session total in SOL
- **Slippage** - Execution efficiency
- **Latency** - Speed metrics
- **Max Drawdown** - Largest loss streak

Access metrics via:
- Real-time logs in console
- SQLite database: `worker_ant_v1/trades.db`
- JSON exports: `worker_ant_v1/session_metrics_*.json`

---

## 🛡️ Safety Features

### Built-in Protections

1. **Daily Loss Limits** - Stops trading if losses exceed threshold
2. **Position Limits** - Max 3 concurrent positions  
3. **Balance Checks** - Ensures sufficient funds before trading
4. **Blacklist Filter** - Avoids known problematic tokens
5. **Emergency Exits** - Rug pull detection and instant sells

### Manual Controls

```python
# Emergency stop all positions
position_manager.emergency_exit_all()

# Check current status
position_manager.get_position_summary()

# Manual exit specific token
position_manager.manual_exit_position("token_address")
```

---

## 📁 Code Structure

```
worker_ant_v1/
├── __init__.py       # Module initialization
├── config.py         # All configuration settings
├── logger.py         # Trade logging and KPI tracking
├── scanner.py        # Token discovery and filtering
├── buyer.py          # Fast trade execution
├── seller.py         # Exit strategy management
├── main.py           # Main orchestrator
├── run.py            # Startup script
├── requirements.txt  # Dependencies
└── README.md         # This file
```

---

## 🔧 Advanced Usage

### Environment Variables

```bash
# Wallet (Required for real trading)
export WALLET_PRIVATE_KEY="base58_private_key"

# Performance  
export SOLANA_RPC_URL="https://your-private-rpc.com"
export BIRDEYE_API_KEY="your_api_key"

# Trading Parameters
export TRADE_AMOUNT_SOL="0.1"
export PROFIT_TARGET_PERCENT="15"
export STOP_LOSS_PERCENT="8"
```

### Programmatic Usage

```python
from worker_ant_v1.main import WorkerAntV1
import asyncio

async def custom_trading():
    bot = WorkerAntV1()
    await bot.start()

# Run custom logic
asyncio.run(custom_trading())
```

---

## 🚨 Important Notes

### ⚠️ Risk Warnings

- **Memecoin trading is extremely risky**
- **Most tokens go to zero**  
- **Only trade with money you can afford to lose**
- **Start with small amounts to test**

### 💡 Optimization Tips

1. **Use private RPC** for faster execution
2. **Monitor during high activity** (US market hours)
3. **Adjust trade size** based on market conditions
4. **Review logs regularly** to optimize settings

### 🔒 Security

- **Never share your private key**
- **Use a dedicated trading wallet**
- **Keep most funds in cold storage**
- **Monitor wallet activity regularly**

---

## 📈 What's Next

This MVP focuses on **one profitable ant**. Future enhancements:

1. **Multiple Strategy Variants** (different timing, sizing)
2. **Enhanced Rug Pull Detection** 
3. **Social Sentiment Integration**
4. **Multi-Wallet Scaling**
5. **Advanced KPI Dashboard**

---

## 🐜 Philosophy

> "The best trading bot is one that makes money consistently, not one with the most features."

Worker Ant V1 strips away complexity to focus on what matters:
- **Speed** - Get in fast before others
- **Discipline** - Stick to profit targets and stop losses  
- **Safety** - Protect capital above all else
- **Simplicity** - Code you can understand and modify

---

## 📞 Support

For issues or questions:

1. Check the logs in `worker_ant_v1/trades.log`
2. Review the SQLite database for trade history
3. Verify your wallet has sufficient balance
4. Ensure API keys are valid

**Remember: This is an MVP focused on profitability, not feature completeness.**

---

*🐜 Happy hunting! May your trades be swift and profitable. 🐜* 