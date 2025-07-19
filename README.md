# Solana Trading Bot

A sophisticated automated trading bot for Solana blockchain, designed for memecoin trading with AI-driven sentiment analysis and risk management.

## Features

- **Automated Trading**: Real-time market scanning and trade execution
- **AI Sentiment Analysis**: FinBERT-based sentiment analysis for trading decisions
- **Multi-Wallet Management**: Genetic evolution system with 10 trading wallets
- **Risk Management**: Kill switches, stop-losses, and position sizing
- **Real-time Monitoring**: Comprehensive performance tracking and alerts
- **Jupiter DEX Integration**: Direct integration with Jupiter aggregator
- **Vault System**: Secure profit protection and compounding

## Quick Start

### Prerequisites
- Python 3.11+
- Solana wallet with SOL for trading
- API keys for Helius, Birdeye, Jupiter, and DexScreener

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd antbotNew

# Install dependencies
pip install -r config/requirements.txt

# Set up environment
cp config/env.template .env.production
# Edit .env.production with your API keys and configuration
```

### Configuration

Required API keys in `.env.production`:
```bash
HELIUS_API_KEY=your_helius_key
BIRDEYE_API_KEY=your_birdeye_key
JUPITER_API_KEY=your_jupiter_key
DEXSCREENER_API_KEY=your_dexscreener_key
```

### Running the Bot

```bash
# Simulation mode (paper trading)
python entry_points/run_bot.py --mode simulation --capital 300

# Production mode (real trading)
python entry_points/run_bot.py --mode production --capital 300

# Test mode
python entry_points/run_bot.py --mode test
```

## Architecture

The bot consists of several core systems:

- **Trading Engine**: Handles trade execution and position management
- **Market Scanner**: Scans for trading opportunities across DEXs
- **Sentiment AI**: Analyzes market sentiment for trading decisions
- **Wallet Manager**: Manages multiple trading wallets with genetic evolution
- **Vault System**: Secures profits and handles compounding
- **Safety Systems**: Kill switches and risk management

## Configuration

Key trading parameters:
- `INITIAL_CAPITAL`: Starting capital in USD
- `POSITION_SIZE_PERCENT`: Percentage of capital per position
- `MAX_POSITION_SIZE_SOL`: Maximum SOL per position
- `MIN_PROFIT_TARGET`: Minimum profit target percentage
- `MAX_LOSS_PERCENT`: Maximum loss percentage
- `MAX_HOLD_TIME_HOURS`: Maximum position hold time

## Safety Features

- **Kill Switch**: Emergency shutdown on significant losses
- **Position Limits**: Maximum concurrent positions
- **Capital Protection**: Vault system secures profits
- **Stop Loss**: Automatic loss protection
- **Time Limits**: Maximum position hold times

## Monitoring

The bot provides real-time monitoring of:
- Trading performance and profit/loss
- Active positions and win rate
- System health and uptime
- Wallet performance and evolution

## Risk Warnings

- **High Risk**: Memecoin trading is extremely volatile
- **Capital Loss**: You can lose your entire investment
- **No Guarantees**: Past performance doesn't guarantee future results
- **Market Conditions**: Performance depends on market conditions
- **Technical Risks**: Software bugs, network issues, API failures

## Legal Considerations

- Ensure compliance with local trading regulations
- Trading profits may be taxable
- Respect API terms and rate limits
- You are responsible for your trading decisions

## Support

For issues and questions:
- Check the configuration and API keys
- Review logs for error messages
- Test in simulation mode first
- Ensure sufficient SOL for gas fees

## License

This project is for educational and research purposes. Use at your own risk. 