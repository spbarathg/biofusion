# 🦍 SMART APE TRADING BOT - PRODUCTION READY

A sophisticated memecoin trading bot with real-time sentiment analysis, multi-wallet management, and advanced risk management.

## 🚀 Features

### Core Trading Engine
- **Real DEX Integration**: Connects to Jupiter DEX for actual trades
- **Multi-Wallet Swarm**: 10 evolving wallets with performance tracking
- **Live Market Scanning**: Real-time opportunity detection from multiple sources
- **Surgical Execution**: Precision trade execution with minimal slippage

### Intelligence Systems
- **Twitter Sentiment Analysis via Discord**: Real-time sentiment scoring from a Discord server that relays Twitter data
- **Pattern Recognition**: Advanced market pattern detection

### Safety & Monitoring
- **Kill Switch**: Emergency shutdown with multiple triggers
- **Rug Detection**: Advanced scam detection and filtering
- **Vault System**: Automatic profit protection
- **Real-time Monitoring**: Comprehensive dashboards and alerts

### Performance
- **100+ Trades/Hour**: High-frequency trading capability
- **Adaptive Sleep**: Dynamic timing based on market activity
- **Rate Limiting**: Intelligent API usage management
- **Error Recovery**: Robust error handling and retry logic

## 📋 Prerequisites

### Required Software
- Python 3.9+
- Redis Server
- Solana CLI (optional)

### Required API Keys
- **Discord Bot Token**: For connecting to the subscription server that relays Twitter data

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd antbotNew
```

### 2. Install Dependencies
```bash
pip install -r config/requirements.txt
```

### 3. Set Up Environment
```bash
# Copy environment template
cp config/env.template .env.production

# Edit with your actual values
nano .env.production
```

### 4. Configure API Keys
Edit `.env.production` and add your Discord bot credentials:

```env
# Discord bot for Twitter sentiment feed
DISCORD_BOT_TOKEN=your_discord_bot_token
DISCORD_SERVER_ID=your_subscription_server_id
DISCORD_CHANNEL_ID=twitter_feed_channel_id
```

### 5. Set Up Wallets
Add your wallet private keys to `.env.production`:

```bash
# Trading wallets (comma-separated)
WALLET_PRIVATE_KEYS=your_private_key_1,your_private_key_2,your_private_key_3

# Vault wallet for profit protection
VAULT_WALLET_PRIVATE_KEY=your_vault_wallet_private_key
```

### 6. Start Redis
```bash
# Install Redis (Ubuntu/Debian)
sudo apt-get install redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

## 🚀 Usage

### Production Mode
```bash
python entry_points/run_bot.py --mode production --capital 10.0
```

### Simulation Mode (Paper Trading)
```bash
python entry_points/run_bot.py --mode simulation --capital 10.0
```

### Test Mode
```bash
python entry_points/run_bot.py --mode test
```

## 📊 Configuration

### Trading Parameters
```bash
# Position sizing
MAX_POSITION_SIZE_SOL=1.0
MAX_CONCURRENT_POSITIONS=5

# Risk management
TAKE_PROFIT_PERCENT=10.0
STOP_LOSS_PERCENT=5.0
MAX_DAILY_LOSS_SOL=5.0

# Market filters
MIN_LIQUIDITY_SOL=10.0
MIN_VOLUME_24H_SOL=100.0
MAX_SLIPPAGE_PERCENT=2.0
```

### Performance Tuning
```bash
# Trading frequency
MAX_TRADES_PER_HOUR=100
SCAN_INTERVAL_SECONDS=30
POSITION_CHECK_INTERVAL_SECONDS=15

# Sentiment updates
SENTIMENT_UPDATE_INTERVAL_SECONDS=60
```

## 🔧 How It Works

### 1. Market Scanning
The bot continuously scans for trading opportunities:
- **Jupiter API**: Real-time price and liquidity data
- **Birdeye API**: Trending tokens and market data
- **Twitter Sentiment via Discord**: Real-time Twitter data is relayed to a Discord server, which the bot monitors for sentiment analysis

### 2. Sentiment Analysis
For each opportunity, the bot analyzes sentiment:
- **Twitter**: Real-time tweet sentiment scoring
- **News**: Crypto news sentiment via CryptoPanic
- **Reddit**: Community sentiment from Reddit posts

### 3. Opportunity Validation
Opportunities are filtered based on:
- Minimum sentiment score (0.7+)
- Minimum liquidity (10+ SOL)
- Minimum volume (100+ SOL 24h)
- Maximum price impact (3% max)

### 4. Trade Execution
When an opportunity is validated:
- Selects best performing wallet
- Calculates optimal position size
- Executes trade via Jupiter DEX
- Monitors position for exit conditions

### 5. Position Management
Positions are managed with:
- **Take Profit**: 10% profit target
- **Stop Loss**: 5% loss limit
- **Sentiment Exit**: Exit if sentiment drops below 0.3
- **Time Exit**: Maximum 2-hour hold time

## 🛡️ Safety Features

### Kill Switch
The bot automatically shuts down if:
- Daily loss exceeds threshold
- Anomaly detection triggers
- System errors exceed limit
- Security breach detected

### Rug Detection
Advanced filtering prevents trading:
- Known scam tokens
- Low liquidity tokens
- Suspicious contract patterns
- Honeypot tokens

### Vault Protection
Profits are automatically moved to:
- Secure vault wallet
- Encrypted storage
- Backup systems

## 📈 Monitoring

### Real-time Metrics
- Total trades executed
- Win rate percentage
- Total profit/loss
- Active positions
- System health status

### Alerts
The bot sends alerts for:
- New trades executed
- Profitable exits
- Losses taken
- System errors
- Kill switch activation

### Dashboards
Optional monitoring dashboards:
- Grafana for metrics visualization
- Prometheus for data collection
- Custom web interface

## 🔄 Wallet Evolution

### Performance Tracking
Each wallet tracks:
- Total trades executed
- Success rate
- Average profit per trade
- Risk score

### Evolution Process
Every 24 hours:
- Bottom 20% wallets are removed
- Top performers are cloned
- New wallets are created with mutations
- Performance metrics are recalibrated

## 🚨 Emergency Procedures

### Manual Kill Switch
```bash
# Stop the bot immediately
Ctrl+C

# Or trigger emergency shutdown
echo "EMERGENCY" > /tmp/kill_switch
```

### Recovery
```bash
# Reset kill switch (requires override code)
python -c "from worker_ant_v1.safety.kill_switch import EnhancedKillSwitch; EnhancedKillSwitch().reset('MANUAL_OVERRIDE_ACCEPTED')"
```

## 📝 Logging

### Log Files
- `logs/trading_bot.log`: Main trading log
- `logs/errors.log`: Error tracking
- `logs/performance.log`: Performance metrics

### Log Levels
- `DEBUG`: Detailed debugging information
- `INFO`: General information
- `WARNING`: Warning messages
- `ERROR`: Error messages
- `CRITICAL`: Critical system errors

## 🔧 Troubleshooting

### Common Issues

**1. API Rate Limits**
```bash
# Reduce scan frequency
SCAN_INTERVAL_SECONDS=60
SENTIMENT_UPDATE_INTERVAL_SECONDS=120
```

**2. Network Issues**
```bash
# Increase timeouts
REQUEST_TIMEOUT_SECONDS=30
MAX_RETRIES=5
```

**3. Wallet Issues**
```bash
# Check wallet balances
python -c "from worker_ant_v1.core.wallet_manager import get_wallet_manager; print(await get_wallet_manager().get_balances())"
```

**4. Redis Connection**
```bash
# Test Redis connection
redis-cli ping
```

## 📚 API Documentation

### Required APIs

#### Twitter API v2
- **Purpose**: Sentiment analysis
- **Setup**: https://developer.twitter.com/en/docs/getting-started
- **Rate Limit**: 300 requests/15min

#### Jupiter API
- **Purpose**: DEX trading
- **Setup**: https://station.jup.ag/docs/apis/swap-api
- **Rate Limit**: 100 requests/min

#### Birdeye API
- **Purpose**: Market data
- **Setup**: https://public-api.birdeye.so
- **Rate Limit**: 50 requests/min

### Optional APIs

#### CryptoPanic API
- **Purpose**: News sentiment
- **Setup**: https://cryptopanic.com/developers/api/
- **Rate Limit**: 100 requests/min

#### Reddit API
- **Purpose**: Community sentiment
- **Setup**: https://www.reddit.com/prefs/apps
- **Rate Limit**: 60 requests/min

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational purposes only. Trading cryptocurrencies involves significant risk. Use at your own risk. The authors are not responsible for any financial losses.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the logs for error details

---

**🚀 Ready to trade like a smart ape? Set up your environment and start the bot!** 