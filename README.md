# 🐜 Ant Bot: Production-Grade Solana Memecoin Colony Trading Bot

A hybrid Rust + Python trading bot that emulates ant colony behavior for high-frequency memecoin trading on Solana. The bot implements a sophisticated colony-based trading system that combines high-frequency trading strategies with smart capital management and compounding mechanisms.

## 🧠 Core Concept

> "**Compound Interest + Ant Colony = Exponential Capital Multiplication**"  
> A = P(1 + R)^N  

The bot implements a sophisticated colony-based trading system that:
- Uses high-frequency trading strategies optimized for Solana memecoins
- Implements smart wallet hierarchies for capital management
- Employs compounding mechanisms for exponential growth
- Scales operations across multiple VPS instances
- Implements advanced risk management and monitoring

## 🏗️ Architecture

### Hybrid Stack
- **Rust Core**: High-performance trading engine, pathfinding, and transaction execution
  - Optimized for low-latency operations
  - Handles critical trading paths
  - Manages transaction execution
- **Python Layer**: Capital management, wallet orchestration, and colony logic
  - Queen: Colony management and capital allocation
  - Princess: Growth-focused trading accounts
  - Workers: High-frequency trading agents
  - Drones: Capital splitting and distribution

### Colony Structure
- **Queen Wallet**: 
  - Manages founding capital
  - Controls colony expansion
  - Oversees profit distribution
  - Handles emergency reserves
- **Princess Wallets**: 
  - Growth-focused trading accounts
  - Minimum 5% growth rate target
  - 20% maximum capital allocation
  - 80% reinvestment ratio
- **Worker Ants**: 
  - High-frequency trading agents
  - Maximum 10 trades per hour
  - 1% minimum profit threshold
  - 2% maximum slippage tolerance
- **Savings Vault**: 
  - Capital preservation
  - 90% profit allocation
  - Emergency reserve management
  - Compound interest mechanism

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Rust 1.70+
- Solana CLI tools
- Access to Solana RPC node
- VPS provider account (DigitalOcean recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/antbot.git
cd antbot
```

2. Install Python dependencies:
```bash
cd python_bot
pip install -r requirements.txt
```

3. Build Rust core:
```bash
cd rust_core
cargo build --release
```

4. Initialize colony:
```bash
python3 python_bot/queen.py --init-capital 10
```

5. Start the dashboard:
```bash
python python_bot/run_dashboard.py
```

## 📁 Project Structure

```
ant_bot/
├── python_bot/          # Python-based colony management
│   ├── queen.py         # Colony management and coordination
│   ├── princess.py      # Growth-focused trading accounts
│   ├── worker.py        # High-frequency trading agents
│   ├── drone.py         # Capital splitting and distribution
│   ├── capital_manager.py # Capital allocation and management
│   ├── deploy.py        # VPS deployment and scaling
│   ├── dashboard.py     # Streamlit dashboard interface
│   └── run_dashboard.py # Dashboard launcher script
├── rust_core/           # High-performance trading engine
├── bindings/            # Python-Rust integration
├── utils/              # Shared utilities
├── config/             # Configuration files
│   ├── settings.yaml   # Main configuration
│   └── queen.yaml      # Queen-specific settings
└── tests/              # Test suite
```

## 📊 Dashboard

The Ant Bot includes a minimalistic, dark-themed Streamlit dashboard for monitoring and controlling your trading system:

### Features
- 🔒 Password-protected access
- 📊 Real-time colony metrics
- 📈 Performance visualizations
- 🛠️ Colony controls
- ⚙️ Configuration management

### Running the Dashboard
```bash
# Using the provided script
python python_bot/run_dashboard.py

# Or directly with Streamlit
streamlit run python_bot/dashboard.py
```

### Default Password
The default password is `antbot`. It's recommended to change this by setting the `ANTBOT_PASSWORD` environment variable:

```bash
# On Windows
set ANTBOT_PASSWORD=your_secure_password

# On Linux/Mac
export ANTBOT_PASSWORD=your_secure_password
```

### Security Notes
- The dashboard is designed for private use only
- Always use a strong password
- Consider running the dashboard only on your local machine or a private VPS
- If deploying to a VPS, use HTTPS and a reverse proxy like Nginx

## 🔧 Configuration

The bot is highly configurable through `config/settings.yaml`. Key configuration sections include:

### Colony Management
- Initial capital allocation
- Princess spawning thresholds
- Worker limits per VPS
- Health check intervals

### Capital Management
- Savings ratio (90% default)
- Reinvestment strategies
- Compound frequency
- Emergency reserves

### Trading Parameters
- Maximum trades per hour
- Profit thresholds
- Slippage limits
- Position sizing
- Hold time limits

### Risk Management
- Position size limits
- Stop-loss settings
- Trailing stops
- Daily loss limits
- Take-profit targets

### Network Settings
- RPC endpoints
- WebSocket connections
- Timeout configurations
- DEX preferences

### Monitoring
- Log levels and rotation
- Health check intervals
- Performance metrics
- Alert thresholds

## 📊 Performance Metrics

- Trading speed: <5ms per trade
- Target profit: 10-20% per trade
- Hold time: 10-30 seconds
- Capital preservation: 90-95%
- Maximum concurrent trades: 5
- Target trades per minute: 60

## 🔒 Security Features

- Maximum failed attempts limit
- Block duration for suspicious activity
- Trade confirmation requirements
- Emergency shutdown mechanisms
- Secure wallet management
- VPS isolation
- Password-protected dashboard

## 🚨 Risk Warning

This bot is experimental software. Trading cryptocurrencies carries significant risk:
- High volatility in memecoin markets
- Potential for rapid capital loss
- Smart contract risks
- Network congestion issues
- Market manipulation risks

Use at your own discretion and never invest more than you can afford to lose.

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our repository.

## 📞 Support

For support, please open an issue in our GitHub repository or join our community Discord channel. 