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

## 🐜 Implementation Progress

### ✅ Phase 1: Wallet Management & Core Infrastructure
- [x] **Implemented Wallet Creation**
  - [x] Solana wallet keypair generation
  - [x] Secure wallet storage with encryption
  - [x] Wallet backup and recovery mechanisms
  - [x] Added wallet naming/labeling system

- [x] **Implemented Capital Transfer**
  - [x] Basic SOL transfer between wallets
  - [x] Transaction confirmation
  - [x] Gas fee handling
  - [x] External wallet transfers

- [x] **Implemented Balance Checking**
  - [x] Connected to Solana RPC endpoints
  - [x] Added secure balance querying
  - [x] Implemented balance metrics and tracking

### 🔜 Phase 2: Trading & Profit Generation
- [ ] **Implement Market Data Collection**
- [ ] **Implement Trading Strategy**
- [ ] **Implement Profit Collection & Reinvestment**

### 🔜 Phase 3: Scaling & Optimization
- [ ] **Implement Princess Spawning Logic**
- [ ] **Implement Worker Performance Metrics**
- [ ] **Implement Cloud Deployment**

### 🔜 Phase 4: Interface & Monitoring Enhancements
- [ ] **Complete Dashboard Integration**

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

### Using the Wallet CLI

The AntBot includes a wallet CLI for managing wallets and transactions:

```bash
# Create a new wallet
python wallet_cli.py create --name "queen_main" --type queen

# List all wallets
python wallet_cli.py list

# Check wallet balance
python wallet_cli.py balance --name "queen_main"

# Transfer SOL between wallets
python wallet_cli.py transfer --from "queen_main" --to "worker_1" --amount 1.0

# Create a wallet backup
python wallet_cli.py backup --path ./my_backup.json

# Restore from backup
python wallet_cli.py restore --path ./my_backup.json
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
├── utils/               # Shared utilities
│   ├── wallet_manager.py # Solana wallet management
│   ├── config_loader.py # Configuration management
│   └── monitor.py       # System monitoring utilities
├── rust_core/           # High-performance trading engine
├── bindings/            # Python-Rust integration
├── config/              # Configuration files
│   ├── settings.yaml    # Main configuration
│   └── queen.yaml       # Queen-specific settings
├── wallets/             # Encrypted wallet storage
└── wallet_cli.py        # Wallet management CLI
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

- **Wallet Security**
  - Secure Solana keypair generation using cryptographically secure random number generation
  - AES-GCM encryption for wallet private keys with PBKDF2 key derivation
  - Isolated wallet storage with unique UUIDs for each wallet
  - Encrypted backup and restore functionality with secure key handling
  
- **Transaction Security**
  - Balance verification before transactions to prevent overdrafts
  - Transaction confirmation verification
  - Automatic error handling and retry logic
  
- **System Security**
  - Maximum failed attempts limit
  - Block duration for suspicious activity
  - Trade confirmation requirements for large transactions
  - Emergency shutdown mechanisms
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