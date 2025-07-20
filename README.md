# Antbot - Advanced Solana Memecoin Trading Swarm

A state-of-the-art automated trading system for Solana blockchain, featuring advanced AI architectures, multi-agent swarm intelligence, and sophisticated risk management for memecoin trading.

## 🚀 Core Capabilities

### **Multi-Agent Swarm Intelligence**
- **10-Wallet Neural Swarm**: Coordinated trading across multiple wallets with genetic evolution
- **Swarm Decision Engine**: Consensus-based decision making with AI + On-Chain + Social validation
- **Squad Formation**: Dynamic squad creation for high-conviction opportunities
- **Genetic Evolution**: Nightly evolution system optimizing wallet performance

### **Advanced AI Architectures**
- **Oracle Ant**: Transformer-based time series prediction for price, volume, and holder forecasting
- **Hunter Ant**: Reinforcement Learning agents learning optimal trading policies through experience
- **Network Ant**: Graph Neural Networks detecting smart money movements and manipulation patterns
- **SentimentFirst AI**: Real-time sentiment analysis from social media and on-chain data
- **Battle Pattern Intelligence**: High-speed pattern recognition for wallet behavior analysis

### **Comprehensive Trading Intelligence**
- **Market Scanner**: Real-time opportunity detection across Solana DEXs
- **Rug Detection**: Advanced rug pull detection with multiple risk factors
- **Stealth Operations**: Anti-detection mechanisms for optimal execution
- **Caller Intelligence**: Analysis of token callers and their success patterns
- **Technical Analysis**: RSI, MACD, Bollinger Bands, and custom indicators

### **Risk Management & Safety**
- **Enhanced Kill Switch**: Multi-level emergency shutdown systems
- **Position Sizing**: Dynamic position sizing based on confidence and risk
- **Vault System**: Secure profit protection and compounding mechanisms
- **Stop Loss**: Intelligent stop-loss placement and management
- **Capital Protection**: Multiple layers of capital preservation

### **Infrastructure & Monitoring**
- **Jupiter DEX Integration**: Direct integration with Jupiter aggregator for best execution
- **Real-time Monitoring**: Comprehensive performance tracking and alerting
- **Production Monitoring**: System health, uptime, and performance metrics
- **Live Broadcasting**: Real-time trading activity broadcasting
- **Grafana Dashboards**: Advanced visualization and analytics

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Solana wallet with SOL for trading
- API keys for Helius, Birdeye, Jupiter, and DexScreener
- GPU recommended for ML models (optional)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd antbotNew

# Install dependencies
pip install -r config/requirements.txt

# Install ML dependencies (optional, for advanced AI features)
pip install -r config/ml_requirements.txt

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

# Colony Commander (advanced swarm management)
python entry_points/colony_commander.py
```

## 🏗️ System Architecture

### **Core Trading Systems**
- **Unified Trading Engine**: Handles trade execution and position management
- **Market Scanner**: Real-time opportunity detection across Solana DEXs
- **Jupiter DEX Integration**: Best execution routing and liquidity aggregation
- **Vault Wallet System**: Secure profit protection and compounding

### **Intelligence Layer**
- **Swarm Decision Engine**: Neural command center coordinating all trading decisions
- **SentimentFirst AI**: Real-time sentiment analysis from multiple sources
- **Battle Pattern Intelligence**: High-speed pattern recognition and wallet analysis
- **Enhanced Rug Detector**: Advanced rug pull detection with multiple risk factors

### **AI & Machine Learning**
- **Oracle Ant**: Transformer-based time series prediction
- **Hunter Ant**: Reinforcement Learning trading agents
- **Network Ant**: Graph Neural Networks for on-chain intelligence
- **Neural Command Center**: Centralized AI decision making

### **Safety & Risk Management**
- **Enhanced Kill Switch**: Multi-level emergency shutdown systems
- **Enterprise Error Handling**: Comprehensive error recovery and logging
- **Alert System**: Real-time notifications and monitoring
- **Production Monitoring**: System health and performance tracking

### **Evolution & Optimization**
- **Nightly Evolution System**: Genetic optimization of wallet performance
- **Squad Manager**: Dynamic squad formation for high-conviction trades
- **Stealth Operations**: Anti-detection mechanisms and optimal execution
- **Caller Intelligence**: Analysis of token callers and success patterns

## ⚙️ Configuration

### **Trading Parameters**
- `INITIAL_CAPITAL`: Starting capital in USD
- `POSITION_SIZE_PERCENT`: Percentage of capital per position
- `MAX_POSITION_SIZE_SOL`: Maximum SOL per position
- `MIN_PROFIT_TARGET`: Minimum profit target percentage
- `MAX_LOSS_PERCENT`: Maximum loss percentage
- `MAX_HOLD_TIME_HOURS`: Maximum position hold time

### **AI & ML Configuration**
- `ML_MODELS_ENABLED`: Enable advanced AI architectures
- `ORACLE_ANT_WEIGHT`: Weight for Oracle Ant predictions
- `HUNTER_ANT_WEIGHT`: Weight for Hunter Ant predictions
- `NETWORK_ANT_WEIGHT`: Weight for Network Ant predictions
- `CONSENSUS_THRESHOLD`: Minimum consensus for trade execution

### **Swarm Configuration**
- `SWARM_SIZE`: Number of wallets in the swarm (default: 10)
- `EVOLUTION_ENABLED`: Enable genetic evolution system
- `SQUAD_FORMATION_ENABLED`: Enable dynamic squad formation
- `STEALTH_MODE_ENABLED`: Enable anti-detection mechanisms

## 🛡️ Safety Features

- **Enhanced Kill Switch**: Multi-level emergency shutdown on significant losses
- **Position Limits**: Maximum concurrent positions and position sizing
- **Capital Protection**: Vault system secures profits and prevents capital erosion
- **Intelligent Stop Loss**: Dynamic stop-loss placement based on market conditions
- **Time Limits**: Maximum position hold times to prevent overholding
- **Risk Assessment**: Multi-factor risk analysis before trade execution
- **Consensus Validation**: AI + On-Chain + Social must agree for trade execution

## 📊 Monitoring & Analytics

### **Real-time Performance Tracking**
- Trading performance and profit/loss across all wallets
- Active positions, win rate, and position sizing
- System health, uptime, and performance metrics
- Wallet performance and genetic evolution progress

### **Advanced Analytics**
- **Grafana Dashboards**: Comprehensive visualization of all metrics
- **Live Broadcasting**: Real-time trading activity streaming
- **Production Monitoring**: System health and alerting
- **AI Model Performance**: Oracle, Hunter, and Network Ant metrics
- **Swarm Intelligence**: Consensus scores and decision breakdowns

### **Alerting & Notifications**
- Real-time alerts for significant events
- Performance threshold notifications
- System health monitoring
- Risk assessment alerts

## ⚠️ Risk Warnings

- **Extreme Volatility**: Memecoin trading is extremely volatile and unpredictable
- **Capital Loss**: You can lose your entire investment, including initial capital
- **No Guarantees**: Past performance doesn't guarantee future results
- **Market Conditions**: Performance heavily depends on market conditions and timing
- **Technical Risks**: Software bugs, network issues, API failures, and MEV attacks
- **AI Limitations**: Machine learning models may not perform as expected in all market conditions
- **Regulatory Risks**: Trading regulations may change and affect bot operation

## 📋 Legal Considerations

- Ensure compliance with local trading regulations and tax laws
- Trading profits may be taxable - consult with a tax professional
- Respect API terms and rate limits for all services
- You are responsible for all trading decisions and their consequences
- This software is for educational and research purposes

## 🆘 Support & Troubleshooting

### **Common Issues**
- Check configuration and API keys are correctly set
- Review logs for detailed error messages and debugging
- Test in simulation mode before running with real capital
- Ensure sufficient SOL for gas fees and transaction costs
- Verify all dependencies are properly installed

### **Performance Optimization**
- Use GPU acceleration for ML models when available
- Monitor system resources and performance metrics
- Adjust trading parameters based on market conditions
- Regularly backup configuration and model checkpoints

### **Getting Help**
- Review the configuration examples and documentation
- Check system logs for detailed error information
- Test with small amounts before scaling up
- Monitor the bot's performance and adjust parameters as needed

---

**⚠️ Disclaimer**: This software is for educational and research purposes. Trading cryptocurrencies involves substantial risk of loss. Use at your own risk and never invest more than you can afford to lose. 