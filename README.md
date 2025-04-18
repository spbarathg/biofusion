# AntBot: Solana Trading Bot

AntBot is a multi-agent trading bot system for Solana built with a hybrid Python-Rust architecture. The system uses a colony structure with Queen and Worker agents to manage capital and execute trades.

## Architecture

AntBot uses a simplified multi-layered architecture:

- **Python Layer**: Manages capital allocation, agent coordination, and monitoring
- **Rust Core**: Powers the high-performance trading engine (with Python fallback when Rust isn't available)
- **FFI Bindings**: Connects Python and Rust layers through native bindings

## Directory Structure

```
antbotNew/
├── config/                     # Configuration files
│   └── settings.yaml           # Main settings file
├── src/                        # Main source code
│   ├── core/                   # Core functionality
│   │   ├── agents/             # Agent implementations
│   │   │   ├── queen.py        # Queen agent 
│   │   │   └── worker.py       # Worker agent
│   │   ├── capital/            # Capital management
│   │   │   └── capital_manager.py
│   │   └── wallet/             # Wallet management
│   │       └── wallet_manager.py
│   ├── dashboard/              # Dashboard components
│   │   ├── app.py              # Dashboard application
│   │   └── components/         # UI components
│   ├── bindings/               # Rust-Python bindings
│   │   └── worker_bridge.py    # Bridge to the Rust engine
│   └── utils/                  # Utility functions
│       ├── logging/            # Logging utilities
│       │   └── logger.py
│       └── monitoring/         # Monitoring tools
├── rust_core/                  # Rust core implementation
│   ├── src/
│   │   ├── lib.rs
│   │   ├── worker.rs
│   │   └── ...
│   ├── Cargo.toml
│   └── build.rs
├── data/                       # Data directory
│   ├── wallets/                # Wallet storage (encrypted)
│   └── backups/                # Wallet backups
├── logs/                       # Log files
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── README.md                   # Project documentation
```

## Installation

### Prerequisites

- Python 3.8+ 
- Rust 1.5X+ (optional - Python fallback available)
- Solana CLI tools

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/antbot.git
   cd antbot
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install package in development mode
   ```

3. **Build the Rust core (optional):**
   ```bash
   cd rust_core
   cargo build --release
   cd ..
   ```

4. **Set up configuration:**
   Copy and edit the example config file
   ```bash
   cp config/settings.yaml.example config/settings.yaml
   ```

## Usage

### Running the Bot

1. **Start the Queen:**
   ```bash
   python -m src.core.agents.queen
   ```

2. **Start the Dashboard:**
   ```bash
   python -m src.dashboard.app
   ```

   Or with Streamlit:
   ```bash
   streamlit run src/dashboard/app.py
   ```

## Bot Architecture

### Agent Types

- **Queen:** Capital manager, orchestrates the colony and allocates funds
- **Worker:** Execution agent, performs trades on specific markets

### Trading Strategy

The bot supports a basic "find and execute" strategy that can be extended further:

- Market making
- Arbitrage between DEXes
- Liquidity provision

## Configuration

Configuration is stored in YAML files in the `config/` directory:

- `settings.yaml`: Main configuration file

Key configuration sections:

```yaml
# Colony Management (handles agent spawning and coordination)
colony:
  initial_capital: 10.0
  min_workers: 3
  max_workers: 10

# Capital Management (handles profits and reinvestment)
capital:
  savings_ratio: 0.90
  reinvestment_ratio: 0.80
  compound_frequency: 24

# Worker Configuration (handles trading parameters)
worker:
  max_trades_per_hour: 10
  min_profit_threshold: 0.01
  max_slippage: 0.02
```

## Monitoring

AntBot includes a real-time monitoring dashboard:

- Web interface with Streamlit
- Performance metrics
- Trading history
- Agent status

## Security

- All wallet private keys are encrypted at rest
- Regular wallet backups are recommended

## Development

### Testing

```bash
python -m pytest tests/
```

### Installing Development Dependencies

```bash
pip install -e ".[dev]"
```

### Installing Dashboard Dependencies

```bash
pip install -e ".[dashboard]"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
