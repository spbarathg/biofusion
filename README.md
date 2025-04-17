# AntBot: Solana Trading Bot

AntBot is a multi-agent trading bot system for Solana built with a hybrid Python-Rust architecture. The system uses a colony structure with Queen, Princess, and Worker agents to manage capital and execute trades.

## Architecture

AntBot uses a multi-layered architecture:

- **Python Layer**: Manages higher-level capital management, agent coordination, and monitoring
- **Rust Core**: Powers the high-performance trading engine and execution components
- **FFI Bindings**: Connects Python and Rust layers through native bindings

## Directory Structure

```
antbotNew/
├── config/                     # Configuration files
│   ├── settings.yaml           # Main settings file
│   └── queen.yaml              # Queen-specific settings
├── src/                        # Main source code
│   ├── core/                   # Core functionality
│   │   ├── wallet_manager.py   # Wallet management
│   │   └── config_loader.py    # Configuration loading
│   ├── logging/                # Logging configuration
│   │   ├── logger.py
│   │   └── log_config.py
│   ├── models/                 # Bot models
│   │   ├── queen.py            # Queen agent
│   │   ├── princess.py         # Princess agent
│   │   ├── worker.py           # Worker agent
│   │   ├── drone.py            # Drone agent
│   │   └── capital_manager.py  # Capital management
│   ├── dashboard/              # Dashboard components
│   │   ├── dashboard.py
│   │   └── run_dashboard.py
│   ├── utils/                  # Utility functions
│   │   └── monitor.py
│   └── bindings/               # Rust-Python bindings
│       └── worker_bridge.py
├── rust_core/                  # Rust core implementation
│   ├── src/
│   │   ├── lib.rs
│   │   ├── main.rs
│   │   ├── worker.rs
│   │   ├── worker_ant.rs
│   │   ├── wallet.rs
│   │   ├── tx_executor.rs
│   │   ├── pathfinder.rs 
│   │   ├── dex_client.rs
│   │   └── config.rs
│   ├── Cargo.toml
│   └── Cargo.lock
├── scripts/                    # Utility scripts
│   └── wallet_cli.py           # Wallet management CLI
├── data/                       # Data directory
│   ├── wallets/                # Wallet storage (encrypted)
│   └── backups/                # Wallet backups
├── logs/                       # Log files
├── systemd/                    # SystemD service files
│   ├── antbot.service
│   ├── antbot-dashboard.service
│   └── README.md
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Installation

### Prerequisites

- Python 3.8+ 
- Rust 1.5X+
- Solana CLI tools
- Node.js/npm (for dashboard)

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

3. **Build the Rust core:**
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

5. **Prepare the environment:**
   Either create a `.env` file with required variables or set them in your environment.

## Quick Build Guide

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install package in development mode
   ```

2. **Build Rust components:**
   ```bash
   cd rust_core
   cargo build --release
   cd ..
   ```

3. **Test wallet CLI:**
   ```bash
   python scripts/wallet_cli.py --help
   ```

4. **Start the queen:**
   ```bash
   python -m src.models.queen --state
   ```

## Usage

### Wallet Management

Use the wallet CLI to manage wallets:

```bash
python scripts/wallet_cli.py create --name "queen_wallet" --type queen
python scripts/wallet_cli.py list
python scripts/wallet_cli.py balance --id <wallet_id>
python scripts/wallet_cli.py backup
```

### Running the Bot

1. **Start the Queen:**
   ```bash
   python -m src.models.queen
   ```

2. **Start the Dashboard:**
   ```bash
   python src/dashboard/run_dashboard.py
   ```

3. **Deploy as a service:**
   ```bash
   sudo cp systemd/antbot.service /etc/systemd/system/
   sudo systemctl enable antbot
   sudo systemctl start antbot
   ```

## Bot Architecture

### Agent Types

- **Queen:** Capital manager, orchestrates the colony and allocates funds
- **Princess:** Regional capital manager, spawned by the Queen
- **Worker:** Execution agent, performs trades on specific markets
- **Drone:** Utility agent, performs support tasks like rebalancing

### Trading Strategy

The bot supports multiple configurable strategies:

- Arbitrage between DEXes
- Market making
- Liquidity farming
- Custom strategies via plugin system

## Configuration

Configuration is stored in YAML files in the `config/` directory:

- `settings.yaml`: Main configuration file
- `queen.yaml`: Queen-specific settings

See the documentation in `docs/` for detailed configuration options.

## Monitoring

AntBot includes a real-time monitoring dashboard:

- Web interface on port 8501 (default)
- Performance metrics
- Trading history
- Agent status

## Security

- All wallet private keys are encrypted at rest
- API keys and secrets should be stored in `.env` file (not included in repo)
- Regular wallet backups are recommended

## Development

### Testing

```bash
python -m pytest tests/
```

### Adding New Components

Follow the existing patterns for structure and imports to maintain consistency.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
