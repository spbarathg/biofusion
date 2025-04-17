# AntBot Source Code Structure

This directory contains the main source code for the AntBot Solana trading bot. The code is organized into the following modules:

## Core Architecture

- **core/**: Core functionality and fundamental components
  - `wallet_manager.py`: Wallet and key management
  - `config_loader.py`: Configuration loading and management
  - `paths.py`: Path handling utilities
  
- **models/**: Bot agent models
  - `queen.py`: Queen agent (capital manager and orchestrator)
  - `princess.py`: Princess agent (regional capital manager)
  - `worker.py`: Worker agent (execution agent)
  - `drone.py`: Drone agent (utility tasks)
  - `capital_manager.py`: Capital allocation and management
  - `deploy.py`: Deployment utilities

- **dashboard/**: Dashboard and monitoring components
  - `dashboard.py`: Main Streamlit dashboard app
  - `run_dashboard.py`: Dashboard launcher script

- **logging/**: Logging configuration and utilities
  - `logger.py`: Centralized logging configuration
  - `log_config.py`: Log format and configuration
  
- **utils/**: General utility functions
  - `monitor.py`: System monitoring tools
  
- **bindings/**: Rust-Python bindings
  - Contains bridge code for interfacing with the Rust core

## Module Dependencies

The module dependencies flow generally as follows:

core <- models <- dashboard
 ^      ^
 |      |
 +------+-- utils
 |      |
 +------+-- logging
        |
        +-- bindings 