# LEGACY ML COMPONENTS - ARCHIVED
## Sunset Date: Current Refactoring Cycle

This directory contains **archived ML components** that were removed from the active Antbot codebase during the architectural consolidation to a **lean mathematical core**.

## Archived Components

### `process_pool_manager.py` (763+ lines)
- **Purpose**: Multi-process ML inference system for CPU-intensive models
- **Why Archived**: Added unacceptable latency and complexity for live trading
- **Replacement**: Direct mathematical calculations in main async loop

### `hunter_ant.py` (485+ lines) 
- **Purpose**: PPO reinforcement learning trading agent
- **Why Archived**: Black box decision making incompatible with transparency requirements
- **Replacement**: Auditable mathematical pipeline (WCCA → Naive Bayes → Kelly Criterion)

## Architectural Decision

**Chosen Path**: Pure lean mathematical core for maximum:
- **Transparency**: Every decision traceable to mathematical formula
- **Speed**: Millisecond-level decision latency
- **Reliability**: Minimal failure points and dependencies
- **Maintainability**: Clean, auditable codebase

**Rejected Path**: Hybrid ML system due to:
- Operational complexity and latency
- Debugging and interpretability challenges  
- Violation of "Singularity of Purpose" principle

## Preservation Purpose

These components are preserved for:
- **Intellectual Property**: Future advanced system development
- **Reference**: Understanding previous architectural approaches
- **Research**: Potential offline analysis and backtesting

## Usage Warning

⚠️ **DO NOT** reintroduce these components without addressing the fundamental architectural concerns that led to their removal.

---

*For current trading functionality, use `worker_ant_v1/trading/simplified_trading_bot.py` and `worker_ant_v1/core/unified_trading_engine.py`* 