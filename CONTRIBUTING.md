# ANTBOT DEVELOPMENT MANIFESTO
## Architectural Principles for 10/10 Code Quality

---

## **Core Philosophy: Singularity of Purpose**

The Antbot codebase is governed by an unwavering principle: **One logical function, one module. No duplicates, no alternatives.**

This manifesto establishes the inviolable architectural rules that maintain our 10/10 production-ready codebase. Every contributor must understand and strictly adhere to these principles.

---

## **Canonical Module Hierarchy**

### **🎯 Single Source of Truth Principle**

Each functional domain has **one and only one** canonical module. Creating duplicates is strictly forbidden.

#### **Core Systems** (`worker_ant_v1/core/`)
- **Configuration**: `unified_config.py` - All system configuration
- **Database**: `database.py` - All database operations  
- **Wallet Management**: `wallet_manager.py` - All wallet operations
- **Trading Engine**: `unified_trading_engine.py` - Core trading logic
- **API Keys**: `configure_api_keys.py` - **CANONICAL** API configuration

#### **Intelligence Systems** (`worker_ant_v1/intelligence/`)
- **Sentiment Analysis**: `sentiment_first_ai.py` - **CANONICAL** sentiment engine
- **Technical Analysis**: `technical_analyzer.py` - All technical indicators
- **Rug Detection**: `enhanced_rug_detector.py` - Risk assessment
- **Token Intelligence**: `token_intelligence_system.py` - Token analysis

#### **Utilities** (`worker_ant_v1/utils/`)
- **Logging**: `logger.py` - **CANONICAL** system-wide logging
- **Market Data**: `market_data_fetcher.py` - All market data operations
- **Constants**: `constants.py` - All system constants
- **Codebase Management**: `codebase_cleaner.py` - **CANONICAL** cleanup utility

#### **Trading Operations** (`worker_ant_v1/trading/`)
- **Market Scanning**: `market_scanner.py` - Opportunity detection
- **Order Execution**: `order_buyer.py`, `order_seller.py` - Trade execution
- **Strategy Classes**: `main.py`, `hyper_compound_squad.py` - **NON-EXECUTABLE** classes only

---

## **Entry Point Doctrine**

### **🚪 Single Entry Point Rule**

The `entry_points/` directory is the **ONLY** place for executable scripts. No exceptions.

- **Master Entry Point**: `entry_points/run_bot.py` - **SOLE** executable for all operations
- **Forbidden**: Any `if __name__ == "__main__":` blocks outside `entry_points/`
- **Forbidden**: Direct execution of files in `worker_ant_v1/`

### **Command Structure**
```bash
# ✅ CORRECT - All operations through master entry point
python entry_points/run_bot.py --mode production --strategy hyper_intelligent
python entry_points/run_bot.py --mode simulation --strategy hyper_compound

# ❌ FORBIDDEN - Direct execution of modules
python worker_ant_v1/trading/main.py
python worker_ant_v1/trading/hyper_compound_squad.py
```

---

## **Import Discipline**

### **🔗 Canonical Import Paths**

Always import from the canonical module. Creating or using duplicate imports is forbidden.

#### **✅ CANONICAL IMPORTS**
```python
# Logging - ALWAYS use utils logger
from worker_ant_v1.utils.logger import setup_logger, get_logger

# Sentiment Analysis - ALWAYS use intelligence module  
from worker_ant_v1.intelligence.sentiment_first_ai import SentimentFirstAI

# Configuration - ALWAYS use core module
from worker_ant_v1.core.configure_api_keys import setup_api_keys
```

#### **❌ FORBIDDEN IMPORTS**
```python
# These paths are FORBIDDEN - they represent duplicates that must be eliminated
from worker_ant_v1.trading.logger import *  # NO! Use utils.logger
from worker_ant_v1.trading.sentiment_first_ai import *  # NO! Use intelligence.sentiment_first_ai
from worker_ant_v1.trading.configure_api_keys import *  # NO! Use core.configure_api_keys
```

---

## **Code Organization Rules**

### **📁 Separation of Concerns**

Different types of code must reside in their designated domains:

- **Application Code**: `worker_ant_v1/`
- **Test Code**: `tests/` (root level)
- **Entry Points**: `entry_points/`
- **Configuration**: `config/`
- **Scripts**: `scripts/` (utilities only, not main application)

### **❌ FORBIDDEN**
- Test files in application directories
- Application logic in scripts directory
- Multiple entry points for the same functionality

---

## **Quality Gates**

### **🔒 Mandatory Checks Before Merging**

1. **No Duplicates**: `grep -r "duplicate_filename" .` returns zero results
2. **No Forbidden Imports**: All imports use canonical paths
3. **Single Entry**: Only `entry_points/run_bot.py` can execute application logic
4. **CI Pipeline**: All tests pass in `.github/workflows/ci.yml`
5. **Code Style**: All code follows established patterns

### **🚨 Automatic Rejection Criteria**

Pull requests will be automatically rejected if they:
- Create duplicate modules
- Add `if __name__ == "__main__":` outside `entry_points/`
- Import from non-canonical paths
- Mix test code with application code

---

## **Evolution Protocol**

### **📈 How to Modify the Architecture**

1. **Propose Changes**: Architectural changes require discussion
2. **Single Migration**: All changes to canonical structure must be atomic
3. **Update This Manifesto**: Any architectural changes must update this document
4. **Zero Tolerance**: No exceptions to these rules under any circumstances

---

## **Enforcement**

### **👮 Responsibility**

Every developer is responsible for:
- Understanding these principles before contributing
- Enforcing these rules in code review
- Reporting violations immediately
- Maintaining the 10/10 standard

### **🛡️ Protection Measures**

- Pre-commit hooks validate canonical imports
- CI pipelines enforce single entry point rule
- Code review checklist includes manifesto compliance
- Architectural debt is treated as a critical bug

---

## **Conclusion**

This manifesto ensures that the Antbot codebase remains a pristine, deterministic, 10/10 production system. These rules are not suggestions—they are the foundation of our architectural integrity.

**Violation of these principles is not acceptable under any circumstances.**

---

*"In code, as in nature, survival belongs to the most organized."*

**Document Version**: 1.0  
**Last Updated**: Operation Citadel - July 2025  
**Status**: IMMUTABLE ARCHITECTURAL LAW 