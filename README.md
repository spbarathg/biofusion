# ANTBOT - LEAN MATHEMATICAL TRADING SYSTEM

A fast, transparent, and reliable trading bot for Solana memecoins using pure mathematical decision-making.

## 🎯 **Architectural Philosophy: Pure Lean Mathematical Core**

Antbot follows the **"Singularity of Purpose"** principle with a clean three-stage decision pipeline:

```
Market Opportunity → SURVIVE → QUANTIFY EDGE → BET OPTIMALLY → Execute Trade
                     (WCCA)    (Naive Bayes)   (Kelly Criterion)
```

### **Why Lean Over Complex ML?**

✅ **Transparency**: Every decision traceable to mathematical formula  
✅ **Speed**: Millisecond-level decision latency  
✅ **Reliability**: Minimal failure points and dependencies  
✅ **Maintainability**: Clean, auditable codebase  

❌ **Complex ML Rejected**: Black box models, multi-process overhead, debugging challenges

---

## 🚀 **Quick Start**

### **Production Trading**
```bash
python entry_points/run_bot.py --mode production --strategy simplified
```

### **Simulation Mode**
```bash
python entry_points/run_bot.py --mode simulation --strategy simplified
```

### **High Availability Colony**
```bash
python entry_points/run_bot.py --mode production --strategy colony
```

### **Configuration Setup**

1. **For Simple Trading** (recommended for beginners):
   ```bash
   cp config/simplified.env.template .env.production
   ```

2. **For Full System** (HA, monitoring, advanced features):
   ```bash
   cp config/env.template .env.production
   ```

3. **Edit your .env.production file** with your API keys and settings

---

## 📁 **Core Architecture**

### **Single Sources of Truth**
- **Configuration**: `worker_ant_v1/core/unified_config.py` - All system settings
- **Trading Engine**: `worker_ant_v1/core/unified_trading_engine.py` - Order execution
- **Mathematical Core**: `worker_ant_v1/trading/simplified_trading_bot.py` - Decision pipeline
- **Risk Filter**: `worker_ant_v1/trading/devils_advocate_synapse.py` - WCCA survival analysis

### **Entry Point Doctrine**
- **ONLY** executable entry point: `entry_points/run_bot.py`
- All operations go through the unified launcher
- No direct execution of modules in `worker_ant_v1/`

### **Archived Components**
- **Legacy ML**: `worker_ant_v1/trading/_legacy_ml/` - Complex ML components (archived for reference)

---

## 🛡️ **Safety Features**

- **WCCA Risk Analysis**: Pre-mortem risk assessment before every trade
- **Kill Switch**: Automatic emergency stops on anomalous conditions
- **Vault System**: Automatic profit protection and secure storage
- **Configuration Validation**: Production-grade settings verification

---

## 📊 **Mathematical Decision Engine**

### **Stage 1: Survival Filter (WCCA)**
Risk-Adjusted Expected Loss = P(Loss) × |Position_Size|
- Threshold: 0.1 SOL maximum risk per trade

### **Stage 2: Win Rate Engine (Naive Bayes)**  
P(Win | Signals) ∝ P(Win) × Π P(Signal_i | Win)
- Minimum 60% confidence threshold

### **Stage 3: Growth Maximizer (Kelly Criterion)**
f* = p - ((1 - p) / b)
- 25% safety fraction applied

--- 