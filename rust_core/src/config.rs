use std::fs;
use std::path::Path;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use anyhow::{Result, anyhow};
use log::{info, warn, error};
use serde_yaml::Value;
use solana_sdk::commitment_config;

/// Worker configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    pub max_trades_per_hour: u32,
    pub min_profit_threshold: f64,
    pub max_slippage: f64,
    pub max_trade_size: f64,
    pub min_liquidity: f64,
    pub max_hold_time: u64,
    pub target_trades_per_minute: u32,
    pub max_concurrent_trades: u32,
}

/// RPC configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcConfig {
    pub rpc_url: String,
    pub commitment: commitment_config::CommitmentConfig,
}

/// Network configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub rpc_endpoint: String,
    pub ws_endpoint: String,
    pub commitment: String,
    pub timeout: u32,
}

/// DEX configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DexConfig {
    pub preferred_dexes: Vec<String>,
    pub min_dex_volume: f64,
    pub max_price_impact: f64,
    pub min_market_cap: u64,
}

/// Risk management configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    pub max_position_size: f64,
    pub stop_loss: f64,
    pub trailing_stop: f64,
    pub max_daily_loss: f64,
    pub take_profit_percentage: f64,
}

/// Config struct containing all configuration components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub worker: WorkerConfig,
    pub network: NetworkConfig,
    pub dex: DexConfig,
    pub risk: RiskConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            worker: WorkerConfig {
                max_trades_per_hour: 10,
                min_profit_threshold: 0.01,
                max_slippage: 0.02,
                max_trade_size: 1.0,
                min_liquidity: 1000.0,
                max_hold_time: 30,
                target_trades_per_minute: 60,
                max_concurrent_trades: 5,
            },
            network: NetworkConfig {
                rpc_endpoint: "https://api.mainnet-beta.solana.com".to_string(),
                ws_endpoint: "wss://api.mainnet-beta.solana.com".to_string(),
                commitment: "confirmed".to_string(),
                timeout: 30,
            },
            dex: DexConfig {
                preferred_dexes: vec![
                    "jupiter".to_string(),
                    "orca".to_string(),
                    "raydium".to_string(),
                ],
                min_dex_volume: 10000.0,
                max_price_impact: 0.01,
                min_market_cap: 1000000,
            },
            risk: RiskConfig {
                max_position_size: 0.1,
                stop_loss: 0.05,
                trailing_stop: 0.02,
                max_daily_loss: 50.0,
                take_profit_percentage: 0.15,
            },
        }
    }
}

/// ConfigManager singleton for loading and accessing configuration
pub struct ConfigManager {
    config: Arc<RwLock<Config>>,
    config_path: String,
}

impl ConfigManager {
    /// Create a new ConfigManager with the specified path
    pub fn new(config_path: &str) -> Self {
        Self {
            config: Arc::new(RwLock::new(Config::default())),
            config_path: config_path.to_string(),
        }
    }

    /// Load configuration from file
    pub async fn load_config(&self) -> Result<()> {
        let config_path = Path::new(&self.config_path);
        if !config_path.exists() {
            warn!("Config file not found at {}, using defaults", self.config_path);
            return Ok(());
        }

        let config_str = fs::read_to_string(config_path)?;
        let yaml_value: Value = serde_yaml::from_str(&config_str)?;
        
        let mut config = Config::default();
        
        // Parse worker config
        if let Some(worker_value) = yaml_value.get("worker") {
            if let Ok(worker_config) = serde_yaml::from_value(worker_value.clone()) {
                config.worker = worker_config;
            }
        }
        
        // Parse network config
        if let Some(network_value) = yaml_value.get("network") {
            if let Ok(network_config) = serde_yaml::from_value(network_value.clone()) {
                config.network = network_config;
            }
        }
        
        // Parse dex config
        if let Some(dex_value) = yaml_value.get("dex") {
            if let Ok(dex_config) = serde_yaml::from_value(dex_value.clone()) {
                config.dex = dex_config;
            }
        }
        
        // Parse risk config
        if let Some(risk_value) = yaml_value.get("risk") {
            if let Ok(risk_config) = serde_yaml::from_value(risk_value.clone()) {
                config.risk = risk_config;
            }
        }
        
        // Update the config
        let mut writable_config = self.config.write().await;
        *writable_config = config;
        
        info!("Successfully loaded configuration from {}", self.config_path);
        Ok(())
    }
    
    /// Get worker configuration
    pub async fn get_worker_config(&self) -> WorkerConfig {
        self.config.read().await.worker.clone()
    }
    
    /// Get network configuration
    pub async fn get_network_config(&self) -> NetworkConfig {
        self.config.read().await.network.clone()
    }
    
    /// Get DEX configuration
    pub async fn get_dex_config(&self) -> DexConfig {
        self.config.read().await.dex.clone()
    }
    
    /// Get risk configuration
    pub async fn get_risk_config(&self) -> RiskConfig {
        self.config.read().await.risk.clone()
    }
    
    /// Get full configuration
    pub async fn get_config(&self) -> Config {
        self.config.read().await.clone()
    }
    
    /// Reload configuration from file
    pub async fn reload_config(&self) -> Result<()> {
        self.load_config().await
    }
}

// Create a global instance for easy access
lazy_static::lazy_static! {
    static ref CONFIG_MANAGER: ConfigManager = ConfigManager::new("config/settings.yaml");
}

/// Initialize the configuration system
pub async fn init_config() -> Result<()> {
    CONFIG_MANAGER.load_config().await
}

/// Get worker configuration
pub async fn get_worker_config() -> WorkerConfig {
    CONFIG_MANAGER.get_worker_config().await
}

/// Get network configuration
pub async fn get_network_config() -> NetworkConfig {
    CONFIG_MANAGER.get_network_config().await
}

/// Get DEX configuration
pub async fn get_dex_config() -> DexConfig {
    CONFIG_MANAGER.get_dex_config().await
}

/// Get risk configuration
pub async fn get_risk_config() -> RiskConfig {
    CONFIG_MANAGER.get_risk_config().await
}

/// Get full configuration
pub async fn get_config() -> Config {
    CONFIG_MANAGER.get_config().await
}

/// Reload configuration from file
pub async fn reload_config() -> Result<()> {
    CONFIG_MANAGER.reload_config().await
}

impl NetworkConfig {
    pub fn to_rpc_config(&self) -> RpcConfig {
        RpcConfig {
            rpc_url: self.rpc_endpoint.clone(),
            commitment: match self.commitment.as_str() {
                "confirmed" => commitment_config::CommitmentConfig::confirmed(),
                "finalized" => commitment_config::CommitmentConfig::finalized(),
                "processed" => commitment_config::CommitmentConfig::processed(),
                _ => commitment_config::CommitmentConfig::confirmed(),
            },
        }
    }
}

/// Get RPC configuration
pub fn get_rpc_config() -> Result<RpcConfig> {
    // This is a blocking version that can be used in sync contexts
    use tokio::runtime::Runtime;
    
    let rt = Runtime::new()?;
    let network_config = rt.block_on(get_network_config());
    
    Ok(network_config.to_rpc_config())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_default_config() {
        let config_manager = ConfigManager::new("nonexistent.yaml");
        
        // Default config should be used
        let worker_config = config_manager.get_worker_config().await;
        assert_eq!(worker_config.max_trades_per_hour, 10);
        assert_eq!(worker_config.min_profit_threshold, 0.01);
    }
    
    #[tokio::test]
    async fn test_load_config() {
        // For this test, we need a test config file
        let temp_dir = tempfile::tempdir().unwrap();
        let config_path = temp_dir.path().join("test_config.yaml");
        
        // Write a test config to the file
        let test_config = r#"
worker:
  max_trades_per_hour: 20
  min_profit_threshold: 0.05
        "#;
        
        fs::write(&config_path, test_config).unwrap();
        
        // Now load the config
        let config_manager = ConfigManager::new(config_path.to_str().unwrap());
        config_manager.load_config().await.unwrap();
        
        // Check that values were loaded correctly
        let worker_config = config_manager.get_worker_config().await;
        assert_eq!(worker_config.max_trades_per_hour, 20);
        assert_eq!(worker_config.min_profit_threshold, 0.05);
    }
} 