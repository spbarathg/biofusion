use std::sync::Arc;
use tokio::sync::Mutex;
use log::{info, error};
use anyhow::Result;
use solana_sdk::keypair::Keypair;

mod worker_ant;
mod pathfinder;
mod dex_client;
mod tx_executor;
mod wallet;
mod config;

use worker_ant::{WorkerAnt, WorkerConfig};
use pathfinder::PathFinder;
use dex_client::DexClient;
use tx_executor::TxExecutor;
use wallet::WalletManager;
use config::{init_config, get_worker_config, get_network_config};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    info!("Starting Ant Bot Core...");

    // Initialize configuration
    info!("Loading configuration...");
    init_config().await?;
    
    // Get configuration
    let worker_config = get_worker_config().await;
    let network_config = get_network_config().await;

    // Initialize components with configuration
    let rpc_client = Arc::new(solana_client::rpc_client::RpcClient::new(
        network_config.rpc_endpoint
    ));
    
    let wallet_manager = Arc::new(Mutex::new(
        WalletManager::new(rpc_client.clone(), "wallets".to_string())?
    ));
    
    let dex_client = Arc::new(DexClient::new()?);
    let pathfinder = Arc::new(PathFinder::new());
    let tx_executor = Arc::new(TxExecutor::new()?);

    // Create worker ant
    let worker_id = std::env::var("WORKER_ID").unwrap_or_else(|_| "worker_1".to_string());
    
    info!("Initializing Worker Ant {}...", worker_id);
    let worker = WorkerAnt::new(
        worker_id,
        worker_config,
        dex_client,
        tx_executor,
        Keypair::new(),
    );

    // Start trading loop
    info!("Worker ant initialized, starting trading loop...");
    worker.start().await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_worker_initialization() {
        let worker_config = WorkerConfig {
            max_trades_per_hour: 10,
            min_profit_threshold: 0.01,
            max_slippage: 0.02,
            max_trade_size: 1.0,
            min_liquidity: 1000.0,
            max_hold_time: 3600,
            target_trades_per_minute: 1,
            max_concurrent_trades: 3,
        };
        
        let dex_client = Arc::new(DexClient::new().unwrap());
        let tx_executor = Arc::new(TxExecutor::new().unwrap());
        
        let worker = WorkerAnt::new(
            "test_worker".to_string(),
            worker_config,
            dex_client,
            tx_executor,
            Keypair::new(),
        );
        
        assert_eq!(worker.id, "test_worker");
    }
} 