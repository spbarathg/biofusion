use crate::config::WorkerConfig;
use crate::dex_client::DexClient;
use crate::pathfinder::{Token, TradePath};
use crate::tx_executor::TxExecutor;
use anyhow::Result;
use log::{info, error};
use serde::{Deserialize, Serialize};
use solana_sdk::signature::Keypair;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerAnt {
    id: String,
    config: WorkerConfig,
    dex_client: Arc<DexClient>,
    tx_executor: Arc<TxExecutor>,
    wallet: Arc<Mutex<Keypair>>,
    is_active: Arc<Mutex<bool>>,
    trades_executed: Arc<Mutex<u32>>,
    total_profit: Arc<Mutex<f64>>,
}

impl WorkerAnt {
    pub fn new(
        id: String,
        config: WorkerConfig,
        dex_client: Arc<DexClient>,
        tx_executor: Arc<TxExecutor>,
        wallet: Keypair,
    ) -> Self {
        Self {
            id,
            config,
            dex_client,
            tx_executor,
            wallet: Arc::new(Mutex::new(wallet)),
            is_active: Arc::new(Mutex::new(true)),
            trades_executed: Arc::new(Mutex::new(0)),
            total_profit: Arc::new(Mutex::new(0.0)),
        }
    }

    pub async fn start(&self) -> Result<()> {
        info!("Worker {} starting...", self.id);
        
        while *self.is_active.lock().await {
            match self.execute_trading_cycle().await {
                Ok(_) => (),
                Err(e) => {
                    error!("Error in trading cycle: {}", e);
                }
            }
            
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
        
        Ok(())
    }

    pub async fn stop(&self) -> Result<()> {
        info!("Worker {} stopping...", self.id);
        *self.is_active.lock().await = false;
        Ok(())
    }

    async fn execute_trading_cycle(&self) -> Result<()> {
        // Get available tokens
        let tokens = self.get_available_tokens().await?;
        
        // Find trading opportunities
        let opportunities = self.find_opportunities(&tokens).await?;
        
        // Execute best opportunity if found
        if let Some(opportunity) = opportunities.first() {
            self.execute_trade(opportunity).await?;
        }
        
        Ok(())
    }

    async fn get_available_tokens(&self) -> Result<Vec<Token>> {
        // TODO: Implement token fetching
        Ok(Vec::new())
    }

    async fn find_opportunities(&self, tokens: &[Token]) -> Result<Vec<TradePath>> {
        // TODO: Implement opportunity finding
        Ok(Vec::new())
    }

    async fn execute_trade(&self, path: &TradePath) -> Result<()> {
        // TODO: Implement trade execution
        Ok(())
    }

    pub async fn get_metrics(&self) -> Result<serde_json::Value> {
        let mut metrics = serde_json::Map::new();
        
        metrics.insert("id".to_string(), serde_json::Value::String(self.id.clone()));
        metrics.insert("is_active".to_string(), serde_json::Value::Bool(*self.is_active.lock().await));
        metrics.insert("trades_executed".to_string(), serde_json::Value::Number((*self.trades_executed.lock().await).into()));
        metrics.insert("total_profit".to_string(), serde_json::Value::Number((*self.total_profit.lock().await).into()));
        
        Ok(serde_json::Value::Object(metrics))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_worker_creation() {
        let config = WorkerConfig {
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
        
        let wallet = Keypair::new();
        
        let worker = WorkerAnt::new(
            "test_worker".to_string(),
            config,
            dex_client,
            tx_executor,
            wallet,
        );
        
        assert_eq!(worker.id, "test_worker");
    }
} 