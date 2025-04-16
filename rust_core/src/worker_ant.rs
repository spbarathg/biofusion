use std::sync::Arc;
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};
use log::{info, warn, error};
use anyhow::{Result, anyhow};
use async_trait::async_trait;

use crate::pathfinder::{Token, Swap, TradePath};
use crate::dex_client::DexClient;
use crate::tx_executor::TxExecutor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    pub max_trades_per_hour: u32,
    pub min_profit_threshold: f64,
    pub max_slippage: f64,
    pub max_trade_size: f64,
    pub min_liquidity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeResult {
    pub success: bool,
    pub profit: f64,
    pub path: Option<TradePath>,
    pub error: Option<String>,
}

pub struct WorkerAnt {
    id: String,
    config: WorkerConfig,
    dex_client: Arc<DexClient>,
    tx_executor: Arc<TxExecutor>,
    is_active: Arc<Mutex<bool>>,
}

impl WorkerAnt {
    pub fn new(
        id: String,
        config: WorkerConfig,
        dex_client: Arc<DexClient>,
        tx_executor: Arc<TxExecutor>,
    ) -> Self {
        Self {
            id,
            config,
            dex_client,
            tx_executor,
            is_active: Arc::new(Mutex::new(true)),
        }
    }

    pub async fn start(&self) -> Result<()> {
        info!("Worker ant {} starting...", self.id);
        
        while *self.is_active.lock().await {
            match self.execute_trading_cycle().await {
                Ok(_) => {
                    info!("Worker ant {} completed trading cycle", self.id);
                }
                Err(e) => {
                    error!("Worker ant {} error: {}", self.id, e);
                }
            }
            
            // Sleep between cycles
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
        
        Ok(())
    }

    pub async fn stop(&self) -> Result<()> {
        info!("Worker ant {} stopping...", self.id);
        *self.is_active.lock().await = false;
        Ok(())
    }

    async fn execute_trading_cycle(&self) -> Result<TradeResult> {
        // Get optimal trading path
        let path = self.find_optimal_path().await?;
        
        // Validate path
        if !self.validate_path(&path).await? {
            return Ok(TradeResult {
                success: false,
                profit: 0.0,
                path: None,
                error: Some("Path validation failed".to_string()),
            });
        }
        
        // Execute trades
        match self.execute_trades(&path).await {
            Ok(profit) => {
                Ok(TradeResult {
                    success: true,
                    profit,
                    path: Some(path),
                    error: None,
                })
            }
            Err(e) => {
                Ok(TradeResult {
                    success: false,
                    profit: 0.0,
                    path: Some(path),
                    error: Some(e.to_string()),
                })
            }
        }
    }

    async fn find_optimal_path(&self) -> Result<TradePath> {
        // Get available tokens
        let tokens = self.get_available_tokens().await?;
        
        // Find best path using pathfinder
        // This is a placeholder - actual implementation would use the pathfinder
        todo!("Implement path finding logic");
    }

    async fn validate_path(&self, path: &TradePath) -> Result<bool> {
        // Check if path meets criteria
        if path.expected_profit < self.config.min_profit_threshold {
            return Ok(false);
        }
        
        if path.total_slippage > self.config.max_slippage {
            return Ok(false);
        }
        
        // Check liquidity for each swap
        for swap in &path.swaps {
            let token = self.dex_client.get_token_info(&swap.to_token).await?;
            if token.liquidity < self.config.min_liquidity {
                return Ok(false);
            }
        }
        
        Ok(true)
    }

    async fn execute_trades(&self, path: &TradePath) -> Result<f64> {
        let mut total_profit = 0.0;
        
        for swap in &path.swaps {
            // Get quote
            let quote = self.dex_client.get_best_quote(
                &swap.from_token,
                &swap.to_token,
                swap.amount,
            ).await?;
            
            // Execute swap
            let tx_hash = self.tx_executor.execute_swap(&quote).await?;
            
            // Wait for confirmation
            if !self.tx_executor.wait_for_confirmation(&tx_hash).await? {
                return Err(anyhow!("Transaction failed: {}", tx_hash));
            }
            
            // Update profit
            total_profit += quote.output_amount - quote.input_amount;
        }
        
        Ok(total_profit)
    }

    async fn get_available_tokens(&self) -> Result<Vec<Token>> {
        // This is a placeholder - actual implementation would fetch from DEX
        todo!("Implement token fetching");
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
        };
        
        let dex_client = Arc::new(DexClient::new().unwrap());
        let tx_executor = Arc::new(TxExecutor::new().unwrap());
        
        let worker = WorkerAnt::new(
            "test_worker".to_string(),
            config,
            dex_client,
            tx_executor,
        );
        
        assert_eq!(worker.id, "test_worker");
    }
} 