use crate::config::WorkerConfig;
use crate::dex_client::DexClient;
use crate::dex_provider::Token;
use crate::pathfinder::{TradePath, PathFinder};
use crate::tx_executor::TxExecutor;
use anyhow::{Result, anyhow};
use log::{info, error, debug};
use serde::{Deserialize, Serialize};
use solana_sdk::signature::Keypair;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{Duration, sleep};

#[derive(Debug)]
pub struct WorkerAnt {
    pub id: String,
    pub dex_client: Arc<Mutex<DexClient>>,
    pub tx_executor: Arc<TxExecutor>,
    pub pathfinder: Arc<PathFinder>,
    wallet: Arc<Mutex<Keypair>>,
    pub is_running: Arc<Mutex<bool>>,
    config: WorkerConfig,
    trades_executed: Arc<Mutex<u32>>,
    total_profit: Arc<Mutex<f64>>,
}

impl Serialize for WorkerAnt {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("WorkerAnt", 3)?;
        state.serialize_field("id", &self.id)?;
        state.serialize_field("config", &self.config)?;
        // Skip fields that can't be serialized
        state.end()
    }
}

impl<'de> Deserialize<'de> for WorkerAnt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let id = String::deserialize(deserializer)?;
        let config = WorkerConfig::default();
        let dex_client = Arc::new(Mutex::new(DexClient::new().map_err(serde::de::Error::custom)?));
        let tx_executor = Arc::new(TxExecutor::new().map_err(serde::de::Error::custom)?);
        let wallet = Keypair::new();
        
        // Create a new runtime for async operations
        let rt = tokio::runtime::Runtime::new().map_err(serde::de::Error::custom)?;
        
        // Run the async new function in the runtime
        Ok(rt.block_on(async {
            WorkerAnt::new(
                id,
                config,
                dex_client,
                tx_executor,
                wallet,
            ).await
        }))
    }
}

impl WorkerAnt {
    pub async fn new(
        id: String,
        config: WorkerConfig,
        dex_client: Arc<Mutex<DexClient>>,
        tx_executor: Arc<TxExecutor>,
        wallet: Keypair,
    ) -> Self {
        let dex_client_guard = dex_client.lock().await;
        let pathfinder = Arc::new(PathFinder::new(
            (*dex_client_guard).clone(),
            3,  // max_path_length
            0.01, // min_profit_threshold (1%)
            0.05, // max_price_impact (5%)
        ));
        drop(dex_client_guard);

        Self {
            id,
            config,
            dex_client: dex_client.clone(),
            tx_executor,
            pathfinder,
            wallet: Arc::new(Mutex::new(wallet)),
            is_running: Arc::new(Mutex::new(true)),
            trades_executed: Arc::new(Mutex::new(0)),
            total_profit: Arc::new(Mutex::new(0.0)),
        }
    }

    pub async fn start(&self) -> Result<()> {
        info!("Worker {} starting...", self.id);
        *self.is_running.lock().await = true;
        
        while *self.is_running.lock().await {
            match self.execute_trading_cycle().await {
                Ok(_) => {
                    debug!("Trading cycle completed successfully for worker {}", self.id);
                },
                Err(e) => {
                    error!("Error in trading cycle for worker {}: {}", self.id, e);
                    // Add backoff on error to avoid spamming
                    sleep(Duration::from_secs(5)).await;
                }
            }
            
            // Respect the target trades per minute rate
            let sleep_duration = if self.config.target_trades_per_minute > 0 {
                Duration::from_secs_f64(60.0 / self.config.target_trades_per_minute as f64)
            } else {
                Duration::from_secs(1)
            };
            
            sleep(sleep_duration).await;
        }
        
        info!("Worker {} stopped", self.id);
        Ok(())
    }

    pub async fn stop(&self) -> Result<()> {
        info!("Worker {} stopping...", self.id);
        *self.is_running.lock().await = false;
        Ok(())
    }

    async fn execute_trading_cycle(&self) -> Result<()> {
        // Get wallet balance to determine trade size
        let balance = self.tx_executor.get_balance().await?;
        if balance == 0 {
            return Err(anyhow!("Wallet has zero balance"));
        }
        
        // Get available tokens
        let tokens = self.get_available_tokens().await?;
        if tokens.is_empty() {
            debug!("No tradable tokens found for worker {}", self.id);
            return Ok(());
        }
        
        // Find trading opportunities
        let opportunities = self.find_opportunities(&tokens).await?;
        if opportunities.is_empty() {
            debug!("No profitable opportunities found for worker {}", self.id);
            return Ok(());
        }
        
        // Execute best opportunity if it meets the profit threshold
        let best_opportunity = &opportunities[0];
        if best_opportunity.estimated_profit_percentage < self.config.min_profit_threshold {
            debug!("Best opportunity profit ({:.4}%) is below threshold ({:.4}%)",
                   best_opportunity.estimated_profit_percentage * 100.0,
                   self.config.min_profit_threshold * 100.0);
            return Ok(());
        }
        
        // Execute the trade
        info!("Executing trade with estimated profit of {:.4}%",
              best_opportunity.estimated_profit_percentage * 100.0);
        
        match self.execute_trade(best_opportunity).await {
            Ok(signature) => {
                info!("Trade executed successfully: {}", signature);
                
                // Update metrics
                let mut trades = self.trades_executed.lock().await;
                *trades += 1;
                
                let mut profit = self.total_profit.lock().await;
                *profit += best_opportunity.estimated_profit_amount;
                
                Ok(())
            },
            Err(e) => {
                error!("Failed to execute trade: {}", e);
                Err(e)
            }
        }
    }

    async fn get_available_tokens(&self) -> Result<Vec<Token>> {
        info!("Getting available tokens for worker {}", self.id);
        
        // Get tokens from the DEX client
        let tokens = self.dex_client.lock().await.get_tokens().await?;
        
        // Filter tokens based on liquidity and other criteria
        let filtered_tokens = tokens.into_iter()
            .filter(|token| token.liquidity >= self.config.min_liquidity)
            .collect::<Vec<_>>();
        
        debug!("Found {} tokens with sufficient liquidity", filtered_tokens.len());
        
        Ok(filtered_tokens)
    }

    async fn find_opportunities(&self, tokens: &[Token]) -> Result<Vec<TradePath>> {
        info!("Finding trading opportunities for worker {}", self.id);
        
        // Use the pathfinder to find potential trade paths
        let mut trade_paths = self.dex_client.lock().await.find_arbitrage_paths(tokens).await?;
        
        // Sort by estimated profit percentage (descending)
        trade_paths.sort_by(|a, b| {
            b.estimated_profit_percentage.partial_cmp(&a.estimated_profit_percentage)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Take only the top opportunities based on max_concurrent_trades
        let top_opportunities = trade_paths.into_iter()
            .take(self.config.max_concurrent_trades as usize)
            .collect::<Vec<_>>();
        
        debug!("Found {} potential opportunities", top_opportunities.len());
        
        Ok(top_opportunities)
    }

    async fn execute_trade(&self, path: &TradePath) -> Result<String> {
        info!("Executing trade for path: {:?}", path);
        
        // Calculate trade size based on config and wallet balance
        let balance = self.tx_executor.get_balance().await?;
        let trade_size = (balance as f64 * 0.8).min(self.config.max_trade_size);
        
        // Get the quote for the trade
        let quote = self.dex_client.lock().await.get_best_quote(
            &path.from_token.address,
            &path.to_token.address,
            trade_size
        ).await?;
        
        // Execute the swap
        let signature = self.tx_executor.execute_swap(&quote).await?;
        
        // Update metrics
        let profit = quote.output_amount - quote.input_amount;
        if profit > 0.0 {
            let mut total_profit = self.total_profit.lock().await;
            *total_profit += profit;
        }
        
        Ok(signature)
    }

    pub async fn get_metrics(&self) -> Result<serde_json::Value> {
        let mut metrics = serde_json::Map::new();
        
        metrics.insert("id".to_string(), serde_json::Value::String(self.id.clone()));
        metrics.insert("is_active".to_string(), serde_json::Value::Bool(*self.is_running.lock().await));
        
        // Get wallet balance
        let balance = match self.tx_executor.get_balance().await {
            Ok(balance) => balance as f64 / 1_000_000_000.0, // Convert lamports to SOL
            Err(_) => 0.0,
        };
        metrics.insert("balance".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(balance).unwrap_or(serde_json::Number::from(0))));
        
        let trades = *self.trades_executed.lock().await;
        metrics.insert("trades_executed".to_string(), serde_json::Value::Number(trades.into()));
        
        let profit = *self.total_profit.lock().await;
        // Convert f64 to string to avoid precision issues
        metrics.insert("total_profit".to_string(), serde_json::Value::String(profit.to_string()));
        
        // Add config info
        let mut config_map = serde_json::Map::new();
        config_map.insert("max_trades_per_hour".to_string(), serde_json::Value::Number(self.config.max_trades_per_hour.into()));
        config_map.insert("min_profit_threshold".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(self.config.min_profit_threshold).unwrap()));
        metrics.insert("config".to_string(), serde_json::Value::Object(config_map));
        
        Ok(serde_json::Value::Object(metrics))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_worker_creation() {
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
        
        let dex_client = Arc::new(Mutex::new(DexClient::new().unwrap()));
        let tx_executor = Arc::new(TxExecutor::new().unwrap());
        let wallet = Keypair::new();
        
        let worker = WorkerAnt::new(
            "test_worker".to_string(),
            worker_config,
            dex_client,
            tx_executor,
            wallet,
        ).await;
        
        assert_eq!(worker.id, "test_worker");
    }
    
    #[tokio::test]
    async fn test_worker_metrics() {
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
        
        let dex_client = Arc::new(Mutex::new(DexClient::new().unwrap()));
        let tx_executor = Arc::new(TxExecutor::new().unwrap());
        let wallet = Keypair::new();
        
        let worker = WorkerAnt::new(
            "test_worker".to_string(),
            worker_config,
            dex_client,
            tx_executor,
            wallet,
        ).await;
        
        let metrics = worker.get_metrics().await.unwrap();
        assert!(metrics.is_object());
    }
} 