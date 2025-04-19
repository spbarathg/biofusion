use crate::config::WorkerConfig;
use crate::dex_client::DexClient;
use crate::tx_executor::TxExecutor;
use anyhow::Result;
use log::{error, info, debug, warn};
use serde::{Deserialize, Serialize};
use solana_sdk::{
    pubkey::Pubkey,
    signature::{Keypair, Signature},
    transaction::Transaction,
};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Serialize, Deserialize)]
pub struct Worker {
    pub id: String,
    pub wallet_address: String,
    pub config: WorkerConfig,
    pub dex_client: Arc<DexClient>,
    pub capital: f64,
    pub active: bool,
    pub trades_executed: usize,
    pub total_profit: f64,
    pub memecoin_focus: bool,
    pub memecoin_list: Vec<String>,
    pub trade_frequency_priority: bool,
}

impl Worker {
    pub fn new(
        id: String, 
        wallet_address: String, 
        capital: f64, 
        config: WorkerConfig,
        dex_client: Arc<DexClient>,
    ) -> Self {
        let memecoin_list = vec![
            "BONK".to_string(), "WIF".to_string(), "MYRO".to_string(),
            "BOPE".to_string(), "POPCAT".to_string(), "SLERF".to_string(),
            "FLOKI".to_string(), "MEMU".to_string(), "JUP".to_string(),
            "BOOK".to_string()
        ];
        
        Self {
            id,
            wallet_address,
            config,
            dex_client,
            capital,
            active: false,
            trades_executed: 0,
            total_profit: 0.0,
            memecoin_focus: true,  // Focus on memecoins by default
            memecoin_list,
            trade_frequency_priority: true,  // Prioritize frequency over size
        }
    }
    
    pub fn set_memecoin_focus(&mut self, enabled: bool) {
        self.memecoin_focus = enabled;
        info!("Worker {}: Memecoin focus {}", self.id, if enabled { "enabled" } else { "disabled" });
    }
    
    pub fn set_trade_frequency_priority(&mut self, enabled: bool) {
        self.trade_frequency_priority = enabled;
        info!("Worker {}: Trade frequency priority {}", self.id, if enabled { "enabled" } else { "disabled" });
    }
    
    pub fn update_memecoin_list(&mut self, tokens: Vec<String>) {
        self.memecoin_list = tokens;
        info!("Worker {}: Updated memecoin list with {} tokens", self.id, tokens.len());
    }
    
    pub fn is_token_allowed(&self, token: &str) -> bool {
        if !self.memecoin_focus {
            return true;
        }
        
        self.memecoin_list.iter().any(|t| t.to_uppercase() == token.to_uppercase())
    }
    
    pub fn calculate_trade_size(&self, opportunity_score: f64) -> f64 {
        if self.trade_frequency_priority {
            // For frequency priority, use smaller trade sizes
            let default_size = self.capital * self.config.default_trade_size_percentage;
            let max_size = self.capital * 0.05; // Max 5% per trade for frequency strategy
            
            let size = if opportunity_score > 0.8 {
                // Great opportunity - use higher size but still limited
                default_size * 1.5
            } else if opportunity_score > 0.5 {
                // Good opportunity - use normal size
                default_size
            } else {
                // Mediocre opportunity - use smaller size
                default_size * 0.7
            };
            
            size.min(max_size)
        } else {
            // For magnitude priority, use larger trade sizes based on opportunity
            let default_size = self.capital * self.config.default_trade_size_percentage;
            let max_size = self.capital * self.config.max_per_trade_percentage;
            
            let size = if opportunity_score > 0.8 {
                // Great opportunity - use higher size
                default_size * 2.5
            } else if opportunity_score > 0.5 {
                // Good opportunity - use normal size
                default_size * 1.5
            } else {
                // Mediocre opportunity - use normal size
                default_size
            };
            
            size.min(max_size)
        }
    }
    
    pub fn start(&mut self) -> Result<()> {
        if self.active {
            warn!("Worker {} is already active", self.id);
            return Ok(());
        }
        
        self.active = true;
        info!("Started worker {} with {} SOL initial capital", self.id, self.capital);
        
        Ok(())
    }
    
    pub fn stop(&mut self) -> Result<()> {
        if !self.active {
            warn!("Worker {} is already stopped", self.id);
            return Ok(());
        }
        
        self.active = false;
        info!("Stopped worker {}", self.id);
        
        Ok(())
    }
    
    pub fn get_status(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "wallet_address": self.wallet_address,
            "capital": self.capital,
            "active": self.active,
            "trades_executed": self.trades_executed,
            "total_profit": self.total_profit,
            "memecoin_focus": self.memecoin_focus,
            "trade_frequency_priority": self.trade_frequency_priority,
            "memecoins": self.memecoin_list
        })
    }
    
    pub fn update_metrics(&mut self, trades: i32, profit: f64) {
        if trades > 0 {
            self.trades_executed += trades as usize;
        }
        
        self.total_profit += profit;
        debug!("Worker {} metrics updated: {} trades, ${:.4} profit", 
               self.id, self.trades_executed, self.total_profit);
    }
} 