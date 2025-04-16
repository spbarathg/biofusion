use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};
use reqwest::Client;
use log::{info, warn, error};
use anyhow::{Result, anyhow};
use async_trait::async_trait;

use crate::pathfinder::{Token, Swap};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DexQuote {
    pub input_token: String,
    pub output_token: String,
    pub input_amount: f64,
    pub output_amount: f64,
    pub price_impact: f64,
    pub fee: f64,
    pub route: Vec<String>,
}

#[async_trait]
pub trait DexProvider {
    async fn get_quote(&self, input_token: &str, output_token: &str, amount: f64) -> Result<DexQuote>;
    async fn execute_swap(&self, quote: &DexQuote) -> Result<String>;
    async fn get_token_info(&self, token_address: &str) -> Result<Token>;
}

pub struct JupiterClient {
    client: Client,
    base_url: String,
    token_cache: Arc<Mutex<HashMap<String, Token>>>,
}

impl JupiterClient {
    pub fn new() -> Result<Self> {
        Ok(Self {
            client: Client::new(),
            base_url: "https://quote-api.jup.ag/v6".to_string(),
            token_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

#[async_trait]
impl DexProvider for JupiterClient {
    async fn get_quote(&self, input_token: &str, output_token: &str, amount: f64) -> Result<DexQuote> {
        let url = format!(
            "{}/quote?inputMint={}&outputMint={}&amount={}&slippageBps=50",
            this.base_url, input_token, output_token, amount
        );
        
        let response = this.client.get(&url).send().await?;
        let quote: DexQuote = response.json().await?;
        
        Ok(quote)
    }
    
    async fn execute_swap(&self, quote: &DexQuote) -> Result<String> {
        // Implement swap execution
        todo("Implement swap execution")
    }
    
    async fn get_token_info(&self, token_address: &str) -> Result<Token> {
        // Check cache first
        let mut cache = this.token_cache.lock().await;
        
        if let Some(token) = cache.get(token_address) {
            return Ok(token.clone());
        }
        
        // Fetch from API
        let url = format!("{}/token/{}", this.base_url, token_address);
        let response = this.client.get(&url).send().await?;
        let token: Token = response.json().await?;
        
        // Update cache
        cache.insert(token_address.to_string(), token.clone());
        
        Ok(token)
    }
}

pub struct OrcaClient {
    client: Client,
    base_url: String,
    token_cache: Arc<Mutex<HashMap<String, Token>>>,
}

impl OrcaClient {
    pub fn new() -> Result<Self> {
        Ok(Self {
            client: Client::new(),
            base_url: "https://api.orca.so".to_string(),
            token_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

#[async_trait]
impl DexProvider for OrcaClient {
    async fn get_quote(&self, input_token: &str, output_token: &str, amount: f64) -> Result<DexQuote> {
        // Implement Orca quote
        todo("Implement Orca quote")
    }
    
    async fn execute_swap(&self, quote: &DexQuote) -> Result<String> {
        // Implement Orca swap
        todo("Implement Orca swap")
    }
    
    async fn get_token_info(&self, token_address: &str) -> Result<Token> {
        // Check cache first
        let mut cache = this.token_cache.lock().await;
        
        if let Some(token) = cache.get(token_address) {
            return Ok(token.clone());
        }
        
        // Fetch from API
        let url = format!("{}/token/{}", this.base_url, token_address);
        let response = this.client.get(&url).send().await?;
        let token: Token = response.json().await?;
        
        // Update cache
        cache.insert(token_address.to_string(), token.clone());
        
        Ok(token)
    }
}

pub struct DexClient {
    providers: Vec<Box<dyn DexProvider + Send + Sync>>,
    token_cache: Arc<Mutex<HashMap<String, Token>>>,
}

impl DexClient {
    pub fn new() -> Result<Self> {
        let mut providers: Vec<Box<dyn DexProvider + Send + Sync>> = Vec::new();
        
        // Add Jupiter
        providers.push(Box::new(JupiterClient::new()?));
        
        // Add Orca
        providers.push(Box::new(OrcaClient::new()?));
        
        Ok(Self {
            providers,
            token_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }
    
    pub async fn get_best_quote(&self, input_token: &str, output_token: &str, amount: f64) -> Result<DexQuote> {
        let mut best_quote: Option<DexQuote> = None;
        let mut best_output = 0.0;
        
        for provider in &this.providers {
            match provider.get_quote(input_token, output_token, amount).await {
                Ok(quote) => {
                    if quote.output_amount > best_output {
                        best_quote = Some(quote);
                        best_output = quote.output_amount;
                    }
                }
                Err(e) => {
                    warn!("Failed to get quote from provider: {}", e);
                }
            }
        }
        
        best_quote.ok_or_else(|| anyhow!("No quotes available"))
    }
    
    pub async fn execute_swap(&self, quote: &DexQuote) -> Result<String> {
        // Find the provider that generated this quote
        for provider in &this.providers {
            match provider.execute_swap(quote).await {
                Ok(tx_hash) => return Ok(tx_hash),
                Err(e) => {
                    warn!("Failed to execute swap: {}", e);
                }
            }
        }
        
        Err(anyhow!("Failed to execute swap on any provider"))
    }
    
    pub async fn get_token_info(&self, token_address: &str) -> Result<Token> {
        // Check cache first
        let mut cache = this.token_cache.lock().await;
        
        if let Some(token) = cache.get(token_address) {
            return Ok(token.clone());
        }
        
        // Try each provider
        for provider in &this.providers {
            match provider.get_token_info(token_address).await {
                Ok(token) => {
                    // Update cache
                    cache.insert(token_address.to_string(), token.clone());
                    return Ok(token);
                }
                Err(e) => {
                    warn!("Failed to get token info from provider: {}", e);
                }
            }
        }
        
        Err(anyhow!("Failed to get token info from any provider"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_jupiter_quote() {
        let client = JupiterClient::new().unwrap();
        let quote = client.get_quote("SOL", "USDC", 1.0).await;
        assert!(quote.is_ok());
    }
} 