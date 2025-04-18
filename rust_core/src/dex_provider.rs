use anyhow::Result;
use async_trait::async_trait;
use std::fmt::Debug;

#[async_trait]
pub trait DexProvider: Send + Sync + Debug {
    async fn get_quote(&self, input_token: &str, output_token: &str, amount: f64) -> Result<DexQuote>;
    async fn execute_swap(&self, quote: &DexQuote) -> Result<String>;
    async fn get_token_info(&self, token_address: &str) -> Result<TokenInfo>;
    async fn get_swap_rate(&self, from_token: &Token, to_token: &Token, amount: f64) -> Result<Swap>;
    fn clone_box(&self) -> Box<dyn DexProvider>;
}

#[derive(Debug, Clone)]
pub struct DexQuote {
    pub input_token: String,
    pub output_token: String,
    pub input_amount: f64,
    pub output_amount: f64,
    pub price_impact: f64,
    pub fee: f64,
    pub route: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TokenInfo {
    pub symbol: String,
    pub decimals: u8,
    pub price: f64,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub address: String,
    pub symbol: String,
    pub decimals: u8,
    pub price: f64,
    pub liquidity: f64,
    pub volume_24h: f64,
}

#[derive(Debug, Clone)]
pub struct Swap {
    pub from_token: String,
    pub to_token: String,
    pub dex: String,
    pub input_amount: f64,
    pub expected_output: f64,
    pub price_impact: f64,
    pub fee: f64,
} 