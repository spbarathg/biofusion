use serde::{Serialize, Deserialize};

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    pub address: String,
    pub symbol: String,
    pub decimals: u8,
    pub price: f64,
    pub liquidity: f64,
    pub volume_24h: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    pub symbol: String,
    pub decimals: u8,
    pub price: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Swap {
    pub from_token: Token,
    pub to_token: Token,
    pub dex: String,
    pub input_amount: f64,
    pub expected_output: f64,
    pub price_impact: f64,
    pub fee: f64,
}

#[async_trait::async_trait]
pub trait DexProvider: Send + Sync + std::fmt::Debug {
    async fn get_quote(&self, input_token: &str, output_token: &str, amount: f64) -> anyhow::Result<DexQuote>;
    async fn execute_swap(&self, quote: &DexQuote) -> anyhow::Result<String>;
    async fn get_token_info(&self, token_address: &str) -> anyhow::Result<TokenInfo>;
    async fn get_swap_rate(&self, from_token: &Token, to_token: &Token, amount: f64) -> anyhow::Result<Swap>;
    fn clone_box(&self) -> Box<dyn DexProvider>;
} 