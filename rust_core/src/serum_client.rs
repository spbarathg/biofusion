use crate::dex_provider::{DexProvider, DexQuote, TokenInfo};
use anyhow::Result;
use async_trait::async_trait;
use log::debug;
use reqwest::Client;

#[derive(Clone, Debug)]
pub struct SerumClient {
    base_url: String,
    client: Client,
}

impl SerumClient {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            client: Client::new(),
        }
    }
}

#[async_trait]
impl DexProvider for SerumClient {
    async fn get_quote(&self, input_token: &str, output_token: &str, amount: f64) -> Result<DexQuote> {
        debug!("Getting quote for {} {} to {}", amount, input_token, output_token);
        // TODO: Implement actual Serum API call
        Ok(DexQuote {
            input_token: input_token.to_string(),
            output_token: output_token.to_string(),
            input_amount: amount,
            output_amount: amount * 1.0, // Placeholder
            price_impact: 0.0,
            route: vec![],
        })
    }

    async fn get_swap_rate(&self, input_token: &str, output_token: &str) -> Result<f64> {
        debug!("Getting swap rate for {} to {}", input_token, output_token);
        // TODO: Implement actual Serum API call
        Ok(1.0) // Placeholder
    }

    async fn get_token_info(&self, token: &str) -> Result<TokenInfo> {
        debug!("Getting token info for {}", token);
        // TODO: Implement actual Serum API call
        Ok(TokenInfo {
            symbol: token.to_string(),
            decimals: 9,
            price: 1.0,
        })
    }
} 