use std::collections::HashSet;
use std::cmp::Ordering;
use log::{info, warn, debug};
use anyhow::Result;

use crate::dex_client::DexClient;
use crate::dex_provider::{Token, Swap};

#[derive(Debug, Clone)]
pub struct TradePath {
    pub path_id: String,
    pub swaps: Vec<Swap>,
    pub profit_percentage: f64,
    pub estimated_profit_amount: f64,
    pub estimated_profit_percentage: f64,
    pub total_price_impact: f64,
    pub from_token: Token,
    pub to_token: Token,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct Node {
    token: String,
    cost: i64,
    profit: i64,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        let self_score = self.profit - self.cost;
        let other_score = other.profit - other.cost;
        other_score.cmp(&self_score)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
pub struct PathFinder {
    dex_client: DexClient,
    max_path_length: usize,
    min_profit_threshold: f64,
    max_price_impact: f64,
    swaps: Vec<Swap>,
}

impl PathFinder {
    pub fn new(dex_client: DexClient, max_path_length: usize, min_profit_threshold: f64, max_price_impact: f64) -> Self {
        Self {
            dex_client,
            max_path_length,
            min_profit_threshold,
            max_price_impact,
            swaps: Vec::new(),
        }
    }

    pub fn update_token_data(&mut self, _token: Token) {
        // Implementation needed
    }

    pub fn add_swap_opportunity(&mut self, swap: Swap) {
        self.swaps.push(swap);
    }

    fn calculate_path_profit(&self, path: &[Swap]) -> f64 {
        if path.is_empty() {
            return 0.0;
        }
        let first_swap = &path[0];
        let last_swap = &path[path.len() - 1];
        (last_swap.expected_output - first_swap.input_amount) / first_swap.input_amount
    }

    pub async fn find_arbitrage_paths(&self, _tokens: &[Token], base_token: &Token, amount: f64) -> Result<Vec<TradePath>> {
        info!("Finding arbitrage paths starting with {} {}", amount, base_token.symbol);
        
        let mut paths = Vec::new();
        let mut visited = HashSet::new();
        
        visited.insert(base_token.symbol.clone());
        
        self.dfs_find_paths(
            base_token,
            base_token,
            &mut visited,
            &mut Vec::new(),
            &mut paths,
            self.max_path_length,
        ).await?;
        
        paths.sort_by(|a, b| {
            b.estimated_profit_percentage.partial_cmp(&a.estimated_profit_percentage)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        debug!("Found {} profitable arbitrage paths", paths.len());
        
        Ok(paths)
    }
    
    async fn dfs_find_paths_boxed(
        &self,
        base_token: &Token,
        current_token: &Token,
        visited: &mut HashSet<String>,
        current_path: &mut Vec<Swap>,
        paths: &mut Vec<TradePath>,
        depth: usize,
    ) -> Result<()> {
        if depth == 0 {
            return Ok(());
        }

        visited.insert(current_token.symbol.clone());
        
        let swap_rate = self.dex_client.get_swap_rate(1.0, current_token, base_token).await?;
        
        // Create a temporary swap for the current rate
        let temp_swap = Swap {
            from_token: current_token.clone(),
            to_token: base_token.clone(),
            input_amount: 1.0,
            expected_output: swap_rate,
            price_impact: 0.0,
            dex: "unknown".to_string(),
            fee: 0.0,
        };
        
        if !visited.contains(&temp_swap.to_token.symbol) {
            current_path.push(temp_swap.clone());
            
            if temp_swap.to_token.symbol == base_token.symbol {
                let path_id = current_path.iter()
                    .map(|s| format!("{}->{}", s.from_token.symbol, s.to_token.symbol))
                    .collect::<Vec<String>>()
                    .join("->");
                    
                let profit_percentage = self.calculate_path_profit(current_path);
                
                paths.push(TradePath {
                    path_id: path_id.clone(),
                    swaps: current_path.clone(),
                    profit_percentage,
                    estimated_profit_amount: temp_swap.expected_output - temp_swap.input_amount,
                    estimated_profit_percentage: profit_percentage,
                    total_price_impact: temp_swap.price_impact,
                    from_token: base_token.clone(),
                    to_token: base_token.clone(),
                });
                
                debug!("Found profitable path: {} -> profit: {:.4}%", path_id, profit_percentage);
            } else {
                Box::pin(self.dfs_find_paths_boxed(
                    base_token,
                    &temp_swap.to_token,
                    visited,
                    current_path,
                    paths,
                    depth - 1,
                )).await?;
            }
            
            current_path.pop();
        }
        
        visited.remove(&current_token.symbol);
        Ok(())
    }
    
    pub async fn dfs_find_paths(
        &self,
        base_token: &Token,
        current_token: &Token,
        visited: &mut HashSet<String>,
        current_path: &mut Vec<Swap>,
        paths: &mut Vec<TradePath>,
        depth: usize,
    ) -> Result<()> {
        if depth == 0 {
            return Ok(());
        }

        visited.insert(current_token.symbol.clone());
        
        let swap_rate = self.dex_client.get_swap_rate(1.0, current_token, base_token).await?;
        
        // Create a temporary swap for the current rate
        let temp_swap = Swap {
            from_token: current_token.clone(),
            to_token: base_token.clone(),
            input_amount: 1.0,
            expected_output: swap_rate,
            price_impact: 0.0,
            dex: "unknown".to_string(),
            fee: 0.0,
        };
        
        if !visited.contains(&temp_swap.to_token.symbol) {
            current_path.push(temp_swap.clone());
            
            if temp_swap.to_token.symbol == base_token.symbol {
                let path_id = current_path.iter()
                    .map(|s| format!("{}->{}", s.from_token.symbol, s.to_token.symbol))
                    .collect::<Vec<String>>()
                    .join("->");
                    
                let profit_percentage = self.calculate_path_profit(current_path);
                
                paths.push(TradePath {
                    path_id: path_id.clone(),
                    swaps: current_path.clone(),
                    profit_percentage,
                    estimated_profit_amount: temp_swap.expected_output - temp_swap.input_amount,
                    estimated_profit_percentage: profit_percentage,
                    total_price_impact: temp_swap.price_impact,
                    from_token: base_token.clone(),
                    to_token: base_token.clone(),
                });
                
                debug!("Found profitable path: {} -> profit: {:.4}%", path_id, profit_percentage);
            } else {
                Box::pin(self.dfs_find_paths(
                    base_token,
                    &temp_swap.to_token,
                    visited,
                    current_path,
                    paths,
                    depth - 1,
                )).await?;
            }
            
            current_path.pop();
        }
        
        visited.remove(&current_token.symbol);
        Ok(())
    }
    
    pub async fn find_direct_paths(&self, from_token: &Token, to_token: &Token, amount: f64) -> Result<Vec<TradePath>> {
        info!("Finding direct paths from {} to {}", from_token.symbol, to_token.symbol);
        
        let mut paths = Vec::new();
        
        // Try to get direct swap
        match self.dex_client.get_swap_rate(amount, from_token, to_token).await {
            Ok(swap_rate) => {
                let path_id = format!("{}->{}", from_token.symbol, to_token.symbol);
                
                let swap = Swap {
                    from_token: from_token.clone(),
                    to_token: to_token.clone(),
                    input_amount: amount,
                    expected_output: swap_rate,
                    price_impact: 0.0, // This would be calculated in a real implementation
                    dex: "unknown".to_string(),
                    fee: 0.0,
                };
                
                let trade_path = TradePath {
                    path_id,
                    swaps: vec![swap.clone()],
                    profit_percentage: 0.0,
                    estimated_profit_amount: swap.expected_output - amount,
                    estimated_profit_percentage: (swap.expected_output - amount) / amount,
                    total_price_impact: swap.price_impact,
                    from_token: from_token.clone(),
                    to_token: to_token.clone(),
                };
                
                paths.push(trade_path);
            }
            Err(e) => {
                warn!("No direct path from {} to {}: {}", from_token.symbol, to_token.symbol, e);
            }
        }
        
        Ok(paths)
    }
    
    pub async fn filter_paths_by_liquidity(&self, paths: Vec<TradePath>, min_liquidity: f64) -> Vec<TradePath> {
        paths.into_iter()
            .filter(|path| {
                // Check liquidity for all tokens in the path
                path.swaps.iter().all(|swap| {
                    // In a real implementation, we would check the liquidity of the tokens
                    // For now, we're using a simplified approach
                    swap.from_token.liquidity >= min_liquidity && swap.to_token.liquidity >= min_liquidity
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dex_client::DexClient;

    #[test]
    fn test_path_finding() {
        // Create a mock DexClient for testing
        let dex_client = DexClient::new().unwrap();
        let mut pathfinder = PathFinder::new(
            3,  // max_path_length
            0.01, // min_profit_threshold (1%)
            0.05, // max_price_impact (5%)
        );
        
        // Add test tokens
        let sol_token = Token {
            address: "SOL".to_string(),
            symbol: "SOL".to_string(),
            decimals: 9,
            price: 100.0,
            liquidity: 1_000_000.0,
            volume_24h: 100_000.0,
        };
        
        let usdc_token = Token {
            address: "USDC".to_string(),
            symbol: "USDC".to_string(),
            decimals: 6,
            price: 1.0,
            liquidity: 10_000_000.0,
            volume_24h: 1_000_000.0,
        };
        
        // Add test swap
        pathfinder.add_swap_opportunity(Swap {
            from_token: sol_token.clone(),
            to_token: usdc_token.clone(),
            input_amount: 100.0,
            expected_output: 100.0,
            price_impact: 0.001,
        });
        
        // Test path finding - this would be an async test in a real scenario
        // For now we just verify the pathfinder was created correctly
        assert_eq!(pathfinder.max_path_length, 3);
        assert_eq!(pathfinder.min_profit_threshold, 0.01);
        assert_eq!(pathfinder.max_price_impact, 0.05);
    }

    #[tokio::test]
    async fn test_pathfinder_creation() {
        let dex_client = DexClient::new().unwrap();
        let pathfinder = PathFinder::new(
            3,  // max_path_length
            0.01, // min_profit_threshold (1%)
            0.05, // max_price_impact (5%)
        );
        
        assert_eq!(pathfinder.max_path_length, 3);
        assert_eq!(pathfinder.min_profit_threshold, 0.01);
        assert_eq!(pathfinder.max_price_impact, 0.05);
    }
    
    // More tests would be added in a real implementation
} 