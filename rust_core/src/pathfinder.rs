use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;
use serde::{Deserialize, Serialize};
use log::{info, warn, error};
use anyhow::Result;

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
pub struct Swap {
    pub from_token: String,
    pub to_token: String,
    pub dex: String,
    pub expected_output: f64,
    pub slippage: f64,
    pub fee: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradePath {
    pub swaps: Vec<Swap>,
    pub expected_profit: f64,
    pub total_slippage: f64,
    pub total_fees: f64,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct Node {
    token: String,
    cost: i64,
    profit: i64,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        // Prioritize nodes with higher profit and lower cost
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

pub struct PathFinder {
    tokens: HashMap<String, Token>,
    swaps: HashMap<String, Vec<Swap>>,
    max_hops: usize,
    min_profit_threshold: f64,
    max_slippage: f64,
}

impl PathFinder {
    pub fn new() -> Self {
        Self {
            tokens: HashMap::new(),
            swaps: HashMap::new(),
            max_hops: 3,
            min_profit_threshold: 0.10, // 10% minimum profit
            max_slippage: 0.02,         // 2% maximum slippage
        }
    }

    pub fn update_token_data(&mut self, token: Token) {
        self.tokens.insert(token.address.clone(), token);
    }

    pub fn add_swap_opportunity(&mut self, swap: Swap) {
        self.swaps
            .entry(swap.from_token.clone())
            .or_insert_with(Vec::new)
            .push(swap);
    }

    pub fn find_optimal_path(&self, start_token: &str, target_profit: f64) -> Option<TradePath> {
        let mut best_path: Option<TradePath> = None;
        let mut best_profit = 0.0;
        
        // Use A* to find the best path
        let mut open_set = BinaryHeap::new();
        let mut closed_set = HashSet::new();
        let mut came_from = HashMap::new();
        let mut g_score = HashMap::new();
        let mut f_score = HashMap::new();
        
        // Initialize start node
        open_set.push(Node {
            token: start_token.to_string(),
            cost: 0,
            profit: 0,
        });
        
        g_score.insert(start_token.to_string(), 0);
        f_score.insert(start_token.to_string(), 0);
        
        while !open_set.is_empty() {
            let current = open_set.pop().unwrap();
            
            if current.profit as f64 >= target_profit * 1_000_000.0 {
                // Reconstruct path
                if let Some(path) = self.reconstruct_path(&came_from, &current.token) {
                    if path.expected_profit > best_profit {
                        best_path = Some(path);
                        best_profit = path.expected_profit;
                    }
                }
                break;
            }
            
            closed_set.insert(current.token.clone());
            
            // Get all possible swaps from current token
            if let Some(swaps) = self.swaps.get(&current.token) {
                for swap in swaps {
                    let neighbor = swap.to_token.clone();
                    
                    if closed_set.contains(&neighbor) {
                        continue;
                    }
                    
                    // Calculate tentative scores
                    let tentative_g_score = g_score[&current.token] + (swap.fee * 1_000_000.0) as i64;
                    let tentative_profit = current.profit + (swap.expected_output * 1_000_000.0) as i64;
                    
                    if !g_score.contains_key(&neighbor) || tentative_g_score < g_score[&neighbor] {
                        came_from.insert(neighbor.clone(), (current.token.clone(), swap.clone()));
                        g_score.insert(neighbor.clone(), tentative_g_score);
                        f_score.insert(neighbor.clone(), tentative_g_score);
                        
                        open_set.push(Node {
                            token: neighbor,
                            cost: tentative_g_score,
                            profit: tentative_profit,
                        });
                    }
                }
            }
        }
        
        best_path
    }

    fn reconstruct_path(&self, came_from: &HashMap<String, (String, Swap)>, current: &str) -> Option<TradePath> {
        let mut swaps = Vec::new();
        let mut current_token = current.to_string();
        let mut total_slippage = 0.0;
        let mut total_fees = 0.0;
        
        while let Some((prev_token, swap)) = came_from.get(&current_token) {
            swaps.push(swap.clone());
            total_slippage += swap.slippage;
            total_fees += swap.fee;
            current_token = prev_token.clone();
            
            // Prevent infinite loops
            if swaps.len() > self.max_hops {
                break;
            }
        }
        
        if swaps.is_empty() {
            return None;
        }
        
        // Reverse to get correct order
        swaps.reverse();
        
        // Calculate expected profit
        let expected_profit = swaps.last().unwrap().expected_output - swaps.first().unwrap().expected_output;
        
        // Check if path meets criteria
        if expected_profit < self.min_profit_threshold || total_slippage > self.max_slippage {
            return None;
        }
        
        Some(TradePath {
            swaps,
            expected_profit,
            total_slippage,
            total_fees,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_finding() {
        let mut pathfinder = PathFinder::new();
        
        // Add test tokens
        pathfinder.update_token_data(Token {
            address: "SOL".to_string(),
            symbol: "SOL".to_string(),
            decimals: 9,
            price: 100.0,
            liquidity: 1_000_000.0,
            volume_24h: 100_000.0,
        });
        
        pathfinder.update_token_data(Token {
            address: "USDC".to_string(),
            symbol: "USDC".to_string(),
            decimals: 6,
            price: 1.0,
            liquidity: 10_000_000.0,
            volume_24h: 1_000_000.0,
        });
        
        // Add test swap
        pathfinder.add_swap_opportunity(Swap {
            from_token: "SOL".to_string(),
            to_token: "USDC".to_string(),
            dex: "Orca".to_string(),
            expected_output: 100.0,
            slippage: 0.001,
            fee: 0.0001,
        });
        
        // Test path finding
        let path = pathfinder.find_optimal_path("SOL", 0.05);
        assert!(path.is_some());
    }
} 