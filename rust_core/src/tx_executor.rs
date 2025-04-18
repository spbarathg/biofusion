use std::sync::Arc;
use tokio::sync::Mutex;
use solana_sdk::{
    commitment_config::CommitmentConfig,
    pubkey::Pubkey,
    signature::{Keypair, Signature, Signer},
    transaction::Transaction,
    system_instruction,
    instruction::Instruction,
};
use solana_client::rpc_client::RpcClient;
use log::{info, warn, error};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use std::fmt;

use crate::dex_provider::{DexQuote, Token, Swap};
use crate::config::RpcConfig;

pub struct TxExecutor {
    rpc_client: Arc<RpcClient>,
    wallet: Arc<Mutex<Keypair>>,
    commitment: CommitmentConfig,
}

impl fmt::Debug for TxExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TxExecutor")
            .field("commitment", &self.commitment)
            .field("wallet", &"<redacted>") // Don't print the keypair
            .finish()
    }
}

impl Serialize for TxExecutor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("TxExecutor", 1)?;
        // We can only serialize the commitment config
        state.serialize_field("commitment", &self.commitment.commitment)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for TxExecutor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            commitment: String,
        }
        
        let helper = Helper::deserialize(deserializer)?;
        
        // Create with defaults
        let commitment = match helper.commitment.as_str() {
            "confirmed" => CommitmentConfig::confirmed(),
            "finalized" => CommitmentConfig::finalized(),
            "processed" => CommitmentConfig::processed(),
            _ => CommitmentConfig::confirmed(),
        };
        
        Ok(TxExecutor {
            rpc_client: Arc::new(RpcClient::new_with_commitment(
                "https://api.mainnet-beta.solana.com".to_string(),
                commitment,
            )),
            wallet: Arc::new(Mutex::new(Keypair::new())),
            commitment,
        })
    }
}

impl TxExecutor {
    pub fn new() -> Result<Self> {
        let rpc_client = Arc::new(RpcClient::new("https://api.mainnet-beta.solana.com".to_string()));
        
        Ok(Self {
            rpc_client,
            wallet: Arc::new(Mutex::new(Keypair::new())),
            commitment: CommitmentConfig::confirmed(),
        })
    }
    
    pub fn new_with_config(config: &RpcConfig, wallet: Keypair) -> Result<Self> {
        let rpc_client = Arc::new(RpcClient::new_with_commitment(
            config.rpc_url.clone(),
            config.commitment,
        ));
        
        Ok(Self {
            rpc_client,
            wallet: Arc::new(Mutex::new(wallet)),
            commitment: config.commitment,
        })
    }
    
    pub async fn get_balance(&self) -> Result<u64> {
        let wallet = self.wallet.lock().await;
        let pubkey = wallet.pubkey();
        
        let balance = self.rpc_client.get_balance(&pubkey)?;
        
        Ok(balance)
    }
    
    pub async fn execute_swap(&self, quote: &DexQuote) -> Result<String> {
        // Get wallet
        let wallet = self.wallet.lock().await;
        let pubkey = wallet.pubkey();
        
        // Get recent blockhash
        let recent_blockhash = self.rpc_client.get_latest_blockhash()?;
        
        // Create transaction
        let mut transaction = Transaction::new_with_payer(
            &[],
            Some(&pubkey),
        );
        
        // Set recent blockhash
        transaction.message.recent_blockhash = recent_blockhash;
        
        // Add swap instruction - this is a stub implementation
        // In a real implementation, we would create the proper swap instruction
        // based on the DEX being used (e.g., Jupiter, Orca, etc.)
        let instructions: Vec<Instruction> = Vec::new();
        
        // Sign transaction
        transaction.sign(&[&*wallet], recent_blockhash);
        
        // Send transaction
        let signature = self.rpc_client.send_and_confirm_transaction_with_spinner(&transaction)?;
        
        info!("Swap executed: {}", signature);
        
        Ok(signature.to_string())
    }
    
    pub fn execute_transaction(&self, transaction: &Transaction) -> Result<String> {
        // Send transaction
        let signature = self.rpc_client.send_and_confirm_transaction_with_spinner(transaction)?;
        
        info!("Transaction executed: {}", signature);
        
        Ok(signature.to_string())
    }
    
    pub async fn transfer_sol(&self, to_pubkey: &Pubkey, amount: f64) -> Result<String> {
        // Get wallet
        let wallet = self.wallet.lock().await;
        let from_pubkey = wallet.pubkey();
        
        // Convert SOL to lamports
        let lamports = (amount * 1_000_000_000.0) as u64;
        
        // Get recent blockhash
        let recent_blockhash = self.rpc_client.get_latest_blockhash()?;
        
        // Create transfer instruction
        let instruction = system_instruction::transfer(
            &from_pubkey,
            to_pubkey,
            lamports,
        );
        
        // Create transaction
        let mut transaction = Transaction::new_with_payer(
            &[instruction],
            Some(&from_pubkey),
        );
        
        // Set recent blockhash
        transaction.message.recent_blockhash = recent_blockhash;
        
        // Sign transaction
        transaction.sign(&[&*wallet], recent_blockhash);
        
        // Send transaction
        let signature = self.rpc_client.send_and_confirm_transaction_with_spinner(&transaction)?;
        
        info!("Transfer executed: {}", signature);
        
        Ok(signature.to_string())
    }
    
    pub async fn wait_for_confirmation(&self, signature: &str) -> Result<bool> {
        let signature = signature.parse::<Signature>()?;
        let blockhash = self.rpc_client.get_latest_blockhash()?;
        
        match self.rpc_client.confirm_transaction_with_spinner(&signature, &blockhash, self.commitment) {
            Ok(_) => {
                info!("Transaction confirmed: {}", signature);
                Ok(true)
            },
            Err(e) => {
                error!("Failed to confirm transaction {}: {}", signature, e);
                Ok(false)
            }
        }
    }
    
    pub async fn send_transaction(&self, instructions: Vec<Instruction>) -> Result<Signature> {
        // Get wallet
        let wallet = self.wallet.lock().await;
        let pubkey = wallet.pubkey();
        
        // Get recent blockhash
        let recent_blockhash = self.rpc_client.get_latest_blockhash()?;
        
        // Create transaction
        let mut transaction = Transaction::new_with_payer(
            &instructions,
            Some(&pubkey),
        );
        
        // Set recent blockhash
        transaction.message.recent_blockhash = recent_blockhash;
        
        // Sign transaction
        transaction.sign(&[&*wallet], recent_blockhash);
        
        // Send transaction
        let signature = self.rpc_client.send_transaction(&transaction)?;
        
        info!("Transaction sent: {}", signature);
        
        Ok(signature)
    }
    
    pub async fn send_transaction_with_retry(&self, instructions: Vec<Instruction>) -> Result<Signature> {
        const MAX_RETRIES: u8 = 5;
        let mut attempt = 0;
        
        while attempt < MAX_RETRIES {
            match self.send_transaction(instructions.clone()).await {
                Ok(signature) => {
                    // Wait for confirmation
                    if self.wait_for_confirmation(&signature.to_string()).await? {
                        return Ok(signature);
                    }
                    
                    // If not confirmed, retry
                    attempt += 1;
                    warn!("Transaction not confirmed, retrying ({}/{})", attempt, MAX_RETRIES);
                },
                Err(e) => {
                    attempt += 1;
                    error!("Failed to send transaction, retrying ({}/{}): {}", attempt, MAX_RETRIES, e);
                }
            }
            
            // Back off exponentially
            tokio::time::sleep(tokio::time::Duration::from_secs(2u64.pow(attempt as u32))).await;
        }
        
        Err(anyhow!("Failed to send transaction after {} attempts", MAX_RETRIES))
    }
    
    pub async fn confirm_transaction(&self, signature: &Signature) -> Result<bool> {
        let status = self.rpc_client.get_signature_status(signature)?;
        
        match status {
            Some(Ok(_)) => {
                info!("Transaction confirmed: {}", signature);
                Ok(true)
            },
            Some(Err(e)) => {
                error!("Transaction failed: {}", e);
                Ok(false)
            },
            None => {
                warn!("Transaction not found in ledger: {}", signature);
                Ok(false)
            }
        }
    }
    
    // Set wallet for the executor - useful for testing and worker initialization
    pub async fn set_wallet(&self, wallet: Keypair) -> Result<()> {
        let mut current_wallet = self.wallet.lock().await;
        *current_wallet = wallet;
        Ok(())
    }
    
    // Get wallet public key
    pub async fn get_wallet_pubkey(&self) -> Result<Pubkey> {
        let wallet = self.wallet.lock().await;
        Ok(wallet.pubkey())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;
    
    async fn test_transfer() {
        let executor = TxExecutor::new().unwrap();
        
        // Test wallet with test funds required
        let wallet = Keypair::new();
        executor.set_wallet(wallet.clone()).await.unwrap();
        
        // Transfer a small amount to another wallet
        let to_pubkey = Pubkey::from_str("3h1zGmCwsRJnVk5BuRNMLsPaQu1y2aqXqXDWYCgrp5UG").unwrap();
        
        let result = executor.transfer_sol(&to_pubkey, 0.001).await;
        
        assert!(result.is_err(), "Transfer should fail with no funds");
    }
    
    async fn test_balance() {
        let executor = TxExecutor::new().unwrap();
        
        // Empty wallet should have zero balance
        let balance = executor.get_balance().await.unwrap();
        assert_eq!(balance, 0);
    }
} 
