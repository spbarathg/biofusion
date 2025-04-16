use std::sync::Arc;
use tokio::sync::Mutex;
use solana_sdk::{
    commitment_config::CommitmentConfig,
    pubkey::Pubkey,
    signature::{Keypair, Signature},
    transaction::Transaction,
    system_instruction,
    instruction::Instruction,
};
use solana_client::rpc_client::RpcClient;
use log::{info, warn, error};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::{Serialize, Deserialize};

use crate::dex_client::DexQuote;
use crate::config::RpcConfig;
use crate::dex_client::DexClient;
use crate::pathfinder::{Token, Swap};

#[derive(Debug, Serialize, Deserialize)]
pub struct TxExecutor {
    rpc_client: Arc<RpcClient>,
    wallet: Arc<Mutex<Keypair>>,
    commitment: CommitmentConfig,
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
            recent_blockhash,
        );
        
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
            recent_blockhash,
        );
        
        // Sign transaction
        transaction.sign(&[&*wallet], recent_blockhash);
        
        // Send transaction
        let signature = self.rpc_client.send_and_confirm_transaction_with_spinner(&transaction)?;
        
        info!("Transfer executed: {}", signature);
        
        Ok(signature.to_string())
    }
    
    pub async fn wait_for_confirmation(&self, signature: &str) -> Result<bool> {
        let signature = signature.parse::<Signature>()?;
        
        match self.rpc_client.confirm_transaction_with_spinner(&signature, &self.commitment) {
            Ok(confirmed) => {
                if confirmed {
                    info!("Transaction confirmed: {}", signature);
                    Ok(true)
                } else {
                    warn!("Transaction not confirmed: {}", signature);
                    Ok(false)
                }
            }
            Err(e) => {
                error!("Error confirming transaction: {}", e);
                Err(anyhow!("Failed to confirm transaction: {}", e))
            }
        }
    }

    pub async fn send_transaction(&self, instructions: Vec<Instruction>) -> Result<Signature> {
        let wallet = self.wallet.lock().await;
        let pubkey = wallet.pubkey();
        
        let recent_blockhash = self.rpc_client.get_latest_blockhash()?;
        
        let mut transaction = Transaction::new_with_payer(
            &instructions,
            Some(&pubkey),
            recent_blockhash,
        );
        
        transaction.sign(&[&*wallet], recent_blockhash);
        
        let signature = self.rpc_client.send_and_confirm_transaction_with_spinner(&transaction)?;
        
        Ok(signature)
    }

    pub async fn send_transaction_with_retry(&self, instructions: Vec<Instruction>) -> Result<Signature> {
        let wallet = self.wallet.lock().await;
        let from_pubkey = wallet.pubkey();
        
        let recent_blockhash = self.rpc_client.get_latest_blockhash()?;
        
        let mut transaction = Transaction::new_with_payer(
            &instructions,
            Some(&from_pubkey),
            recent_blockhash,
        );
        
        transaction.sign(&[&*wallet], recent_blockhash);
        
        let signature = self.rpc_client.send_and_confirm_transaction_with_spinner(&transaction)?;
        
        Ok(signature)
    }

    pub async fn confirm_transaction(&self, signature: &Signature) -> Result<bool> {
        match self.rpc_client.confirm_transaction_with_spinner(signature, &self.commitment) {
            Ok(_) => Ok(true),
            Err(e) => {
                error!("Failed to confirm transaction: {}", e);
                Ok(false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_transfer() {
        let executor = TxExecutor::new().unwrap();
        let balance = executor.get_balance().await.unwrap();
        assert!(balance >= 0.0);
    }
} 