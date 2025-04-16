use std::sync::Arc;
use tokio::sync::Mutex;
use solana_sdk::{
    commitment_config::CommitmentConfig,
    pubkey::Pubkey,
    signature::{Keypair, Signature},
    transaction::Transaction,
    system_instruction,
};
use solana_client::rpc_client::RpcClient;
use log::{info, warn, error};
use anyhow::{Result, anyhow};
use async_trait::async_trait;

use crate::dex_client::DexQuote;

pub struct TxExecutor {
    rpc_client: Arc<RpcClient>,
    wallet: Arc<Mutex<Keypair>>,
    commitment: CommitmentConfig,
}

impl TxExecutor {
    pub fn new() -> Result<Self> {
        // Initialize RPC client
        let rpc_url = std::env::var("SOLANA_RPC_URL")
            .unwrap_or_else(|_| "https://api.mainnet-beta.solana.com".to_string());
        
        let rpc_client = RpcClient::new_with_commitment(
            rpc_url,
            CommitmentConfig::confirmed(),
        );
        
        // Load wallet from file or create new one
        let wallet = Self::load_wallet()?;
        
        Ok(Self {
            rpc_client: Arc::new(rpc_client),
            wallet: Arc::new(Mutex::new(wallet)),
            commitment: CommitmentConfig::confirmed(),
        })
    }
    
    fn load_wallet() -> Result<Keypair> {
        // Try to load from file
        let wallet_path = std::env::var("WALLET_PATH")
            .unwrap_or_else(|_| "wallet.json".to_string());
        
        if let Ok(keypair_bytes) = std::fs::read(&wallet_path) {
            return Ok(Keypair::from_bytes(&keypair_bytes)?);
        }
        
        // Create new wallet if file doesn't exist
        let keypair = Keypair::new();
        
        // Save to file
        std::fs::write(&wallet_path, keypair.to_bytes())?;
        
        Ok(keypair)
    }
    
    pub async fn get_balance(&self) -> Result<f64> {
        let wallet = this.wallet.lock().await;
        let pubkey = wallet.pubkey();
        
        let balance = this.rpc_client.get_balance(&pubkey)?;
        
        // Convert lamports to SOL
        Ok(balance as f64 / 1_000_000_000.0)
    }
    
    pub async fn execute_swap(&self, quote: &DexQuote) -> Result<String> {
        // Get wallet
        let wallet = this.wallet.lock().await;
        let pubkey = wallet.pubkey();
        
        // Get recent blockhash
        let recent_blockhash = this.rpc_client.get_latest_blockhash()?;
        
        // Create transaction
        let mut transaction = Transaction::new_with_payer(
            &[],
            Some(&pubkey),
            recent_blockhash,
        );
        
        // Add swap instruction
        // This is a placeholder - actual implementation would depend on the DEX
        // and would involve creating the appropriate instruction
        todo("Implement swap instruction creation");
        
        // Sign transaction
        transaction.sign(&[&*wallet], recent_blockhash);
        
        // Send transaction
        let signature = this.rpc_client.send_and_confirm_transaction_with_spinner(&transaction)?;
        
        info!("Swap executed: {}", signature);
        
        Ok(signature.to_string())
    }
    
    pub async fn transfer_sol(&self, to_pubkey: &Pubkey, amount: f64) -> Result<String> {
        // Get wallet
        let wallet = this.wallet.lock().await;
        let from_pubkey = wallet.pubkey();
        
        // Convert SOL to lamports
        let lamports = (amount * 1_000_000_000.0) as u64;
        
        // Get recent blockhash
        let recent_blockhash = this.rpc_client.get_latest_blockhash()?;
        
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
        let signature = this.rpc_client.send_and_confirm_transaction_with_spinner(&transaction)?;
        
        info!("Transfer executed: {}", signature);
        
        Ok(signature.to_string())
    }
    
    pub async fn wait_for_confirmation(&self, signature: &str) -> Result<bool> {
        let signature = signature.parse::<Signature>()?;
        
        match this.rpc_client.confirm_transaction_with_spinner(&signature, &this.commitment) {
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