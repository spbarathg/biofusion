use solana_client::rpc_client::RpcClient;
use solana_sdk::{
    commitment_config::CommitmentConfig,
    signature::Keypair,
    transaction::Transaction,
};
use std::sync::Arc;
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};
use anyhow::Result;
use log::info;

#[derive(Debug, Serialize, Deserialize)]
pub struct TxExecutor {
    rpc_client: Arc<RpcClient>,
    wallet: Arc<Mutex<Keypair>>,
    commitment: CommitmentConfig,
} 

impl TxExecutor {
    pub fn new(rpc_url: &str, wallet: Arc<Mutex<Keypair>>) -> Self {
        let rpc_client = Arc::new(RpcClient::new(rpc_url.to_string()));
        let commitment = CommitmentConfig::confirmed();
        
        TxExecutor {
            rpc_client,
            wallet,
            commitment,
        }
    }
    
    pub fn execute_transaction(&self, transaction: &Transaction) -> Result<String> {
        // Send transaction
        let signature = self.rpc_client.send_and_confirm_transaction_with_spinner(transaction)?;
        
        info!("Swap executed: {}", signature);
        
        Ok(signature.to_string())
    }
} 