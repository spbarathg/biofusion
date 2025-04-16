use crate::config::WorkerConfig;
use crate::dex_client::DexClient;
use crate::tx_executor::TxExecutor;
use anyhow::Result;
use log::{error, info};
use serde::{Deserialize, Serialize};
use solana_sdk::{
    pubkey::Pubkey,
    signature::{Keypair, Signature},
    transaction::Transaction,
};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Serialize, Deserialize)]
pub struct Worker {
    config: WorkerConfig,
    dex_client: Arc<DexClient>,
    tx_executor: Arc<TxExecutor>,
    wallet: Arc<Mutex<Keypair>>,
}

impl Worker {
    pub fn new(config: WorkerConfig, wallet: Keypair) -> Result<Self> {
        let dex_client = Arc::new(DexClient::new(&config.dex_config)?);
        let tx_executor = Arc::new(TxExecutor::new(&config.rpc_config, wallet.clone())?);
        let wallet = Arc::new(Mutex::new(wallet));

        Ok(Self {
            config,
            dex_client,
            tx_executor,
            wallet,
        })
    }

    pub async fn start(&self) -> Result<()> {
        info!("Starting worker with config: {:?}", self.config);
        
        // Initialize DEX client
        self.dex_client.initialize().await?;
        
        // Start monitoring for opportunities
        self.monitor_opportunities().await?;
        
        Ok(())
    }

    async fn monitor_opportunities(&self) -> Result<()> {
        loop {
            match self.check_opportunities().await {
                Ok(_) => (),
                Err(e) => {
                    error!("Error checking opportunities: {}", e);
                }
            }
            
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    }

    async fn check_opportunities(&self) -> Result<()> {
        // Get wallet balance
        let balance = self.tx_executor.get_balance().await?;
        
        if balance < self.config.min_balance {
            info!("Balance too low: {} lamports", balance);
            return Ok(());
        }
        
        // TODO: Implement opportunity checking logic
        todo!("Implement opportunity checking logic");
        
        Ok(())
    }
} 