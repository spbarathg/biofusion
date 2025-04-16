use crate::config::{init_config, get_worker_config};
use crate::dex_client::DexClient;
use crate::tx_executor::TxExecutor;
use crate::worker_ant::WorkerAnt;
use anyhow::Result;
use log::{info, error};
use serde::{Deserialize, Serialize};
use solana_sdk::signature::Keypair;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

pub mod config;
pub mod dex_client;
pub mod pathfinder;
pub mod tx_executor;
pub mod worker_ant;

#[derive(Debug, Serialize, Deserialize)]
pub struct Colony {
    workers: HashMap<String, Arc<WorkerAnt>>,
    metrics: Arc<Mutex<HashMap<String, serde_json::Value>>>,
}

impl Colony {
    pub fn new() -> Self {
        Self {
            workers: HashMap::new(),
            metrics: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn add_worker(&mut self, worker_id: String) -> Result<()> {
        let config = get_worker_config()?;
        let dex_client = Arc::new(DexClient::new()?);
        let tx_executor = Arc::new(TxExecutor::new(&config.rpc_config, Keypair::new())?);
        
        let worker = WorkerAnt::new(
            worker_id.clone(),
            config,
            dex_client,
            tx_executor,
            Keypair::new(),
        );
        
        self.workers.insert(worker_id, Arc::new(worker));
        Ok(())
    }

    pub async fn remove_worker(&mut self, worker_id: &str) -> Result<()> {
        if let Some(worker) = self.workers.remove(worker_id) {
            worker.stop().await?;
        }
        Ok(())
    }

    pub async fn start_worker(&self, worker_id: &str) -> Result<()> {
        if let Some(worker) = self.workers.get(worker_id) {
            worker.start().await?;
        }
        Ok(())
    }

    pub async fn stop_worker(&self, worker_id: &str) -> Result<()> {
        if let Some(worker) = self.workers.get(worker_id) {
            worker.stop().await?;
        }
        Ok(())
    }

    pub async fn get_metrics(&self) -> Result<String> {
        let mut metrics = HashMap::new();
        
        // Add colony metrics
        metrics.insert("worker_count".to_string(), serde_json::to_value(self.workers.len())?);
        
        // Add worker metrics
        for (id, worker) in &self.workers {
            let worker_metrics = worker.get_metrics().await?;
            metrics.insert(format!("worker_{}", id), worker_metrics);
        }
        
        Ok(serde_json::to_string_pretty(&metrics)?)
    }
} 