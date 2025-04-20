use crate::config::get_worker_config;
use crate::dex_client::DexClient;
use crate::tx_executor::TxExecutor;
use crate::worker_ant::WorkerAnt;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use solana_sdk::signature::Keypair;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

pub mod config;
pub mod dex_client;
pub mod dex_provider;
pub mod pathfinder;
pub mod tx_executor;
pub mod worker_ant;

// FFI exports for Python bindings
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_int};

#[derive(Debug)]
pub struct Colony {
    workers: HashMap<String, Arc<WorkerAnt>>,
    #[allow(dead_code)]
    metrics: Arc<Mutex<HashMap<String, serde_json::Value>>>,
}

impl Serialize for Colony {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Colony", 1)?;
        state.serialize_field("workers", &self.workers)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Colony {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            workers: HashMap<String, Arc<WorkerAnt>>,
        }
        
        let helper = Helper::deserialize(deserializer)?;
        
        Ok(Colony {
            workers: helper.workers,
            metrics: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

impl Colony {
    pub fn new() -> Self {
        Self {
            workers: HashMap::new(),
            metrics: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn add_worker(&mut self, worker_id: String) -> Result<()> {
        let config = get_worker_config().await;
        let dex_client = Arc::new(Mutex::new(DexClient::new()?));
        let tx_executor = Arc::new(TxExecutor::new()?);
        
        // Initialize dex_client
        dex_client.lock().await.initialize().await?;
        
        let worker = WorkerAnt::new(
            worker_id.clone(),
            config,
            dex_client,
            tx_executor,
            Keypair::new(),
        ).await;
        
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

#[no_mangle]
pub extern "C" fn worker_create(
    worker_id: *const c_char,
    wallet_address: *const c_char,
    capital: c_double,
    config_path: *const c_char
) -> c_int {
    // Convert C strings to Rust strings
    let worker_id = unsafe { CStr::from_ptr(worker_id).to_string_lossy().into_owned() };
    let wallet_address = unsafe { CStr::from_ptr(wallet_address).to_string_lossy().into_owned() };
    let config_path = unsafe { CStr::from_ptr(config_path).to_string_lossy().into_owned() };
    
    // Implement worker creation logic here
    println!("Creating worker {} with {} SOL", worker_id, capital);
    
    // Return a dummy handle
    42
}

#[no_mangle]
pub extern "C" fn worker_start(handle: c_int) -> c_int {
    println!("Starting worker with handle {}", handle);
    0 // Success
}

#[no_mangle]
pub extern "C" fn worker_stop(handle: c_int) -> c_int {
    println!("Stopping worker with handle {}", handle);
    0 // Success
}

#[no_mangle]
pub extern "C" fn worker_get_status(handle: c_int, buffer: *mut c_char, buffer_size: c_int) -> c_int {
    let status = r#"{"status": "active", "trades": 0, "profit": 0.0}"#;
    let c_status = CString::new(status).unwrap();
    unsafe {
        libc::strncpy(buffer, c_status.as_ptr(), buffer_size as usize);
    }
    0 // Success
}

#[no_mangle]
pub extern "C" fn worker_update_metrics(handle: c_int, trades: c_int, profit: c_double) -> c_int {
    println!("Updating metrics for worker {}: {} trades, ${} profit", handle, trades, profit);
    0 // Success
} 