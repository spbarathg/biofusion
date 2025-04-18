use std::ffi::{c_char, CStr, CString};
use std::os::raw::c_int;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use log::{info, error, LevelFilter};
use anyhow::Result;
use lazy_static::lazy_static;

use crate::config::init_config;
use crate::worker_ant::WorkerAnt;
use crate::dex_client::DexClient;
use crate::tx_executor::TxExecutor;
use solana_sdk::signature::Keypair;

mod worker_ant;
mod pathfinder;
mod dex_client;
mod tx_executor;
mod wallet;
mod config;

use worker_ant::WorkerAnt;
use pathfinder::PathFinder;
use dex_client::DexClient;
use tx_executor::TxExecutor;
use wallet::WalletManager;
use config::{init_config, get_worker_config, get_network_config};

// Shared state for workers
lazy_static! {
    static ref WORKERS: Mutex<HashMap<i32, Arc<WorkerAnt>>> = Mutex::new(HashMap::new());
    static ref NEXT_ID: Mutex<i32> = Mutex::new(1);
}

// Helper function to initialize logging
fn init_logging() {
    let _ = env_logger::builder()
        .filter_level(LevelFilter::Info)
        .try_init();
}

// FFI functions
#[no_mangle]
pub extern "C" fn worker_create(
    worker_id: *const c_char,
    wallet_address: *const c_char,
    capital: f64,
    config_path: *const c_char,
) -> c_int {
    init_logging();
    
    // Convert C strings to Rust strings
    let worker_id_str = unsafe {
        match CStr::from_ptr(worker_id).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };
    
    let wallet_address_str = unsafe {
        match CStr::from_ptr(wallet_address).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };
    
    let config_path_str = unsafe {
        match CStr::from_ptr(config_path).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };
    
    // Initialize tokio runtime for async operations
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return -1,
    };
    
    // Create worker
    rt.block_on(async {
        match create_worker(worker_id_str, wallet_address_str, capital, config_path_str).await {
            Ok(handle) => handle,
            Err(_) => -1,
        }
    })
}

async fn create_worker(
    worker_id: &str,
    wallet_address: &str,
    capital: f64,
    config_path: &str,
) -> Result<c_int> {
    info!("Creating worker {} with {} SOL", worker_id, capital);
    
    // Initialize configuration
    let config = init_config(Some(config_path.to_string())).await?;
    
    // Create components
    let dex_client = Arc::new(DexClient::new()?);
    let tx_executor = Arc::new(TxExecutor::new()?);
    
    // Create a new keypair
    // In a real implementation, we would import from wallet_address
    let keypair = Keypair::new();
    
    // Create worker
    let worker = WorkerAnt::new(
        worker_id.to_string(),
        config.worker_config.clone(),
        dex_client,
        tx_executor,
        keypair,
    );
    
    // Get next available ID
    let mut next_id = NEXT_ID.lock().await;
    let handle = *next_id;
    *next_id += 1;
    
    // Store worker
    let mut workers = WORKERS.lock().await;
    workers.insert(handle, Arc::new(worker));
    
    info!("Worker created with handle {}", handle);
    
    Ok(handle)
}

#[no_mangle]
pub extern "C" fn worker_start(handle: c_int) -> c_int {
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return -1,
    };
    
    rt.block_on(async {
        // Find worker
        let workers = WORKERS.lock().await;
        let worker = match workers.get(&handle) {
            Some(w) => w.clone(),
            None => return -1,
        };
        
        // Start worker in a background task
        tokio::spawn(async move {
            if let Err(e) = worker.start().await {
                error!("Worker {} failed: {}", handle, e);
            }
        });
        
        0
    })
}

#[no_mangle]
pub extern "C" fn worker_stop(handle: c_int) -> c_int {
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return -1,
    };
    
    rt.block_on(async {
        // Find worker
        let workers = WORKERS.lock().await;
        let worker = match workers.get(&handle) {
            Some(w) => w.clone(),
            None => return -1,
        };
        
        // Stop worker
        match worker.stop().await {
            Ok(_) => 0,
            Err(_) => -1,
        }
    })
}

#[no_mangle]
pub extern "C" fn worker_get_status(
    handle: c_int,
    status_buffer: *mut c_char,
    buffer_size: c_int,
) -> c_int {
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return -1,
    };
    
    rt.block_on(async {
        // Find worker
        let workers = WORKERS.lock().await;
        let worker = match workers.get(&handle) {
            Some(w) => w.clone(),
            None => return -1,
        };
        
        // Get metrics
        let metrics = match worker.get_metrics().await {
            Ok(m) => m,
            Err(_) => return -1,
        };
        
        // Convert to JSON string
        let json = match serde_json::to_string(&metrics) {
            Ok(j) => j,
            Err(_) => return -1,
        };
        
        // Copy to buffer
        let c_str = match CString::new(json) {
            Ok(s) => s,
            Err(_) => return -1,
        };
        
        let bytes = c_str.as_bytes_with_nul();
        if bytes.len() > buffer_size as usize {
            return -1;
        }
        
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr() as *const c_char,
                status_buffer,
                bytes.len(),
            );
        }
        
        0
    })
}

#[no_mangle]
pub extern "C" fn worker_update_metrics(
    handle: c_int,
    trades: c_int,
    profit: f64,
) -> c_int {
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return -1,
    };
    
    rt.block_on(async {
        // In a real implementation, we would update metrics
        // For now, just return success
        0
    })
}

// Main function for standalone execution
#[tokio::main]
async fn main() -> Result<()> {
    init_logging();
    
    info!("Starting AntBot Core");
    
    // Initialize configuration
    let config = init_config(None).await?;
    
    // Create components
    let dex_client = Arc::new(DexClient::new()?);
    dex_client.initialize().await?;
    
    let tx_executor = Arc::new(TxExecutor::new()?);
    
    // Create worker
    let worker = WorkerAnt::new(
        "test_worker".to_string(),
        config.worker_config.clone(),
        dex_client.clone(),
        tx_executor.clone(),
        Keypair::new(),
    );
    
    // Start worker
    info!("Starting worker...");
    worker.start().await?;
    
    // Wait for Ctrl+C
    tokio::signal::ctrl_c().await?;
    info!("Shutting down");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use config::WorkerConfig;

    #[tokio::test]
    async fn test_worker_initialization() {
        let worker_config = WorkerConfig {
            max_trades_per_hour: 10,
            min_profit_threshold: 0.01,
            max_slippage: 0.02,
            max_trade_size: 1.0,
            min_liquidity: 1000.0,
            max_hold_time: 3600,
            target_trades_per_minute: 1,
            max_concurrent_trades: 3,
        };
        
        let dex_client = Arc::new(DexClient::new().unwrap());
        let tx_executor = Arc::new(TxExecutor::new().unwrap());
        
        let worker = WorkerAnt::new(
            "test_worker".to_string(),
            worker_config,
            dex_client,
            tx_executor,
            Keypair::new(),
        );
        
        assert_eq!(worker.id, "test_worker");
    }
} 