use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_int};
use std::sync::Arc;
use std::collections::HashMap;
use std::sync::Mutex as StdMutex;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;
use log::{info, error};
use anyhow::Result;

mod worker_ant;
mod pathfinder;
mod dex_client;
mod tx_executor;
mod wallet;
mod config;

use worker_ant::{WorkerAnt, WorkerConfig, TradeResult};
use pathfinder::PathFinder;
use dex_client::DexClient;
use tx_executor::TxExecutor;
use wallet::WalletManager;
use config::{init_config, get_worker_config, get_network_config};

// Runtime for asynchronous operations
lazy_static::lazy_static! {
    static ref RUNTIME: Runtime = Runtime::new().unwrap();
    static ref WORKERS: StdMutex<HashMap<i32, Arc<WorkerAnt>>> = StdMutex::new(HashMap::new());
    static ref NEXT_HANDLE: StdMutex<i32> = StdMutex::new(1);
}

/// Initialize the library
pub fn initialize() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    // Initialize configuration
    RUNTIME.block_on(async {
        init_config().await?;
        Ok(())
    })
}

/// Create a new worker and return its handle
#[no_mangle]
pub extern "C" fn worker_create(
    worker_id: *const c_char,
    wallet_address: *const c_char,
    capital: c_double,
    config_path: *const c_char,
) -> c_int {
    let result = std::panic::catch_unwind(|| {
        if worker_id.is_null() || wallet_address.is_null() || config_path.is_null() {
            error!("Null pointer passed to worker_create");
            return -1;
        }
        
        // Convert C strings to Rust strings
        let worker_id = unsafe { CStr::from_ptr(worker_id) }
            .to_string_lossy()
            .into_owned();
        
        let wallet_address = unsafe { CStr::from_ptr(wallet_address) }
            .to_string_lossy()
            .into_owned();
        
        let config_path = unsafe { CStr::from_ptr(config_path) }
            .to_string_lossy()
            .into_owned();
        
        // Initialize configuration if needed
        if let Err(e) = initialize() {
            error!("Failed to initialize: {}", e);
            return -1;
        }
        
        // Create worker in the runtime
        RUNTIME.block_on(async {
            // Get worker configuration
            let worker_config = get_worker_config().await;
            
            // Initialize dependencies
            let dex_client = match DexClient::new() {
                Ok(client) => Arc::new(client),
                Err(e) => {
                    error!("Failed to create DexClient: {}", e);
                    return -1;
                }
            };
            
            let tx_executor = match TxExecutor::new() {
                Ok(executor) => Arc::new(executor),
                Err(e) => {
                    error!("Failed to create TxExecutor: {}", e);
                    return -1;
                }
            };
            
            // Create worker
            let worker = WorkerAnt::new(
                worker_id.clone(),
                worker_config,
                dex_client,
                tx_executor,
            );
            
            // Get next handle
            let mut next_handle = NEXT_HANDLE.lock().unwrap();
            let handle = *next_handle;
            *next_handle += 1;
            
            // Store worker
            let mut workers = WORKERS.lock().unwrap();
            workers.insert(handle, Arc::new(worker));
            
            info!("Created worker {} with handle {}", worker_id, handle);
            handle
        })
    });
    
    match result {
        Ok(handle) => handle,
        Err(_) => {
            error!("Panic in worker_create");
            -1
        }
    }
}

/// Start a worker
#[no_mangle]
pub extern "C" fn worker_start(handle: c_int) -> c_int {
    let result = std::panic::catch_unwind(|| {
        let workers = WORKERS.lock().unwrap();
        let worker = match workers.get(&handle) {
            Some(worker) => worker.clone(),
            None => {
                error!("Invalid worker handle: {}", handle);
                return -1;
            }
        };
        
        // Start worker in the runtime
        RUNTIME.spawn(async move {
            if let Err(e) = worker.start().await {
                error!("Worker failed: {}", e);
            }
        });
        
        0
    });
    
    match result {
        Ok(code) => code,
        Err(_) => {
            error!("Panic in worker_start");
            -1
        }
    }
}

/// Stop a worker
#[no_mangle]
pub extern "C" fn worker_stop(handle: c_int) -> c_int {
    let result = std::panic::catch_unwind(|| {
        let workers_guard = WORKERS.lock().unwrap();
        let worker = match workers_guard.get(&handle) {
            Some(worker) => worker.clone(),
            None => {
                error!("Invalid worker handle: {}", handle);
                return -1;
            }
        };
        
        // Stop worker in the runtime
        RUNTIME.block_on(async {
            if let Err(e) = worker.stop().await {
                error!("Failed to stop worker: {}", e);
                return -1;
            }
            0
        })
    });
    
    match result {
        Ok(code) => {
            // Remove worker from map if stop was successful
            if code == 0 {
                let mut workers = WORKERS.lock().unwrap();
                workers.remove(&handle);
            }
            code
        },
        Err(_) => {
            error!("Panic in worker_stop");
            -1
        }
    }
}

/// Get worker status
#[no_mangle]
pub extern "C" fn worker_get_status(
    handle: c_int,
    status_buffer: *mut c_char,
    buffer_size: c_int,
) -> c_int {
    let result = std::panic::catch_unwind(|| {
        if status_buffer.is_null() {
            error!("Null pointer passed to worker_get_status");
            return -1;
        }
        
        let workers = WORKERS.lock().unwrap();
        let worker = match workers.get(&handle) {
            Some(worker) => worker.clone(),
            None => {
                error!("Invalid worker handle: {}", handle);
                return -1;
            }
        };
        
        // Get worker metrics
        RUNTIME.block_on(async {
            let metrics = worker.get_metrics().await;
            
            // Convert to JSON
            let json = match serde_json::to_string(&metrics) {
                Ok(json) => json,
                Err(e) => {
                    error!("Failed to serialize worker metrics: {}", e);
                    return -1;
                }
            };
            
            // Check buffer size
            if json.len() >= buffer_size as usize {
                error!("Status buffer too small: {} < {}", buffer_size, json.len());
                return -2;
            }
            
            // Copy to buffer
            let c_string = match CString::new(json) {
                Ok(s) => s,
                Err(e) => {
                    error!("Failed to create C string: {}", e);
                    return -1;
                }
            };
            
            unsafe {
                std::ptr::copy_nonoverlapping(
                    c_string.as_ptr(),
                    status_buffer,
                    json.len() + 1, // Include null terminator
                );
            }
            
            0
        })
    });
    
    match result {
        Ok(code) => code,
        Err(_) => {
            error!("Panic in worker_get_status");
            -1
        }
    }
}

/// Update worker metrics
#[no_mangle]
pub extern "C" fn worker_update_metrics(
    handle: c_int,
    trades: c_int,
    profit: c_double,
) -> c_int {
    let result = std::panic::catch_unwind(|| {
        let workers = WORKERS.lock().unwrap();
        let worker = match workers.get(&handle) {
            Some(worker) => worker.clone(),
            None => {
                error!("Invalid worker handle: {}", handle);
                return -1;
            }
        };
        
        // Update metrics in the runtime
        RUNTIME.block_on(async {
            if let Err(e) = worker.update_metrics(trades as u32, profit).await {
                error!("Failed to update worker metrics: {}", e);
                return -1;
            }
            0
        })
    });
    
    match result {
        Ok(code) => code,
        Err(_) => {
            error!("Panic in worker_update_metrics");
            -1
        }
    }
}

// Implement required functionality for the WorkerAnt to support these FFI functions
impl WorkerAnt {
    pub async fn get_metrics(&self) -> HashMap<String, serde_json::Value> {
        let mut metrics = HashMap::new();
        
        // Add basic metrics
        metrics.insert("worker_id".to_string(), serde_json::to_value(&self.id).unwrap());
        metrics.insert("is_active".to_string(), serde_json::to_value(*self.is_active.lock().await).unwrap());
        
        // Add trading metrics
        metrics
    }
    
    pub async fn update_metrics(&self, trades: u32, profit: f64) -> Result<()> {
        // Update metrics
        // This is a stub implementation
        Ok(())
    }
} 