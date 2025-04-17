use serde::{Deserialize, Serialize, ser::SerializeStruct};
use std::sync::Arc;
use tokio::sync::Mutex;
use solana_sdk::signature::Keypair;
use log::{info, error};
use anyhow::{Result, anyhow};

// Import the components we created
use crate::dex_client::DexClient;
use crate::tx_executor::TxExecutor;

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerConfig {
    pub max_trades_per_hour: u32,
    pub min_profit_threshold: f64,
    pub max_slippage: f64,
    pub max_position_size: u64,
    pub min_hold_time_seconds: u32,
    pub max_hold_time_seconds: u32,
}

#[derive(Debug)]
pub struct WorkerAnt {
    id: String,
    config: WorkerConfig,
    dex_client: Arc<DexClient>,
    tx_executor: Arc<TxExecutor>,
    wallet: Arc<Mutex<Keypair>>,
    is_active: Arc<Mutex<bool>>,
    trades_executed: Arc<Mutex<u32>>,
    total_profit: Arc<Mutex<f64>>,
}

impl Serialize for WorkerAnt {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("WorkerAnt", 7)?;
        state.serialize_field("id", &self.id)?;
        state.serialize_field("config", &self.config)?;
        state.serialize_field("dex_client", &self.dex_client)?;
        state.serialize_field("tx_executor", &self.tx_executor)?;
        // For Mutex fields, we'll serialize their default values
        state.serialize_field("wallet", &Keypair::new())?;
        state.serialize_field("is_active", &true)?;
        state.serialize_field("trades_executed", &0)?;
        state.serialize_field("total_profit", &0.0)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for WorkerAnt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            Id,
            Config,
            DexClient,
            TxExecutor,
            Wallet,
            IsActive,
            TradesExecuted,
            TotalProfit,
        }

        struct WorkerAntVisitor;
        impl<'de> serde::de::Visitor<'de> for WorkerAntVisitor {
            type Value = WorkerAnt;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct WorkerAnt")
            }

            fn visit_map<V>(self, mut map: V) -> Result<WorkerAnt, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut id = None;
                let mut config = None;
                let mut dex_client = None;
                let mut tx_executor = None;
                let mut wallet = None;
                let mut is_active = None;
                let mut trades_executed = None;
                let mut total_profit = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Id => id = Some(map.next_value()?),
                        Field::Config => config = Some(map.next_value()?),
                        Field::DexClient => dex_client = Some(map.next_value()?),
                        Field::TxExecutor => tx_executor = Some(map.next_value()?),
                        Field::Wallet => wallet = Some(map.next_value()?),
                        Field::IsActive => is_active = Some(map.next_value()?),
                        Field::TradesExecuted => trades_executed = Some(map.next_value()?),
                        Field::TotalProfit => total_profit = Some(map.next_value()?),
                    }
                }

                let id = id.ok_or_else(|| serde::de::Error::missing_field("id"))?;
                let config = config.ok_or_else(|| serde::de::Error::missing_field("config"))?;
                let dex_client = dex_client.ok_or_else(|| serde::de::Error::missing_field("dex_client"))?;
                let tx_executor = tx_executor.ok_or_else(|| serde::de::Error::missing_field("tx_executor"))?;
                let _wallet = wallet.ok_or_else(|| serde::de::Error::missing_field("wallet"))?;
                let _is_active = is_active.ok_or_else(|| serde::de::Error::missing_field("is_active"))?;
                let _trades_executed = trades_executed.ok_or_else(|| serde::de::Error::missing_field("trades_executed"))?;
                let _total_profit = total_profit.ok_or_else(|| serde::de::Error::missing_field("total_profit"))?;

                Ok(WorkerAnt {
                    id,
                    config,
                    dex_client,
                    tx_executor,
                    wallet: Arc::new(Mutex::new(Keypair::new())),
                    is_active: Arc::new(Mutex::new(true)),
                    trades_executed: Arc::new(Mutex::new(0)),
                    total_profit: Arc::new(Mutex::new(0.0)),
                })
            }
        }

        deserializer.deserialize_struct(
            "WorkerAnt",
            &["id", "config", "dex_client", "tx_executor", "wallet", "is_active", "trades_executed", "total_profit"],
            WorkerAntVisitor,
        )
    }
}

impl WorkerAnt {
    pub fn new(
        id: String,
        config: WorkerConfig,
        dex_client: Arc<DexClient>,
        tx_executor: Arc<TxExecutor>,
        wallet: Arc<Mutex<Keypair>>,
    ) -> Self {
        WorkerAnt {
            id,
            config,
            dex_client,
            tx_executor,
            wallet,
            is_active: Arc::new(Mutex::new(true)),
            trades_executed: Arc::new(Mutex::new(0)),
            total_profit: Arc::new(Mutex::new(0.0)),
        }
    }

    pub async fn start(&self) -> Result<()> {
        info!("Starting worker ant: {}", self.id);
        *self.is_active.lock().await = true;
        Ok(())
    }

    pub async fn stop(&self) -> Result<()> {
        info!("Stopping worker ant: {}", self.id);
        *self.is_active.lock().await = false;
        Ok(())
    }

    pub async fn is_active(&self) -> bool {
        *self.is_active.lock().await
    }

    pub async fn get_metrics(&self) -> Result<serde_json::Value> {
        let trades = *self.trades_executed.lock().await;
        let profit = *self.total_profit.lock().await;
        
        let metrics = serde_json::json!({
            "id": self.id,
            "trades_executed": trades,
            "total_profit": profit,
            "is_active": self.is_active().await,
        });
        
        Ok(metrics)
    }

    pub async fn record_trade_result(&self, profit: f64) -> Result<()> {
        let mut trades = self.trades_executed.lock().await;
        let mut total_profit = self.total_profit.lock().await;
        
        *trades += 1;
        *total_profit += profit;
        
        info!("Worker {} recorded trade: trades={}, profit={:.4}, total_profit={:.4}", 
              self.id, *trades, profit, *total_profit);
        
        Ok(())
    }
} 