use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use solana_sdk::{
    pubkey::Pubkey,
    signature::{Keypair, Signature},
    signer::Signer,
};
use solana_client::rpc_client::RpcClient;
use log::{info, warn, error};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletInfo {
    pub pubkey: String,
    pub balance: f64,
    pub is_queen: bool,
    pub is_princess: bool,
    pub is_worker: bool,
    pub parent_pubkey: Option<String>,
    pub created_at: String,
}

pub struct WalletManager {
    rpc_client: Arc<RpcClient>,
    wallets: Arc<Mutex<Vec<WalletInfo>>>,
    wallet_dir: String,
}

impl WalletManager {
    pub fn new(rpc_client: Arc<RpcClient>, wallet_dir: String) -> Result<Self> {
        // Create wallet directory if it doesn't exist
        if !Path::new(&wallet_dir).exists() {
            fs::create_dir_all(&wallet_dir)?;
        }
        
        Ok(Self {
            rpc_client,
            wallets: Arc::new(Mutex::new(Vec::new())),
            wallet_dir,
        })
    }
    
    pub async fn load_wallets(&self) -> Result<()> {
        let wallet_path = Path::new(&self.wallet_dir).join("wallets.json");
        
        if wallet_path.exists() {
            let wallet_data = fs::read_to_string(&wallet_path)?;
            let wallets: Vec<WalletInfo> = serde_json::from_str(&wallet_data)?;
            
            let mut wallet_list = self.wallets.lock().await;
            *wallet_list = wallets;
            
            info!("Loaded {} wallets", wallet_list.len());
        } else {
            info!("No existing wallets found, starting fresh");
        }
        
        Ok(())
    }
    
    pub async fn save_wallets(&self) -> Result<()> {
        let wallet_path = Path::new(&self.wallet_dir).join("wallets.json");
        let wallet_list = self.wallets.lock().await;
        
        let wallet_data = serde_json::to_string_pretty(&*wallet_list)?;
        fs::write(&wallet_path, wallet_data)?;
        
        info!("Saved {} wallets", wallet_list.len());
        
        Ok(())
    }
    
    pub async fn create_wallet(&self, wallet_type: &str, parent_pubkey: Option<String>) -> Result<WalletInfo> {
        // Generate new keypair
        let keypair = Keypair::new();
        let pubkey = keypair.pubkey().to_string();
        
        // Get current balance
        let balance = self.get_balance(&pubkey).await?;
        
        // Create wallet info
        let wallet_info = WalletInfo {
            pubkey: pubkey.clone(),
            balance,
            is_queen: wallet_type == "queen",
            is_princess: wallet_type == "princess",
            is_worker: wallet_type == "worker",
            parent_pubkey,
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        
        // Save keypair to file
        let keypair_path = Path::new(&self.wallet_dir).join(format!("{}.json", pubkey));
        let keypair_data = serde_json::to_string_pretty(&keypair.to_bytes())?;
        fs::write(&keypair_path, keypair_data)?;
        
        // Add to wallet list
        let mut wallet_list = self.wallets.lock().await;
        wallet_list.push(wallet_info.clone());
        
        // Save updated wallet list
        self.save_wallets().await?;
        
        info!("Created new {} wallet: {}", wallet_type, pubkey);
        
        Ok(wallet_info)
    }
    
    pub async fn get_wallet(&self, pubkey: &str) -> Result<Option<WalletInfo>> {
        let wallet_list = self.wallets.lock().await;
        
        for wallet in wallet_list.iter() {
            if wallet.pubkey == pubkey {
                return Ok(Some(wallet.clone()));
            }
        }
        
        Ok(None)
    }
    
    pub async fn get_wallets_by_type(&self, wallet_type: &str) -> Result<Vec<WalletInfo>> {
        let wallet_list = self.wallets.lock().await;
        let mut filtered_wallets = Vec::new();
        
        for wallet in wallet_list.iter() {
            match wallet_type {
                "queen" if wallet.is_queen => filtered_wallets.push(wallet.clone()),
                "princess" if wallet.is_princess => filtered_wallets.push(wallet.clone()),
                "worker" if wallet.is_worker => filtered_wallets.push(wallet.clone()),
                _ => {}
            }
        }
        
        Ok(filtered_wallets)
    }
    
    pub async fn get_balance(&self, pubkey: &str) -> Result<f64> {
        let pubkey = pubkey.parse::<Pubkey>()?;
        let balance = self.rpc_client.get_balance(&pubkey)?;
        
        // Convert lamports to SOL
        Ok(balance as f64 / 1_000_000_000.0)
    }
    
    pub async fn update_balances(&self) -> Result<()> {
        let mut wallet_list = self.wallets.lock().await;
        
        for wallet in wallet_list.iter_mut() {
            wallet.balance = self.get_balance(&wallet.pubkey).await?;
        }
        
        // Save updated balances
        self.save_wallets().await?;
        
        Ok(())
    }
    
    pub async fn load_keypair(&self, pubkey: &str) -> Result<Keypair> {
        let keypair_path = Path::new(&self.wallet_dir).join(format!("{}.json", pubkey));
        
        if !keypair_path.exists() {
            return Err(anyhow!("Keypair file not found for {}", pubkey));
        }
        
        let keypair_data = fs::read_to_string(&keypair_path)?;
        let keypair_bytes: Vec<u8> = serde_json::from_str(&keypair_data)?;
        
        Ok(Keypair::from_bytes(&keypair_bytes)?)
    }
    
    pub async fn update_wallet_type(&self, pubkey: &str, new_type: &str) -> Result<()> {
        let mut wallet_list = self.wallets.lock().await;
        
        for wallet in wallet_list.iter_mut() {
            if wallet.pubkey == pubkey {
                wallet.is_queen = new_type == "queen";
                wallet.is_princess = new_type == "princess";
                wallet.is_worker = new_type == "worker";
                break;
            }
        }
        
        // Save updated wallet list
        self.save_wallets().await?;
        
        info!("Updated wallet {} to type {}", pubkey, new_type);
        
        Ok(())
    }
    
    pub async fn get_children(&self, parent_pubkey: &str) -> Result<Vec<WalletInfo>> {
        let wallet_list = self.wallets.lock().await;
        let mut children = Vec::new();
        
        for wallet in wallet_list.iter() {
            if let Some(parent) = &wallet.parent_pubkey {
                if parent == parent_pubkey {
                    children.push(wallet.clone());
                }
            }
        }
        
        Ok(children)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_wallet_creation() {
        let temp_dir = tempdir().unwrap();
        let wallet_dir = temp_dir.path().to_str().unwrap().to_string();
        
        let rpc_client = Arc::new(RpcClient::new("https://api.mainnet-beta.solana.com".to_string()));
        let wallet_manager = WalletManager::new(rpc_client, wallet_dir).unwrap();
        
        let wallet = wallet_manager.create_wallet("queen", None).await.unwrap();
        assert!(wallet.is_queen);
        assert!(!wallet.is_princess);
        assert!(!wallet.is_worker);
        assert!(wallet.parent_pubkey.is_none());
    }
} 