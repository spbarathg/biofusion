#[derive(Debug, Serialize, Deserialize)]
pub struct TxExecutor {
    rpc_client: Arc<RpcClient>,
    wallet: Arc<Mutex<Keypair>>,
    commitment: CommitmentConfig,
} 

impl TxExecutor {
    pub fn execute_transaction(&self, transaction: &Transaction) -> Result<String> {
        // Send transaction
        let signature = self.rpc_client.send_and_confirm_transaction_with_spinner(transaction)?;
        
        info!("Swap executed: {}", signature);
        
        Ok(signature.to_string())
    }
} 