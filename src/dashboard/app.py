import os
import asyncio
import json
import time
from pathlib import Path
from typing import Optional, Dict, List
import logging

try:
    # Try to import Streamlit
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit not available, install with: pip install streamlit pandas plotly")

# Import core components
from src.core.agents.queen import Queen
from src.utils.logging.logger import setup_logging

# Set up logging
logger = setup_logging("dashboard", "dashboard.log")

class Dashboard:
    """
    Dashboard for monitoring AntBot activity and performance
    """
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = config_path
        self.queen = None
        self.is_connected = False
        self.update_interval = 5  # seconds
        self.performance_data = {
            "timestamp": [],
            "total_capital": [],
            "total_profit": [],
            "active_workers": [],
            "trades": [],
        }
    
    async def connect_to_queen(self) -> bool:
        """Connect to the Queen and initialize if needed"""
        try:
            self.queen = Queen(self.config_path)
            self.is_connected = True
            logger.info(f"Connected to Queen")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Queen: {str(e)}")
            self.is_connected = False
            return False
    
    async def get_colony_state(self) -> Optional[Dict]:
        """Get the current state of the colony"""
        if not self.is_connected:
            success = await self.connect_to_queen()
            if not success:
                return None
        
        try:
            state = await self.queen.get_colony_state()
            return state
        except Exception as e:
            logger.error(f"Failed to get colony state: {str(e)}")
            self.is_connected = False
            return None
    
    async def update_performance_data(self) -> None:
        """Update performance data history"""
        state = await self.get_colony_state()
        if not state:
            return
        
        # Add current state to history
        self.performance_data["timestamp"].append(time.time())
        self.performance_data["total_capital"].append(state.get("total_capital", 0))
        self.performance_data["total_profit"].append(state.get("total_profits", 0))
        
        # Count active workers
        workers = state.get("workers", {})
        active_workers = sum(1 for w in workers.values() if w.get("status", {}).get("is_running", False))
        self.performance_data["active_workers"].append(active_workers)
        
        # Sum trades across all workers
        total_trades = sum(w.get("status", {}).get("trades_executed", 0) for w in workers.values())
        self.performance_data["trades"].append(total_trades)
        
        # Keep only the last 100 data points
        for key in self.performance_data:
            if len(self.performance_data[key]) > 100:
                self.performance_data[key] = self.performance_data[key][-100:]
    
    async def start_queen(self, initial_capital: float = 10.0) -> bool:
        """Start the Queen if not already running"""
        if not self.is_connected:
            success = await self.connect_to_queen()
            if not success:
                return False
        
        try:
            await self.queen.initialize_colony(initial_capital)
            logger.info(f"Started Queen with {initial_capital} SOL")
            return True
        except Exception as e:
            logger.error(f"Failed to start Queen: {str(e)}")
            return False
    
    async def stop_queen(self) -> bool:
        """Stop the Queen and all workers"""
        if not self.is_connected:
            return False
        
        try:
            await self.queen.stop_colony()
            logger.info("Stopped Queen and all workers")
            return True
        except Exception as e:
            logger.error(f"Failed to stop Queen: {str(e)}")
            return False

# Streamlit UI implementation
def render_streamlit_dashboard():
    """Render the Streamlit dashboard UI"""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available, can't render dashboard")
        return
    
    st.set_page_config(
        page_title="AntBot Dashboard",
        page_icon="🐜",
        layout="wide",
    )
    
    st.title("🐜 AntBot Trading Dashboard")
    
    # Initialize dashboard on first run
    if "dashboard" not in st.session_state:
        st.session_state.dashboard = Dashboard()
        st.session_state.colony_state = None
        st.session_state.connected = False
    
    # Create columns for the control panel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Connect to Queen"):
            st.session_state.connected = asyncio.run(
                st.session_state.dashboard.connect_to_queen()
            )
            if st.session_state.connected:
                st.success("Connected to Queen")
            else:
                st.error("Failed to connect to Queen")
    
    with col2:
        capital = st.number_input("Initial Capital (SOL)", min_value=0.1, value=10.0, step=0.1)
        if st.button("Start Colony"):
            success = asyncio.run(st.session_state.dashboard.start_queen(capital))
            if success:
                st.success(f"Started colony with {capital} SOL")
            else:
                st.error("Failed to start colony")
    
    with col3:
        if st.button("Stop Colony"):
            success = asyncio.run(st.session_state.dashboard.stop_queen())
            if success:
                st.success("Colony stopped")
            else:
                st.error("Failed to stop colony")
    
    # Refresh button and auto-refresh toggle
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh Data"):
            st.session_state.colony_state = asyncio.run(
                st.session_state.dashboard.get_colony_state()
            )
            asyncio.run(st.session_state.dashboard.update_performance_data())
    
    with col2:
        if "auto_refresh" not in st.session_state:
            st.session_state.auto_refresh = False
            
        st.session_state.auto_refresh = st.checkbox(
            "Auto Refresh (5s)", value=st.session_state.auto_refresh
        )
    
    # Auto-refresh logic (using Streamlit's rerun functionality)
    if st.session_state.auto_refresh:
        st.session_state.colony_state = asyncio.run(
            st.session_state.dashboard.get_colony_state()
        )
        asyncio.run(st.session_state.dashboard.update_performance_data())
        
        # Add a placeholder for the refresh timer
        refresh_placeholder = st.empty()
        refresh_placeholder.text(f"Last update: {time.strftime('%H:%M:%S')}")
    
    # Display colony state
    if st.session_state.colony_state:
        state = st.session_state.colony_state
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Capital", f"{state.get('total_capital', 0):.3f} SOL")
        with col2:
            st.metric("Total Profits", f"{state.get('total_profits', 0):.3f} SOL")
        with col3:
            worker_count = len(state.get("workers", {}))
            st.metric("Active Workers", f"{worker_count}")
        with col4:
            # Calculate profit percentage
            if state.get('total_capital', 0) > 0:
                profit_pct = (state.get('total_profits', 0) / state.get('total_capital', 1)) * 100
                st.metric("Profit %", f"{profit_pct:.2f}%")
            else:
                st.metric("Profit %", "0.00%")
        
        # Create wallet balance charts
        st.subheader("Wallet Balances")
        
        # Prepare wallet data
        wallet_data = []
        # Queen and savings wallets
        for wallet_type, wallet_info in state.get("wallets", {}).items():
            if isinstance(wallet_info, dict) and "balance" in wallet_info:
                wallet_data.append({
                    "Wallet Type": wallet_type.capitalize(),
                    "Balance (SOL)": wallet_info["balance"],
                    "ID": wallet_info.get("id", "")[:8]
                })
            elif isinstance(wallet_info, list):
                for w in wallet_info:
                    if "balance" in w:
                        wallet_data.append({
                            "Wallet Type": f"{wallet_type.capitalize()} {w.get('id', '')[:8]}",
                            "Balance (SOL)": w["balance"],
                            "ID": w.get("id", "")[:8]
                        })
        
        if wallet_data:
            wallet_df = pd.DataFrame(wallet_data)
            fig = px.bar(
                wallet_df, 
                x="Wallet Type", 
                y="Balance (SOL)",
                title="Wallet Balances",
                color="Wallet Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No wallet data available")
        
        # Worker performance
        st.subheader("Worker Performance")
        
        # Prepare worker data
        worker_data = []
        for worker_id, worker_info in state.get("workers", {}).items():
            status = worker_info.get("status", {})
            worker_data.append({
                "Worker ID": worker_id,
                "Status": "Active" if status.get("is_running", False) else "Inactive",
                "Trades Executed": status.get("trades_executed", 0),
                "Total Profit": status.get("total_profit", 0),
                "Runtime (s)": status.get("runtime", 0),
                "Wallet": worker_info.get("wallet_id", "")[:8]
            })
        
        if worker_data:
            worker_df = pd.DataFrame(worker_data)
            st.dataframe(worker_df)
            
            # Worker profit chart
            fig = px.bar(
                worker_df,
                x="Worker ID",
                y="Total Profit",
                title="Worker Profits",
                color="Status"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No worker data available")
        
        # Performance over time
        st.subheader("Performance Over Time")
        
        # Convert performance data to DataFrame
        perf_data = st.session_state.dashboard.performance_data
        if perf_data["timestamp"]:
            # Convert timestamps to readable format
            import datetime
            readable_timestamps = [
                datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                for ts in perf_data["timestamp"]
            ]
            
            # Capital and profit chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=readable_timestamps,
                y=perf_data["total_capital"],
                mode='lines',
                name='Total Capital'
            ))
            fig.add_trace(go.Scatter(
                x=readable_timestamps,
                y=perf_data["total_profit"],
                mode='lines',
                name='Total Profit'
            ))
            fig.update_layout(title="Capital and Profit Over Time")
            st.plotly_chart(fig, use_container_width=True)
            
            # Workers and trades chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=readable_timestamps,
                y=perf_data["active_workers"],
                mode='lines',
                name='Active Workers'
            ))
            fig.add_trace(go.Scatter(
                x=readable_timestamps,
                y=perf_data["trades"],
                mode='lines',
                name='Total Trades'
            ))
            fig.update_layout(title="Workers and Trades Over Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance history data available yet")
        
    else:
        st.info("Not connected to the Queen. Connect and refresh to see data.")
    
    # Add footer
    st.markdown("---")
    st.markdown("AntBot Dashboard - Simplified Trading System")

# Main function to run the dashboard
def main():
    if STREAMLIT_AVAILABLE:
        # When running via streamlit, it will call this function directly
        render_streamlit_dashboard()
    else:
        print("Streamlit is not available. Install with: pip install streamlit pandas plotly")
        print("Then run: streamlit run src/dashboard/app.py")

if __name__ == "__main__":
    main() 