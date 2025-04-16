import streamlit as st
import yaml
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from pathlib import Path
import time

# Set page config with dark theme
st.set_page_config(
    page_title="Ant Bot",
    page_icon="🐜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stMetric {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #3E3E3E;
    }
    .stMetricLabel {
        color: #9CA3AF;
    }
    .stMetricValue {
        color: #FAFAFA;
    }
    .stMetricDelta {
        color: #10B981;
    }
    .stMetricDelta[data-delta="negative"] {
        color: #EF4444;
    }
    .stButton>button {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #3E3E3E;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #3E3E3E;
    }
    .stDataFrame {
        background-color: #262730;
        border-radius: 5px;
        border: 1px solid #3E3E3E;
    }
    .stDataFrame [data-testid="stDataFrameData"] {
        color: #FAFAFA;
    }
    .stDataFrame [data-testid="stDataFrameData"] th {
        background-color: #3E3E3E;
        color: #FAFAFA;
    }
    .stDataFrame [data-testid="stDataFrameData"] td {
        color: #FAFAFA;
    }
    .stDataFrame [data-testid="stDataFrameData"] tr:nth-child(even) {
        background-color: #262730;
    }
    .stDataFrame [data-testid="stDataFrameData"] tr:nth-child(odd) {
        background-color: #1E1E1E;
    }
    .stPlotlyChart {
        background-color: #262730;
        border-radius: 5px;
        border: 1px solid #3E3E3E;
    }
    .stSidebar {
        background-color: #262730;
    }
    .stSidebar .sidebar-content {
        background-color: #262730;
    }
    .stSidebar .sidebar-content .sidebar-section {
        background-color: #262730;
    }
    .stSidebar .sidebar-content .sidebar-section h1, 
    .stSidebar .sidebar-content .sidebar-section h2, 
    .stSidebar .sidebar-content .sidebar-section h3, 
    .stSidebar .sidebar-content .sidebar-section p {
        color: #FAFAFA;
    }
    .stSidebar .sidebar-content .sidebar-section .stButton>button {
        background-color: #3E3E3E;
        color: #FAFAFA;
        border: 1px solid #3E3E3E;
        border-radius: 5px;
    }
    .stSidebar .sidebar-content .sidebar-section .stButton>button:hover {
        background-color: #4E4E4E;
    }
</style>
""", unsafe_allow_html=True)

# Add authentication
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == os.environ.get("ANTBOT_PASSWORD", "antbot"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # First run or password not correct
    if "password_correct" not in st.session_state:
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    # Password correct
    elif st.session_state["password_correct"]:
        return True
    # Password incorrect
    else:
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False

# Only show dashboard if password is correct
if check_password():
    # Sidebar
    st.sidebar.title("🐜 Ant Bot")
    
    # Colony controls
    st.sidebar.subheader("Colony Controls")
    
    if st.sidebar.button("Start Colony", key="start_colony"):
        st.sidebar.success("Colony started!")
    
    if st.sidebar.button("Stop Colony", key="stop_colony"):
        st.sidebar.warning("Colony stopped!")
    
    if st.sidebar.button("Emergency Shutdown", key="emergency_shutdown"):
        st.sidebar.error("EMERGENCY SHUTDOWN ACTIVATED!")
    
    # Settings
    st.sidebar.subheader("Settings")
    
    # Load configuration
    config_path = Path("config/settings.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    
    # Main dashboard
    st.title("Ant Bot Colony")
    
    # Colony status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Queen Balance", "10.5 SOL", "+0.2 SOL")
    
    with col2:
        st.metric("Total Colony", "45.2 SOL", "+2.1 SOL")
    
    with col3:
        st.metric("Active Workers", "8/10", "+1")
    
    with col4:
        st.metric("24h Profit", "3.2 SOL", "+0.5 SOL")
    
    # Trading activity
    st.subheader("Recent Trades")
    
    # Sample data - replace with actual data from your bot
    trades_data = {
        "Time": [datetime.now().strftime("%H:%M:%S") for _ in range(10)],
        "Token": ["BONK", "WIF", "MYRO", "BOME", "POPCAT", "BONK", "WIF", "MYRO", "BOME", "POPCAT"],
        "Action": ["Buy", "Sell", "Buy", "Sell", "Buy", "Sell", "Buy", "Sell", "Buy", "Sell"],
        "Amount": [0.5, 0.5, 0.3, 0.3, 0.2, 0.2, 0.4, 0.4, 0.1, 0.1],
        "Price": [0.00001, 0.000012, 0.00002, 0.000022, 0.00003, 0.000032, 0.00004, 0.000042, 0.00005, 0.000052],
        "Profit": [0, 0.02, 0, 0.01, 0, 0.015, 0, 0.025, 0, 0.01]
    }
    
    df = pd.DataFrame(trades_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Colony Growth")
        # Sample data - replace with actual data
        growth_data = pd.DataFrame({
            "Date": pd.date_range(start="2023-01-01", periods=30, freq="D"),
            "Value": [10 + i*0.5 + i*0.1 for i in range(30)]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=growth_data["Date"], 
            y=growth_data["Value"],
            mode='lines',
            line=dict(color='#10B981', width=2),
            name='Colony Value'
        ))
        
        fig.update_layout(
            plot_bgcolor='#262730',
            paper_bgcolor='#262730',
            font=dict(color='#FAFAFA'),
            xaxis=dict(
                gridcolor='#3E3E3E',
                zerolinecolor='#3E3E3E'
            ),
            yaxis=dict(
                gridcolor='#3E3E3E',
                zerolinecolor='#3E3E3E'
            ),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Profit Distribution")
        # Sample data - replace with actual data
        profit_data = pd.DataFrame({
            "Wallet": ["Queen", "Princess 1", "Princess 2", "Worker 1", "Worker 2", "Worker 3", "Worker 4", "Worker 5"],
            "Profit": [1.2, 0.8, 0.7, 0.3, 0.4, 0.2, 0.5, 0.1]
        })
        
        fig = go.Figure(data=[go.Pie(
            labels=profit_data["Wallet"],
            values=profit_data["Profit"],
            hole=.3,
            marker=dict(colors=['#10B981', '#3B82F6', '#8B5CF6', '#F59E0B', '#EF4444', '#EC4899', '#6366F1', '#14B8A6'])
        )])
        
        fig.update_layout(
            plot_bgcolor='#262730',
            paper_bgcolor='#262730',
            font=dict(color='#FAFAFA'),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Configuration
    st.subheader("Configuration")
    
    # Display current config
    st.json(config)
    
    # Allow editing (in a real implementation, you'd want to add validation)
    edited_config = st.text_area("Edit Configuration (YAML)", value=yaml.dump(config))
    
    if st.button("Save Configuration"):
        # In a real implementation, you'd want to validate and safely save
        st.success("Configuration saved!")
    
    # Auto-refresh
    time.sleep(5)
    st.experimental_rerun() 