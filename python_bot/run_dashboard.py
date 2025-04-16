import os
import subprocess
import sys
from pathlib import Path

def main():
    # Set environment variables
    os.environ["ANTBOT_PASSWORD"] = "antbot"  # Change this to your desired password
    
    # Get the path to the dashboard.py file
    dashboard_path = Path(__file__).parent / "dashboard.py"
    
    # Run the Streamlit dashboard
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])

if __name__ == "__main__":
    main() 