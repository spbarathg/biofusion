# Ant Bot Dashboard

A minimalistic, dark-themed Streamlit dashboard for monitoring and controlling your Ant Bot trading system.

## Features

- 🔒 Password-protected access
- 📊 Real-time colony metrics
- 📈 Performance visualizations
- 🛠️ Colony controls
- ⚙️ Configuration management

## Installation

1. Make sure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

2. Set your dashboard password (optional):

```bash
# On Windows
set ANTBOT_PASSWORD=your_secure_password

# On Linux/Mac
export ANTBOT_PASSWORD=your_secure_password
```

## Running the Dashboard

Run the dashboard using the provided script:

```bash
python run_dashboard.py
```

Or directly with Streamlit:

```bash
streamlit run dashboard.py
```

## Default Password

The default password is `antbot`. It's recommended to change this by setting the `ANTBOT_PASSWORD` environment variable.

## Customization

You can customize the dashboard by modifying the `dashboard.py` file:

- Change the color scheme by modifying the CSS in the `st.markdown` section
- Add or remove metrics in the colony status section
- Modify the charts and visualizations
- Add additional controls or settings

## Security Notes

- This dashboard is designed for private use only
- Always use a strong password
- Consider running the dashboard only on your local machine or a private VPS
- If deploying to a VPS, use HTTPS and a reverse proxy like Nginx 