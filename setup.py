from setuptools import setup, find_packages

setup(
    name="antbot",
    version="0.1.0",
    description="A Solana trading bot with multi-agent architecture",
    author="AntBot Team",
    author_email="info@example.com",
    packages=find_packages(),
    install_requires=[
        "loguru>=0.5.3",
        "pyyaml>=6.0",
        "asyncio>=3.4.3",
        "solana>=0.27.0",
        "solders>=0.14.0",
        "python-dotenv>=0.19.0",
        "aiohttp>=3.8.0",
        "cffi>=1.15.0",
    ],
    extras_require={
        "dashboard": [
            "streamlit>=1.19.0",
            "pandas>=1.5.0",
            "plotly>=5.10.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=23.1.0",
            "mypy>=1.0.0",
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "antbot=src.core.agents.queen:main",
            "antbot-dashboard=src.dashboard.app:main",
        ],
    },
) 