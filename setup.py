from setuptools import setup, find_packages

setup(
    name="antbot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6.0.1",
        "solana>=0.30.2",
        "anchorpy>=0.14.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
        "aiohttp>=3.8.5",
        "websockets>=11.0.3",
        "streamlit>=1.28.0",
        "plotly>=5.18.0",
        "cryptography>=40.0.2",
        "pytest>=7.4.0"
    ],
) 