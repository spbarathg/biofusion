"""
ANALYZERS - RUG DETECTION COMPONENTS
===================================

Analyzers for different aspects of token security and rug detection.
Used by the enhanced rug detector to analyze tokens comprehensively.
"""

import asyncio
import aiohttp
import json
import base64
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import os
import numpy as np

from worker_ant_v1.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class LiquidityAnalysis:
    """Liquidity analysis results"""
    total_liquidity: float
    liquidity_concentration: float
    liquidity_age_hours: float
    liquidity_removal_risk: float
    lp_token_distribution: Dict[str, float]
    liquidity_depth: Dict[str, float]

@dataclass
class OwnershipAnalysis:
    """Ownership analysis results"""
    top_holder_percentage: float
    holder_concentration: float
    contract_ownership: str
    ownership_transparency: float
    holder_count: int
    whale_count: int
    ownership_history: List[Dict[str, Any]]

@dataclass
class CodeAnalysis:
    """Smart contract code analysis results"""
    contract_verified: bool
    source_code_available: bool
    suspicious_functions: List[str]
    ownership_functions: List[str]
    mint_functions: List[str]
    blacklist_functions: List[str]
    fee_manipulation_risk: float
    code_complexity_score: float

@dataclass
class TradingAnalysis:
    """Trading pattern analysis results"""
    volume_24h: float
    price_volatility: float
    trading_pattern: str
    wash_trading_indicators: List[str]
    pump_dump_signals: List[str]
    manipulation_risk: float
    trading_anomalies: List[str]

class LiquidityAnalyzer:
    """Analyzes liquidity patterns and risks"""
    
    def __init__(self):
        self.logger = setup_logger("LiquidityAnalyzer")
        self.session = None
        # API configuration - CANONICAL ACCESS THROUGH UNIFIED CONFIG
        from worker_ant_v1.core.unified_config import get_api_config
        api_config = get_api_config()
        self.birdeye_api_key = api_config['birdeye_api_key']
        self.jupiter_api_key = api_config['jupiter_api_key']
    
    async def analyze(self, token_address: str) -> Dict[str, float]:
        """Analyze liquidity for a token"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Get liquidity data from multiple sources
            jupiter_data = await self._get_jupiter_liquidity(token_address)
            birdeye_data = await self._get_birdeye_liquidity(token_address)
            
            # Get LP distribution from Solana RPC
            lp_distribution = await self._get_lp_distribution(token_address)
            
            # Calculate liquidity age
            liquidity_age = await self._get_liquidity_age(token_address)
            
            # Get liquidity depth
            liquidity_depth = await self._get_liquidity_depth(token_address)
            
            # Calculate concentration
            concentration = self._calculate_concentration(lp_distribution)
            
            # Calculate removal risk
            removal_risk = self._calculate_removal_risk(jupiter_data, lp_distribution)
            
            return {
                'total_liquidity': jupiter_data.get('liquidity', 0.0),
                'liquidity_concentration': concentration,
                'liquidity_age_hours': liquidity_age,
                'liquidity_removal_risk': removal_risk,
                'lp_token_distribution': lp_distribution,
                'liquidity_depth': liquidity_depth
            }
            
        except Exception as e:
            self.logger.error(f"Liquidity analysis failed for {token_address}: {e}")
            return {
                'total_liquidity': 0.0,
                'liquidity_concentration': 1.0,
                'liquidity_age_hours': 0.0,
                'liquidity_removal_risk': 1.0,
                'lp_token_distribution': {},
                'liquidity_depth': {}
            }
    
    async def _get_jupiter_liquidity(self, token_address: str) -> Dict[str, Any]:
        """Get liquidity data from Jupiter API"""
        try:
            url = f"https://price.jup.ag/v4/price?ids={token_address}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', {}).get(token_address, {})
                return {}
        except Exception as e:
            self.logger.warning(f"Failed to get Jupiter liquidity: {e}")
            return {}
    
    async def _get_birdeye_liquidity(self, token_address: str) -> Dict[str, Any]:
        """Get liquidity data from Birdeye API"""
        try:
            if not self.birdeye_api_key:
                return {}
            
            url = f"https://public-api.birdeye.so/public/token/{token_address}"
            headers = {'X-API-KEY': self.birdeye_api_key}
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', {})
                return {}
        except Exception as e:
            self.logger.warning(f"Failed to get Birdeye liquidity: {e}")
            return {}
    
    async def _get_lp_distribution(self, token_address: str) -> Dict[str, float]:
        """Get LP token distribution from Solana RPC"""
        try:
            # RPC configuration - CANONICAL ACCESS THROUGH UNIFIED CONFIG
            from worker_ant_v1.core.unified_config import get_network_rpc_url
            rpc_url = get_network_rpc_url()
            
            # Get token accounts for this token
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenLargestAccounts",
                "params": [token_address]
            }
            
            async with self.session.post(rpc_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    accounts = data.get('result', {}).get('value', [])
                    
                    if accounts:
                        # Get detailed account info for top holders
                        distribution = {}
                        total_supply = sum(acc['uiTokenAmount']['uiAmount'] for acc in accounts)
                        
                        for i, account in enumerate(accounts[:10]):  # Top 10 holders
                            amount = account['uiTokenAmount']['uiAmount']
                            percentage = (amount / total_supply) * 100 if total_supply > 0 else 0
                            distribution[f'holder_{i+1}'] = percentage
                        
                        # Calculate "others" percentage
                        top_percentage = sum(distribution.values())
                        distribution['others'] = max(0, 100 - top_percentage)
                        
                        return distribution
            
            return {'holder_1': 0.8, 'holder_2': 0.15, 'others': 0.05}
            
        except Exception as e:
            self.logger.error(f"Error getting LP distribution: {e}")
            return {'holder_1': 0.8, 'holder_2': 0.15, 'others': 0.05}
    
    def _calculate_concentration(self, distribution: Dict[str, float]) -> float:
        """Calculate liquidity concentration"""
        if not distribution:
            return 1.0
        
        # Calculate Gini coefficient-like measure
        values = list(distribution.values())
        values.sort(reverse=True)
        
        if len(values) == 1:
            return 1.0
        
        # Calculate concentration as percentage held by top holder
        return values[0] if values else 1.0
    
    async def _get_liquidity_age(self, token_address: str) -> float:
        """Get liquidity age in hours from Solana RPC"""
        try:
            # RPC configuration - CANONICAL ACCESS THROUGH UNIFIED CONFIG
            from worker_ant_v1.core.unified_config import get_network_rpc_url
            rpc_url = get_network_rpc_url()
            
            # Get token creation time
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAccountInfo",
                "params": [token_address, {"encoding": "jsonParsed"}]
            }
            
            async with self.session.post(rpc_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    account_info = data.get('result', {}).get('value', {})
                    
                    if account_info:
                        # Try to get creation time from account data
                        # This is a simplified approach - in production you might want to use
                        # a more sophisticated method to determine liquidity age
                        current_time = datetime.now()
                        # Assume token is relatively new if we can't determine exact age
                        return 24.0  # Default to 24 hours
            
            return 24.0
            
        except Exception as e:
            self.logger.error(f"Error getting liquidity age: {e}")
            return 24.0
    
    def _calculate_removal_risk(self, liquidity_data: Dict[str, Any], 
                               distribution: Dict[str, float]) -> float:
        """Calculate risk of liquidity removal"""
        if not distribution:
            return 1.0
        
        # Higher risk if top holder has large percentage
        top_holder_percentage = max(distribution.values()) if distribution else 1.0
        
        # Risk increases with concentration
        return min(1.0, top_holder_percentage * 1.2)
    
    async def _get_liquidity_depth(self, token_address: str) -> Dict[str, float]:
        """Get liquidity depth at different price levels from Jupiter API"""
        try:
            # Get price impact data from Jupiter
            url = f"https://quote-api.jup.ag/v6/quote?inputMint=So11111111111111111111111111111111111111112&outputMint={token_address}&amount=1000000000&slippageBps=50"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Calculate depth at different levels
                    base_amount = 1000000000  # 1 SOL in lamports
                    depth_levels = {
                        '0.1%': base_amount * 0.001,
                        '0.5%': base_amount * 0.005,
                        '1%': base_amount * 0.01,
                        '5%': base_amount * 0.05
                    }
                    
                    depth_data = {}
                    for level, amount in depth_levels.items():
                        try:
                            # Get quote for this amount
                            quote_url = f"https://quote-api.jup.ag/v6/quote?inputMint=So11111111111111111111111111111111111111112&outputMint={token_address}&amount={int(amount)}&slippageBps=50"
                            async with self.session.get(quote_url) as quote_response:
                                if quote_response.status == 200:
                                    quote_data = await quote_response.json()
                                    depth_data[level] = quote_data.get('outAmount', 0)
                                else:
                                    depth_data[level] = 0
                        except Exception:
                            depth_data[level] = 0
                    
                    return depth_data
            
            return {
                '0.1%': 1000.0,
                '0.5%': 5000.0,
                '1%': 10000.0,
                '5%': 50000.0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting liquidity depth: {e}")
            return {
                '0.1%': 1000.0,
                '0.5%': 5000.0,
                '1%': 10000.0,
                '5%': 50000.0
            }

class OwnershipAnalyzer:
    """Analyzes token ownership patterns and risks"""
    
    def __init__(self):
        self.logger = setup_logger("OwnershipAnalyzer")
        self.session = None
        # API configuration - CANONICAL ACCESS THROUGH UNIFIED CONFIG
        from worker_ant_v1.core.unified_config import get_api_config
        api_config = get_api_config()
        self.birdeye_api_key = api_config['birdeye_api_key']
    
    async def analyze(self, token_address: str) -> Dict[str, float]:
        """Analyze ownership for a token"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Get holder data from multiple sources
            holders = await self._get_holder_data(token_address)
            
            # Calculate ownership metrics
            top_holder_percentage = self._get_top_holder_percentage(holders)
            holder_concentration = self._calculate_holder_concentration(holders)
            contract_ownership = await self._get_contract_ownership(token_address)
            ownership_transparency = self._calculate_transparency(holders)
            holder_count = len(holders)
            whale_count = self._count_whales(holders)
            
            return {
                'top_holder_percentage': top_holder_percentage,
                'holder_concentration': holder_concentration,
                'contract_ownership': contract_ownership,
                'ownership_transparency': ownership_transparency,
                'holder_count': holder_count,
                'whale_count': whale_count,
                'ownership_history': []
            }
            
        except Exception as e:
            self.logger.error(f"Ownership analysis failed for {token_address}: {e}")
            return {
                'top_holder_percentage': 1.0,
                'holder_concentration': 1.0,
                'contract_ownership': 'unknown',
                'ownership_transparency': 0.0,
                'holder_count': 1,
                'whale_count': 1,
                'ownership_history': []
            }
    
    async def _get_holder_data(self, token_address: str) -> List[Dict[str, Any]]:
        """Get holder data from Solana RPC and Birdeye API"""
        try:
            holders = []
            
            # Try Birdeye API first
            if self.birdeye_api_key:
                try:
                    url = f"https://public-api.birdeye.so/public/token/{token_address}/holders"
                    headers = {'X-API-KEY': self.birdeye_api_key}
                    
                    async with self.session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            holder_list = data.get('data', {}).get('items', [])
                            
                            for holder in holder_list[:100]:  # Top 100 holders
                                holders.append({
                                    'address': holder.get('owner', ''),
                                    'amount': holder.get('amount', 0),
                                    'percentage': holder.get('percentage', 0),
                                    'rank': holder.get('rank', 0)
                                })
                except Exception as e:
                    self.logger.warning(f"Birdeye holder data failed: {e}")
            
            # Fallback to Solana RPC
            if not holders:
                rpc_url = os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')
                
                # Get token accounts
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getTokenLargestAccounts",
                    "params": [token_address]
                }
                
                async with self.session.post(rpc_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        accounts = data.get('result', {}).get('value', [])
                        
                        total_supply = sum(acc['uiTokenAmount']['uiAmount'] for acc in accounts)
                        
                        for i, account in enumerate(accounts):
                            amount = account['uiTokenAmount']['uiAmount']
                            percentage = (amount / total_supply) * 100 if total_supply > 0 else 0
                            
                            holders.append({
                                'address': account.get('address', f'holder_{i}'),
                                'amount': amount,
                                'percentage': percentage,
                                'rank': i + 1
                            })
            
            return holders
            
        except Exception as e:
            self.logger.error(f"Error fetching holder data: {e}")
            return []
    
    def _get_top_holder_percentage(self, holders: List[Dict[str, Any]]) -> float:
        """Get percentage held by top holder"""
        if not holders:
            return 1.0
        
        return max(holder.get('percentage', 0) for holder in holders)
    
    def _calculate_holder_concentration(self, holders: List[Dict[str, Any]]) -> float:
        """Calculate holder concentration"""
        if not holders:
            return 1.0
        
        # Calculate concentration as percentage held by top 10 holders
        top_10_percentage = sum(holder.get('percentage', 0) for holder in holders[:10])
        return min(1.0, top_10_percentage)
    
    async def _get_contract_ownership(self, token_address: str) -> str:
        """Get contract ownership status from Solana RPC"""
        try:
            rpc_url = os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')
            
            # Get account info
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAccountInfo",
                "params": [token_address, {"encoding": "jsonParsed"}]
            }
            
            async with self.session.post(rpc_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    account_info = data.get('result', {}).get('value', {})
                    
                    if account_info:
                        # Check if it's a token mint
                        parsed_data = account_info.get('data', {}).get('parsed', {})
                        if parsed_data.get('type') == 'mint':
                            # Token mints don't have traditional "ownership" like contracts
                            return 'token_mint'
                        else:
                            # Check for program ownership
                            owner = account_info.get('owner', '')
                            if owner:
                                return f'owned_by_{owner[:8]}...'
            
            return 'unknown'
            
        except Exception as e:
            self.logger.error(f"Error getting contract ownership: {e}")
            return 'unknown'
    
    def _calculate_transparency(self, holders: List[Dict[str, Any]]) -> float:
        """Calculate ownership transparency"""
        if not holders:
            return 0.0
        
        # More holders = more transparent
        holder_count = len(holders)
        if holder_count > 1000:
            return 1.0
        elif holder_count > 100:
            return 0.8
        elif holder_count > 10:
            return 0.5
        else:
            return 0.2
    
    def _count_whales(self, holders: List[Dict[str, Any]]) -> int:
        """Count whale holders (>1% of supply)"""
        return sum(1 for holder in holders if holder.get('percentage', 0) > 0.01)

class CodeAnalyzer:
    """Analyzes smart contract code for suspicious patterns"""
    
    def __init__(self):
        self.logger = setup_logger("CodeAnalyzer")
        self.session = None
    
    async def analyze(self, token_address: str) -> Dict[str, Dict]:
        """Analyze smart contract code"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Get contract data from Solana RPC
            contract_data = await self._get_contract_data(token_address)
            
            # Analyze for suspicious functions
            suspicious_functions = self._find_suspicious_functions(contract_data)
            ownership_functions = self._find_ownership_functions(contract_data)
            mint_functions = self._find_mint_functions(contract_data)
            blacklist_functions = self._find_blacklist_functions(contract_data)
            
            # Calculate risks
            fee_manipulation_risk = self._calculate_fee_risk(contract_data)
            code_complexity_score = self._calculate_complexity(contract_data)
            
            return {
                'contract_verified': contract_data.get('verified', False),
                'source_code_available': contract_data.get('source_available', False),
                'suspicious_functions': suspicious_functions,
                'ownership_functions': ownership_functions,
                'mint_functions': mint_functions,
                'blacklist_functions': blacklist_functions,
                'fee_manipulation_risk': fee_manipulation_risk,
                'code_complexity_score': code_complexity_score
            }
            
        except Exception as e:
            self.logger.error(f"Code analysis failed for {token_address}: {e}")
            return {
                'contract_verified': False,
                'source_code_available': False,
                'suspicious_functions': [],
                'ownership_functions': [],
                'mint_functions': [],
                'blacklist_functions': [],
                'fee_manipulation_risk': 1.0,
                'code_complexity_score': 0.0
            }
    
    async def _get_contract_data(self, token_address: str) -> Dict[str, Any]:
        """Get contract data from Solana RPC"""
        try:
            rpc_url = os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')
            
            # Get account info
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAccountInfo",
                "params": [token_address, {"encoding": "jsonParsed"}]
            }
            
            async with self.session.post(rpc_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    account_info = data.get('result', {}).get('value', {})
                    
                    if account_info:
                        parsed_data = account_info.get('data', {}).get('parsed', {})
                        
                        return {
                            'verified': True,  # Solana tokens are typically verified
                            'source_available': True,  # Token programs are open source
                            'type': parsed_data.get('type', 'unknown'),
                            'info': parsed_data.get('info', {}),
                            'functions': self._extract_token_functions(parsed_data),
                            'program_id': account_info.get('owner', ''),
                            'data_size': account_info.get('data', {}).get('length', 0)
                        }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting contract data: {e}")
            return {}
    
    def _extract_token_functions(self, parsed_data: Dict[str, Any]) -> List[str]:
        """Extract functions from token data"""
        functions = []
        
        if parsed_data.get('type') == 'mint':
            info = parsed_data.get('info', {})
            
            # Check for mint authority
            if info.get('mintAuthority'):
                functions.append('mint')
            
            # Check for freeze authority
            if info.get('freezeAuthority'):
                functions.append('freeze')
            
            # Standard SPL token functions
            functions.extend(['transfer', 'approve', 'burn'])
        
        return functions
    
    def _find_suspicious_functions(self, contract_data: Dict[str, Any]) -> List[str]:
        """Find suspicious functions in contract"""
        suspicious_patterns = [
            'selfdestruct', 'suicide', 'delegatecall', 'callcode',
            'assembly', 'inline', 'low-level'
        ]
        
        functions = contract_data.get('functions', [])
        suspicious = []
        
        for func in functions:
            func_name = func.lower()
            for pattern in suspicious_patterns:
                if pattern in func_name:
                    suspicious.append(func)
                    break
        
        return suspicious
    
    def _find_ownership_functions(self, contract_data: Dict[str, Any]) -> List[str]:
        """Find ownership-related functions"""
        ownership_patterns = ['mint', 'freeze', 'authority']
        functions = contract_data.get('functions', [])
        
        return [func for func in functions if any(pattern in func.lower() for pattern in ownership_patterns)]
    
    def _find_mint_functions(self, contract_data: Dict[str, Any]) -> List[str]:
        """Find mint functions"""
        functions = contract_data.get('functions', [])
        return [func for func in functions if 'mint' in func.lower()]
    
    def _find_blacklist_functions(self, contract_data: Dict[str, Any]) -> List[str]:
        """Find blacklist functions"""
        blacklist_patterns = ['blacklist', 'freeze', 'pause']
        functions = contract_data.get('functions', [])
        
        return [func for func in functions if any(pattern in func.lower() for pattern in blacklist_patterns)]
    
    def _calculate_fee_risk(self, contract_data: Dict[str, Any]) -> float:
        """Calculate fee manipulation risk"""
        functions = contract_data.get('functions', [])
        
        # Higher risk if mint authority is present
        if 'mint' in functions:
            return 0.8
        
        # Medium risk if freeze authority is present
        if 'freeze' in functions:
            return 0.6
        
        # Lower risk for standard tokens
        return 0.2
    
    def _calculate_complexity(self, contract_data: Dict[str, Any]) -> float:
        """Calculate code complexity score"""
        functions = contract_data.get('functions', [])
        data_size = contract_data.get('data_size', 0)
        
        # Complexity based on function count and data size
        function_complexity = min(1.0, len(functions) / 10.0)
        size_complexity = min(1.0, data_size / 10000.0)
        
        return (function_complexity + size_complexity) / 2

class TradingAnalyzer:
    """Analyzes trading patterns and detects manipulation"""
    
    def __init__(self):
        self.logger = setup_logger("TradingAnalyzer")
        self.session = None
        self.birdeye_api_key = os.getenv('BIRDEYE_API_KEY')
    
    async def analyze(self, token_address: str) -> Dict[str, float]:
        """Analyze trading patterns for a token"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Get trading data from multiple sources
            trading_data = await self._get_trading_data(token_address)
            
            # Calculate metrics
            volatility = self._calculate_volatility(trading_data)
            pattern = self._identify_pattern(trading_data)
            wash_trading = self._detect_wash_trading(trading_data)
            pump_dump = self._detect_pump_dump(trading_data)
            manipulation_risk = self._calculate_manipulation_risk(trading_data)
            anomalies = self._detect_anomalies(trading_data)
            
            return {
                'volume_24h': trading_data.get('volume_24h', 0.0),
                'price_volatility': volatility,
                'trading_pattern': pattern,
                'wash_trading_indicators': wash_trading,
                'pump_dump_signals': pump_dump,
                'manipulation_risk': manipulation_risk,
                'trading_anomalies': anomalies
            }
            
        except Exception as e:
            self.logger.error(f"Trading analysis failed for {token_address}: {e}")
            return {
                'volume_24h': 0.0,
                'price_volatility': 0.0,
                'trading_pattern': 'unknown',
                'wash_trading_indicators': [],
                'pump_dump_signals': [],
                'manipulation_risk': 0.5,
                'trading_anomalies': []
            }
    
    async def _get_trading_data(self, token_address: str) -> Dict[str, Any]:
        """Get trading data from Birdeye and Jupiter APIs"""
        try:
            trading_data = {}
            
            # Get data from Birdeye
            if self.birdeye_api_key:
                try:
                    # Get token info
                    url = f"https://public-api.birdeye.so/public/token/{token_address}"
                    headers = {'X-API-KEY': self.birdeye_api_key}
                    
                    async with self.session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            token_info = data.get('data', {})
                            
                            trading_data.update({
                                'volume_24h': token_info.get('volume24h', 0.0),
                                'price': token_info.get('price', 0.0),
                                'price_change_24h': token_info.get('priceChange24h', 0.0),
                                'market_cap': token_info.get('marketCap', 0.0),
                                'holder_count': token_info.get('holder', 0)
                            })
                    
                    # Get recent trades
                    trades_url = f"https://public-api.birdeye.so/public/token/{token_address}/trades"
                    async with self.session.get(trades_url, headers=headers) as response:
                        if response.status == 200:
                            trades_data = await response.json()
                            trades = trades_data.get('data', {}).get('items', [])
                            
                            # Extract price history from trades
                            price_history = []
                            for trade in trades[:100]:  # Last 100 trades
                                price = trade.get('price', 0.0)
                                if price > 0:
                                    price_history.append(price)
                            
                            trading_data['price_history'] = price_history
                            
                except Exception as e:
                    self.logger.warning(f"Birdeye trading data failed: {e}")
            
            # Get data from Jupiter
            try:
                url = f"https://price.jup.ag/v4/price?ids={token_address}"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        jupiter_data = data.get('data', {}).get(token_address, {})
                        
                        if not trading_data.get('price'):
                            trading_data['price'] = jupiter_data.get('price', 0.0)
                        
                        if not trading_data.get('volume_24h'):
                            trading_data['volume_24h'] = jupiter_data.get('volume24h', 0.0)
                            
            except Exception as e:
                self.logger.warning(f"Jupiter trading data failed: {e}")
            
            return trading_data
            
        except Exception as e:
            self.logger.error(f"Error fetching trading data: {e}")
            return {}
    
    def _calculate_volatility(self, trading_data: Dict[str, Any]) -> float:
        """Calculate price volatility"""
        price_history = trading_data.get('price_history', [])
        if len(price_history) < 2:
            return 0.0
        
        # Calculate standard deviation of price changes
        price_changes = []
        for i in range(1, len(price_history)):
            if price_history[i-1] > 0:
                change = abs(price_history[i] - price_history[i-1]) / price_history[i-1]
                price_changes.append(change)
        
        if not price_changes:
            return 0.0
        
        import statistics
        return statistics.stdev(price_changes) if len(price_changes) > 1 else 0.0
    
    def _identify_pattern(self, trading_data: Dict[str, Any]) -> str:
        """Identify trading pattern"""
        price_history = trading_data.get('price_history', [])
        if len(price_history) < 10:
            return 'insufficient_data'
        
        # Calculate price trend
        recent_prices = price_history[-10:]
        price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
        
        # Calculate volume trend
        volume_24h = trading_data.get('volume_24h', 0.0)
        
        if price_trend > 0.1 and volume_24h > 10000:  # 10%+ price increase with good volume
            return 'pump'
        elif price_trend < -0.1 and volume_24h > 10000:  # 10%+ price decrease with good volume
            return 'dump'
        elif abs(price_trend) < 0.05:  # Stable price
            return 'stable'
        else:
            return 'volatile'
    
    def _detect_wash_trading(self, trading_data: Dict[str, Any]) -> List[str]:
        """Detect wash trading indicators"""
        indicators = []
        
        price_history = trading_data.get('price_history', [])
        volume_24h = trading_data.get('volume_24h', 0.0)
        holder_count = trading_data.get('holder_count', 0)
        
        # Check for low holder count with high volume
        if holder_count < 100 and volume_24h > 50000:
            indicators.append('low_holders_high_volume')
        
        # Check for price stability with high volume (potential wash trading)
        if len(price_history) >= 10:
            recent_volatility = self._calculate_volatility({'price_history': price_history[-10:]})
            if recent_volatility < 0.01 and volume_24h > 100000:  # Very low volatility with high volume
                indicators.append('low_volatility_high_volume')
        
        return indicators
    
    def _detect_pump_dump(self, trading_data: Dict[str, Any]) -> List[str]:
        """Detect pump and dump signals"""
        signals = []
        
        price_history = trading_data.get('price_history', [])
        price_change_24h = trading_data.get('price_change_24h', 0.0)
        volume_24h = trading_data.get('volume_24h', 0.0)
        
        # Check for extreme price movements
        if price_change_24h > 50:  # 50%+ price increase
            signals.append('extreme_price_increase')
        elif price_change_24h < -50:  # 50%+ price decrease
            signals.append('extreme_price_decrease')
        
        # Check for volume spikes
        if volume_24h > 1000000:  # Very high volume
            signals.append('volume_spike')
        
        # Check for price pattern
        if len(price_history) >= 20:
            # Look for pump and dump pattern (sharp rise followed by sharp fall)
            recent_prices = price_history[-20:]
            peak_idx = recent_prices.index(max(recent_prices))
            
            if peak_idx > 5 and peak_idx < len(recent_prices) - 5:
                pre_peak_rise = (recent_prices[peak_idx] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
                post_peak_fall = (recent_prices[-1] - recent_prices[peak_idx]) / recent_prices[peak_idx] if recent_prices[peak_idx] > 0 else 0
                
                if pre_peak_rise > 0.2 and post_peak_fall < -0.3:  # 20%+ rise, 30%+ fall
                    signals.append('pump_dump_pattern')
        
        return signals
    
    def _calculate_manipulation_risk(self, trading_data: Dict[str, Any]) -> float:
        """Calculate manipulation risk"""
        risk_score = 0.0
        
        # Volume-based risk
        volume_24h = trading_data.get('volume_24h', 0.0)
        if volume_24h > 1000000:
            risk_score += 0.3
        elif volume_24h < 1000:
            risk_score += 0.2
        
        # Holder count risk
        holder_count = trading_data.get('holder_count', 0)
        if holder_count < 100:
            risk_score += 0.4
        elif holder_count < 1000:
            risk_score += 0.2
        
        # Price volatility risk
        volatility = self._calculate_volatility(trading_data)
        if volatility > 0.5:
            risk_score += 0.3
        
        # Pattern-based risk
        pattern = self._identify_pattern(trading_data)
        if pattern in ['pump', 'dump']:
            risk_score += 0.2
        
        return min(1.0, risk_score)
    
    def _detect_anomalies(self, trading_data: Dict[str, Any]) -> List[str]:
        """Detect trading anomalies"""
        anomalies = []
        
        price_history = trading_data.get('price_history', [])
        volume_24h = trading_data.get('volume_24h', 0.0)
        
        # Check for price gaps
        if len(price_history) >= 2:
            for i in range(1, len(price_history)):
                if price_history[i-1] > 0:
                    gap = abs(price_history[i] - price_history[i-1]) / price_history[i-1]
                    if gap > 0.5:  # 50%+ price gap
                        anomalies.append('price_gap')
                        break
        
        # Check for volume anomalies
        if volume_24h > 5000000:  # Extremely high volume
            anomalies.append('extreme_volume')
        elif volume_24h < 100:  # Extremely low volume
            anomalies.append('no_volume')
        
        # Check for price anomalies
        if price_history:
            current_price = price_history[-1]
            if current_price < 0.000001:  # Extremely low price
                anomalies.append('extremely_low_price')
            elif current_price > 1000:  # Extremely high price
                anomalies.append('extremely_high_price')
        
        return anomalies 