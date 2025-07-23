"""
PRODUCTION-GRADE SECRETS MANAGER
================================

Professional secrets management system that abstracts away specific providers.
Reference implementation uses HashiCorp Vault for secure secret storage and retrieval.

Features:
- Provider abstraction for easy switching between secret backends
- Built-in caching with TTL and memory limits
- Exponential backoff retry logic with circuit breaker
- Comprehensive error handling and logging
- Support for secret rotation and invalidation
- Development mode fallback for local testing

Supported Providers:
- HashiCorp Vault (production reference)
- Environment variables (development fallback)
- Azure Key Vault (future extension)
- AWS Secrets Manager (future extension)
"""

import asyncio
import base64
import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import aiohttp
from cryptography.fernet import Fernet

from worker_ant_v1.utils.logger import get_logger


class SecretProvider(Enum):
    """Supported secret providers"""
    VAULT = "vault"
    ENVIRONMENT = "environment"
    AZURE_KEYVAULT = "azure_keyvault"
    AWS_SECRETS = "aws_secrets"


class SecretStatus(Enum):
    """Secret status tracking"""
    VALID = "valid"
    EXPIRED = "expired"
    INVALID = "invalid"
    MISSING = "missing"
    ERROR = "error"


@dataclass
class SecretMetadata:
    """Metadata for cached secrets"""
    created_at: datetime
    expires_at: Optional[datetime]
    last_accessed: datetime
    access_count: int = 0
    status: SecretStatus = SecretStatus.VALID
    provider: str = ""
    version: Optional[str] = None


@dataclass
class CachedSecret:
    """Cached secret with metadata"""
    value: str
    metadata: SecretMetadata
    encrypted: bool = False


@dataclass
class SecretsConfig:
    """Configuration for secrets manager"""
    provider: SecretProvider = SecretProvider.VAULT
    cache_ttl_seconds: int = 3600  # 1 hour
    max_cache_size: int = 1000
    retry_attempts: int = 3
    retry_backoff_factor: float = 2.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 300  # 5 minutes
    enable_encryption: bool = True
    
    # Vault-specific configuration
    vault_url: str = "http://localhost:8200"
    vault_token: Optional[str] = None
    vault_mount_path: str = "secret"
    vault_timeout: int = 30
    
    # Development fallback
    allow_env_fallback: bool = True
    env_prefix: str = "ANTBOT_"


class SecretsProvider(ABC):
    """Abstract base class for secret providers"""
    
    @abstractmethod
    async def get_secret(self, key: str, path: Optional[str] = None) -> str:
        """Retrieve a secret by key"""
        pass
    
    @abstractmethod
    async def put_secret(self, key: str, value: str, path: Optional[str] = None) -> bool:
        """Store a secret"""
        pass
    
    @abstractmethod
    async def delete_secret(self, key: str, path: Optional[str] = None) -> bool:
        """Delete a secret"""
        pass
    
    @abstractmethod
    async def list_secrets(self, path: Optional[str] = None) -> List[str]:
        """List available secrets"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check provider health"""
        pass


class VaultProvider(SecretsProvider):
    """HashiCorp Vault secrets provider"""
    
    def __init__(self, config: SecretsConfig):
        self.config = config
        self.logger = get_logger("VaultProvider")
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = config.vault_url.rstrip('/')
        self.headers = {
            "X-Vault-Token": config.vault_token or os.getenv("VAULT_TOKEN", ""),
            "Content-Type": "application/json"
        }
        
    async def _ensure_session(self):
        """Ensure aiohttp session is available"""
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.vault_timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.headers
            )
    
    async def _make_request(self, method: str, path: str, data: Optional[Dict] = None) -> Dict:
        """Make authenticated request to Vault"""
        await self._ensure_session()
        
        url = urljoin(f"{self.base_url}/", path.lstrip('/'))
        
        try:
            async with self.session.request(method, url, json=data) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    raise KeyError(f"Secret not found: {path}")
                else:
                    text = await response.text()
                    raise Exception(f"Vault request failed: {response.status} - {text}")
                    
        except aiohttp.ClientError as e:
            raise Exception(f"Vault connection failed: {e}")
    
    async def get_secret(self, key: str, path: Optional[str] = None) -> str:
        """Retrieve secret from Vault"""
        secret_path = path or "antbot/config"
        vault_path = f"v1/{self.config.vault_mount_path}/data/{secret_path}"
        
        try:
            response = await self._make_request("GET", vault_path)
            data = response.get("data", {}).get("data", {})
            
            if key not in data:
                raise KeyError(f"Secret key '{key}' not found in path '{secret_path}'")
            
            return data[key]
            
        except Exception as e:
            self.logger.error(f"Failed to get secret '{key}' from Vault: {e}")
            raise
    
    async def put_secret(self, key: str, value: str, path: Optional[str] = None) -> bool:
        """Store secret in Vault"""
        secret_path = path or "antbot/config"
        vault_path = f"v1/{self.config.vault_mount_path}/data/{secret_path}"
        
        try:
            # Get existing secrets first to avoid overwriting
            existing_data = {}
            try:
                response = await self._make_request("GET", vault_path)
                existing_data = response.get("data", {}).get("data", {})
            except KeyError:
                pass  # Path doesn't exist yet
            
            # Update with new secret
            existing_data[key] = value
            payload = {"data": existing_data}
            
            await self._make_request("POST", vault_path, payload)
            self.logger.info(f"Successfully stored secret '{key}' in Vault")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store secret '{key}' in Vault: {e}")
            return False
    
    async def delete_secret(self, key: str, path: Optional[str] = None) -> bool:
        """Delete secret from Vault"""
        secret_path = path or "antbot/config"
        vault_path = f"v1/{self.config.vault_mount_path}/data/{secret_path}"
        
        try:
            # Get existing secrets
            response = await self._make_request("GET", vault_path)
            data = response.get("data", {}).get("data", {})
            
            if key in data:
                del data[key]
                payload = {"data": data}
                await self._make_request("POST", vault_path, payload)
                self.logger.info(f"Successfully deleted secret '{key}' from Vault")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete secret '{key}' from Vault: {e}")
            return False
    
    async def list_secrets(self, path: Optional[str] = None) -> List[str]:
        """List secrets in Vault path"""
        secret_path = path or "antbot/config"
        vault_path = f"v1/{self.config.vault_mount_path}/data/{secret_path}"
        
        try:
            response = await self._make_request("GET", vault_path)
            data = response.get("data", {}).get("data", {})
            return list(data.keys())
            
        except KeyError:
            return []  # Path doesn't exist
        except Exception as e:
            self.logger.error(f"Failed to list secrets from Vault: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check Vault health"""
        try:
            await self._make_request("GET", "v1/sys/health")
            return True
        except Exception:
            return False
    
    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()


class EnvironmentProvider(SecretsProvider):
    """Environment variable fallback provider"""
    
    def __init__(self, config: SecretsConfig):
        self.config = config
        self.logger = get_logger("EnvironmentProvider")
    
    def _get_env_key(self, key: str) -> str:
        """Convert key to environment variable name"""
        return f"{self.config.env_prefix}{key.upper()}"
    
    async def get_secret(self, key: str, path: Optional[str] = None) -> str:
        """Get secret from environment variables"""
        env_key = self._get_env_key(key)
        value = os.getenv(env_key)
        
        if value is None:
            raise KeyError(f"Environment variable '{env_key}' not found")
        
        return value
    
    async def put_secret(self, key: str, value: str, path: Optional[str] = None) -> bool:
        """Store secret in environment (not persistent)"""
        env_key = self._get_env_key(key)
        os.environ[env_key] = value
        self.logger.warning(f"Secret '{key}' stored in environment (not persistent)")
        return True
    
    async def delete_secret(self, key: str, path: Optional[str] = None) -> bool:
        """Delete secret from environment"""
        env_key = self._get_env_key(key)
        if env_key in os.environ:
            del os.environ[env_key]
        return True
    
    async def list_secrets(self, path: Optional[str] = None) -> List[str]:
        """List environment variables with prefix"""
        prefix = self.config.env_prefix
        keys = []
        for env_key in os.environ:
            if env_key.startswith(prefix):
                key = env_key[len(prefix):].lower()
                keys.append(key)
        return keys
    
    async def health_check(self) -> bool:
        """Environment provider is always healthy"""
        return True


class SecretsManager:
    """Production-grade secrets manager with caching and retry logic"""
    
    def __init__(self, config: Optional[SecretsConfig] = None):
        self.config = config or SecretsConfig()
        self.logger = get_logger("SecretsManager")
        
        # Cache management
        self.cache: Dict[str, CachedSecret] = {}
        self.cache_stats = {"hits": 0, "misses": 0, "errors": 0}
        
        # Circuit breaker state
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure: Optional[datetime] = None
        self.circuit_breaker_open = False
        
        # Encryption for cache
        self.encryption_key = None
        if self.config.enable_encryption:
            self.encryption_key = Fernet.generate_key()
            self.cipher_suite = Fernet(self.encryption_key)
        
        # Initialize provider
        self.provider = self._create_provider()
        self.fallback_provider = EnvironmentProvider(self.config) if self.config.allow_env_fallback else None
    
    def _create_provider(self) -> SecretsProvider:
        """Create the appropriate secrets provider"""
        if self.config.provider == SecretProvider.VAULT:
            return VaultProvider(self.config)
        elif self.config.provider == SecretProvider.ENVIRONMENT:
            return EnvironmentProvider(self.config)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def _encrypt_value(self, value: str) -> str:
        """Encrypt a value for cache storage"""
        if not self.config.enable_encryption:
            return value
        return self.cipher_suite.encrypt(value.encode()).decode()
    
    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a value from cache"""
        if not self.config.enable_encryption:
            return encrypted_value
        return self.cipher_suite.decrypt(encrypted_value.encode()).decode()
    
    def _generate_cache_key(self, key: str, path: Optional[str] = None) -> str:
        """Generate cache key"""
        full_key = f"{path or 'default'}:{key}"
        return hashlib.sha256(full_key.encode()).hexdigest()[:16]
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open"""
        if not self.circuit_breaker_open:
            return False
        
        if self.circuit_breaker_last_failure:
            time_since_failure = datetime.now() - self.circuit_breaker_last_failure
            if time_since_failure.total_seconds() > self.config.circuit_breaker_timeout:
                self.circuit_breaker_open = False
                self.circuit_breaker_failures = 0
                self.logger.info("Circuit breaker reset - attempting provider reconnection")
        
        return self.circuit_breaker_open
    
    def _record_failure(self):
        """Record a provider failure for circuit breaker"""
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = datetime.now()
        
        if self.circuit_breaker_failures >= self.config.circuit_breaker_threshold:
            self.circuit_breaker_open = True
            self.logger.warning(
                f"Circuit breaker opened after {self.circuit_breaker_failures} failures"
            )
    
    def _record_success(self):
        """Record successful operation"""
        if self.circuit_breaker_failures > 0:
            self.circuit_breaker_failures = 0
            self.logger.info("Provider recovered - failures reset")
    
    async def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Get secret from cache if valid"""
        if cache_key not in self.cache:
            return None
        
        cached = self.cache[cache_key]
        now = datetime.now()
        
        # Check expiration
        if cached.metadata.expires_at and now > cached.metadata.expires_at:
            del self.cache[cache_key]
            return None
        
        # Update access stats
        cached.metadata.last_accessed = now
        cached.metadata.access_count += 1
        self.cache_stats["hits"] += 1
        
        # Decrypt and return
        if cached.encrypted:
            return self._decrypt_value(cached.value)
        return cached.value
    
    def _add_to_cache(self, cache_key: str, value: str, key: str):
        """Add secret to cache"""
        # Clean cache if at capacity
        if len(self.cache) >= self.config.max_cache_size:
            self._cleanup_cache()
        
        encrypted_value = self._encrypt_value(value) if self.config.enable_encryption else value
        expires_at = datetime.now() + timedelta(seconds=self.config.cache_ttl_seconds)
        
        metadata = SecretMetadata(
            created_at=datetime.now(),
            expires_at=expires_at,
            last_accessed=datetime.now(),
            provider=self.config.provider.value
        )
        
        self.cache[cache_key] = CachedSecret(
            value=encrypted_value,
            metadata=metadata,
            encrypted=self.config.enable_encryption
        )
    
    def _cleanup_cache(self):
        """Remove oldest and expired entries from cache"""
        now = datetime.now()
        
        # Remove expired entries first
        expired_keys = [
            k for k, v in self.cache.items()
            if v.metadata.expires_at and now > v.metadata.expires_at
        ]
        for key in expired_keys:
            del self.cache[key]
        
        # If still at capacity, remove least recently used
        if len(self.cache) >= self.config.max_cache_size:
            sorted_cache = sorted(
                self.cache.items(),
                key=lambda x: x[1].metadata.last_accessed
            )
            
            # Remove oldest 20% of entries
            remove_count = max(1, len(sorted_cache) // 5)
            for key, _ in sorted_cache[:remove_count]:
                del self.cache[key]
    
    async def _retry_with_backoff(self, operation, *args, **kwargs):
        """Execute operation with exponential backoff retry"""
        last_exception = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                result = await operation(*args, **kwargs)
                self._record_success()
                return result
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.retry_attempts - 1:
                    delay = self.config.retry_backoff_factor ** attempt
                    await asyncio.sleep(delay)
        
        self._record_failure()
        raise last_exception
    
    async def get_secret(self, key: str, path: Optional[str] = None, use_cache: bool = True) -> str:
        """
        Get secret with caching and fallback support
        
        Args:
            key: Secret key name
            path: Optional path for secret organization
            use_cache: Whether to use cache
            
        Returns:
            Secret value
            
        Raises:
            KeyError: Secret not found
            Exception: Provider errors
        """
        cache_key = self._generate_cache_key(key, path)
        
        # Try cache first
        if use_cache:
            cached_value = await self._get_from_cache(cache_key)
            if cached_value is not None:
                return cached_value
        
        self.cache_stats["misses"] += 1
        
        # Check circuit breaker
        if self._is_circuit_breaker_open():
            if self.fallback_provider:
                self.logger.warning(f"Circuit breaker open - using fallback for '{key}'")
                try:
                    return await self.fallback_provider.get_secret(key, path)
                except Exception as e:
                    self.logger.error(f"Fallback provider failed: {e}")
            raise Exception("Primary provider unavailable and no fallback configured")
        
        try:
            # Get from primary provider with retry
            value = await self._retry_with_backoff(self.provider.get_secret, key, path)
            
            # Cache the result
            if use_cache:
                self._add_to_cache(cache_key, value, key)
            
            return value
            
        except Exception as e:
            self.cache_stats["errors"] += 1
            
            # Try fallback if available
            if self.fallback_provider:
                self.logger.warning(f"Primary provider failed for '{key}' - trying fallback")
                try:
                    return await self.fallback_provider.get_secret(key, path)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback also failed: {fallback_error}")
            
            raise e
    
    async def put_secret(self, key: str, value: str, path: Optional[str] = None) -> bool:
        """Store secret in provider"""
        cache_key = self._generate_cache_key(key, path)
        
        try:
            success = await self._retry_with_backoff(self.provider.put_secret, key, value, path)
            
            if success:
                # Update cache
                self._add_to_cache(cache_key, value, key)
                self.logger.info(f"Successfully stored secret '{key}'")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to store secret '{key}': {e}")
            return False
    
    async def delete_secret(self, key: str, path: Optional[str] = None) -> bool:
        """Delete secret from provider and cache"""
        cache_key = self._generate_cache_key(key, path)
        
        try:
            success = await self._retry_with_backoff(self.provider.delete_secret, key, path)
            
            if success and cache_key in self.cache:
                del self.cache[cache_key]
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete secret '{key}': {e}")
            return False
    
    async def list_secrets(self, path: Optional[str] = None) -> List[str]:
        """List available secrets"""
        try:
            return await self._retry_with_backoff(self.provider.list_secrets, path)
        except Exception as e:
            self.logger.error(f"Failed to list secrets: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health_info = {
            "provider": self.config.provider.value,
            "primary_healthy": False,
            "fallback_healthy": False,
            "cache_size": len(self.cache),
            "cache_stats": self.cache_stats.copy(),
            "circuit_breaker_open": self.circuit_breaker_open,
            "circuit_breaker_failures": self.circuit_breaker_failures
        }
        
        try:
            health_info["primary_healthy"] = await self.provider.health_check()
        except Exception as e:
            self.logger.error(f"Primary provider health check failed: {e}")
        
        if self.fallback_provider:
            try:
                health_info["fallback_healthy"] = await self.fallback_provider.health_check()
            except Exception as e:
                self.logger.error(f"Fallback provider health check failed: {e}")
        
        return health_info
    
    def invalidate_cache(self, key: Optional[str] = None, path: Optional[str] = None):
        """Invalidate cache entries"""
        if key:
            cache_key = self._generate_cache_key(key, path)
            if cache_key in self.cache:
                del self.cache[cache_key]
        else:
            self.cache.clear()
        
        self.logger.info(f"Cache invalidated: {key or 'all entries'}")
    
    async def shutdown(self):
        """Shutdown the secrets manager"""
        self.logger.info("Shutting down secrets manager...")
        
        if hasattr(self.provider, 'close'):
            await self.provider.close()
        
        self.cache.clear()


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_config() -> SecretsConfig:
    """Get secrets configuration from environment"""
    return SecretsConfig(
        provider=SecretProvider(os.getenv("SECRETS_PROVIDER", "vault")),
        vault_url=os.getenv("VAULT_URL", "http://localhost:8200"),
        vault_token=os.getenv("VAULT_TOKEN"),
        vault_mount_path=os.getenv("VAULT_MOUNT_PATH", "secret"),
        cache_ttl_seconds=int(os.getenv("SECRETS_CACHE_TTL", "3600")),
        max_cache_size=int(os.getenv("SECRETS_CACHE_SIZE", "1000")),
        allow_env_fallback=os.getenv("SECRETS_ALLOW_ENV_FALLBACK", "true").lower() == "true",
        enable_encryption=os.getenv("SECRETS_ENABLE_CACHE_ENCRYPTION", "true").lower() == "true"
    )


async def get_secrets_manager() -> SecretsManager:
    """Get or create global secrets manager instance"""
    global _secrets_manager
    
    if _secrets_manager is None:
        config = get_secrets_config()
        _secrets_manager = SecretsManager(config)
        
        # Verify provider health
        health = await _secrets_manager.health_check()
        if not health["primary_healthy"] and not health["fallback_healthy"]:
            raise Exception("No healthy secrets provider available")
    
    return _secrets_manager


async def shutdown_secrets_manager():
    """Shutdown global secrets manager"""
    global _secrets_manager
    
    if _secrets_manager:
        await _secrets_manager.shutdown()
        _secrets_manager = None 