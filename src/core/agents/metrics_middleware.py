"""
Metrics Middleware for AntBot API

This module provides middleware for collecting metrics from API requests.
"""

import time
from functools import wraps
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from src.utils.monitoring.metrics import record_api_request, record_error

class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting Prometheus metrics from FastAPI requests"""
    
    async def dispatch(self, request: Request, call_next):
        """
        Process the request and record metrics
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response from the next handler
        """
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Extract endpoint and method
            endpoint = request.url.path
            method = request.method
            status = str(response.status_code)
            
            # Record the API request metrics
            record_api_request(endpoint, method, status, duration)
            
            return response
            
        except Exception as exc:
            duration = time.time() - start_time
            
            # Record error
            record_error("api", type(exc).__name__)
            
            # Record failed API request
            endpoint = request.url.path
            method = request.method
            status = "500"  # Assume 500 for exceptions
            record_api_request(endpoint, method, status, duration)
            
            # Re-raise the exception for proper handling
            raise

def transaction_metrics(f):
    """
    Decorator for recording transaction metrics
    
    Args:
        f: The function to decorate
        
    Returns:
        Wrapped function that records metrics
    """
    from src.utils.monitoring.metrics import record_transaction
    
    @wraps(f)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Extract transaction parameters
        tx_type = kwargs.get('type', 'unknown')
        market = kwargs.get('market', 'unknown')
        
        try:
            # Call the original function
            result = await f(*args, **kwargs)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Extract amount from result or kwargs
            amount = getattr(result, 'amount', None)
            if amount is None:
                amount = kwargs.get('amount', 0.0)
            
            # Record successful transaction
            record_transaction(tx_type, 'success', market, amount, duration)
            
            return result
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Extract amount from kwargs
            amount = kwargs.get('amount', 0.0)
            
            # Record failed transaction
            record_transaction(tx_type, 'failure', market, amount, duration)
            
            # Re-raise the exception
            raise
    
    return wrapper 