"""
FastAPI Dependencies
Handles database sessions, authentication, and service injection
"""
from typing import Optional, Generator
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
try:
    import jwt
except ImportError:
    jwt = None
from datetime import datetime, timedelta

from app.models.database import get_db, db_manager
from app.services.compliance_engine import ComplianceEngine
from app.services.policy_parser import PolicyParser
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStoreService
from app.config.settings import get_settings
from app.utils.logger import get_logger, compliance_logger
from app.utils.exceptions import AuthenticationException, ServiceUnavailable

settings = get_settings()
logger = get_logger(__name__)
security = HTTPBearer(auto_error=False)

# Global service instances (initialized lazily)
_llm_service: Optional[LLMService] = None
_vector_service: Optional[VectorStoreService] = None


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """
    Get current user from JWT token (placeholder implementation)
    In production, implement proper JWT validation
    """
    if not credentials:
        return "anonymous"
    
    try:
        # In production, validate JWT token here
        token = credentials.credentials
        
        # Placeholder validation
        if token == "admin-token":
            return "admin"
        elif token == "user-token":
            return "user"
        else:
            # For development, accept any token
            return f"user_{token[:8]}"
    
    except Exception as e:
        logger.warning(f"Token validation failed: {e}")
        return "anonymous"


def get_compliance_engine(db: Session = Depends(get_db)) -> ComplianceEngine:
    """Get compliance engine instance"""
    try:
        return ComplianceEngine(db)
    except Exception as e:
        logger.error(f"Failed to initialize compliance engine: {e}")
        raise ServiceUnavailable("compliance_engine", str(e))


def get_policy_parser(db: Session = Depends(get_db)) -> PolicyParser:
    """Get policy parser instance"""
    try:
        return PolicyParser(db)
    except Exception as e:
        logger.error(f"Failed to initialize policy parser: {e}")
        raise ServiceUnavailable("policy_parser", str(e))


def get_llm_service() -> LLMService:
    """Get LLM service instance (singleton)"""
    global _llm_service
    
    if _llm_service is None:
        try:
            _llm_service = LLMService()
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            raise ServiceUnavailable("llm_service", str(e))
    
    return _llm_service


def get_vector_service() -> VectorStoreService:
    """Get vector store service instance (singleton)"""
    global _vector_service
    
    if _vector_service is None:
        try:
            _vector_service = VectorStoreService()
        except Exception as e:
            logger.error(f"Failed to initialize vector store service: {e}")
            raise ServiceUnavailable("vector_store", str(e))
    
    return _vector_service


def get_optional_llm_service() -> Optional[LLMService]:
    """Get LLM service instance without raising exception if unavailable"""
    try:
        return get_llm_service()
    except ServiceUnavailable:
        return None


def get_optional_vector_service() -> Optional[VectorStoreService]:
    """Get vector store service instance without raising exception if unavailable"""
    try:
        return get_vector_service()
    except ServiceUnavailable:
        return None


class RateLimiter:
    """Simple rate limiter for API endpoints"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client"""
        now = datetime.now()
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if (now - req_time).total_seconds() < self.window_seconds
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True


# Global rate limiter instances
compliance_check_limiter = RateLimiter(max_requests=50, window_seconds=3600)  # 50 per hour
file_upload_limiter = RateLimiter(max_requests=20, window_seconds=3600)  # 20 per hour
llm_request_limiter = RateLimiter(max_requests=30, window_seconds=3600)  # 30 per hour


def rate_limit_compliance_check(current_user: str = Depends(get_current_user)):
    """Rate limit for compliance check endpoints"""
    if not compliance_check_limiter.is_allowed(current_user):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded for compliance checks. Try again later."
        )
    return current_user


def rate_limit_file_upload(current_user: str = Depends(get_current_user)):
    """Rate limit for file upload endpoints"""
    if not file_upload_limiter.is_allowed(current_user):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded for file uploads. Try again later."
        )
    return current_user


def rate_limit_llm_requests(current_user: str = Depends(get_current_user)):
    """Rate limit for LLM-powered endpoints"""
    if not llm_request_limiter.is_allowed(current_user):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded for LLM requests. Try again later."
        )
    return current_user


def log_api_request(request_info: dict):
    """Log API request for audit purposes"""
    compliance_logger.log_api_request(
        method=request_info.get("method", "unknown"),
        path=request_info.get("path", "unknown"),
        status_code=request_info.get("status_code", 0),
        duration_ms=request_info.get("duration_ms", 0),
        user_id=request_info.get("user_id")
    )


def validate_admin_access(current_user: str = Depends(get_current_user)):
    """Validate admin access for sensitive operations"""
    if current_user not in ["admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required for this operation"
        )
    return current_user


def validate_file_upload_permissions(current_user: str = Depends(get_current_user)):
    """Validate file upload permissions"""
    # In production, implement proper role-based access control
    allowed_roles = ["admin", "compliance_officer", "analyst"]
    
    if current_user == "anonymous":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for file upload"
        )
    
    return current_user


def check_system_health() -> dict:
    """Check overall system health"""
    health_status = {
        "database": "unknown",
        "llm_service": "unknown",
        "vector_store": "unknown",
        "overall_status": "unknown"
    }
    
    # Check database
    try:
        db = db_manager.get_session()
        db.execute("SELECT 1")
        db.close()
        health_status["database"] = "healthy"
    except Exception as e:
        health_status["database"] = f"unhealthy: {str(e)}"
    
    # Check LLM service
    try:
        llm_service = get_optional_llm_service()
        if llm_service:
            status = llm_service.get_service_status()
            health_status["llm_service"] = status.get("status", "unknown")
        else:
            health_status["llm_service"] = "unavailable"
    except Exception as e:
        health_status["llm_service"] = f"error: {str(e)}"
    
    # Check vector store
    try:
        vector_service = get_optional_vector_service()
        if vector_service and vector_service.is_available():
            health_status["vector_store"] = "healthy"
        else:
            health_status["vector_store"] = "unavailable"
    except Exception as e:
        health_status["vector_store"] = f"error: {str(e)}"
    
    # Determine overall status
    if all(status in ["healthy", "unavailable"] for status in health_status.values() if status != "unknown"):
        if health_status["database"] == "healthy":
            health_status["overall_status"] = "healthy"
        else:
            health_status["overall_status"] = "degraded"
    else:
        health_status["overall_status"] = "unhealthy"
    
    return health_status