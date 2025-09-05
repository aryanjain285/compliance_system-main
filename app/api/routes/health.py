"""
Health Check and System Status API Routes
"""
from fastapi import APIRouter, Depends
from datetime import datetime
from typing import Dict, Any

from app.api.dependencies import (
    get_optional_llm_service, get_optional_vector_service, 
    check_system_health
)
from app.models.database import db_manager
from app.schemas import HealthCheckResponse, ServiceStatus
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStoreService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("")
async def health_check():
    """
    Comprehensive health check endpoint
    Returns detailed status of all system components
    """
    try:
        # Get system health
        health_status = check_system_health()
        
        # Get basic stats
        stats = await _get_system_stats()
        
        return {
            "status": health_status["overall_status"],
            "timestamp": datetime.now(),
            "database": health_status["database"],
            "llm_service": health_status.get("llm_service", "unknown"),
            "vector_store": health_status.get("vector_store", "unknown"),
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now(),
            "database": "error",
            "error": str(e)
        }


# @router.get("/detailed")
# async def detailed_health_check():
#     """
#     Detailed health check with comprehensive system information
#     """
#     try:
#         # Basic health
#         basic_health = await health_check()
        
#         # Additional details
#         additional_details = {
#             "system_info": {
#                 "python_version": "3.8+",
#                 "fastapi_version": "0.104+",
#                 "database_migrations": "up_to_date"
#             },
#             "performance_metrics": await _get_performance_metrics(),
#             "resource_usage": await _get_resource_usage(),
#             "recent_errors": await _get_recent_errors()
#         }
        
#         return {
#             **basic_health.dict(),
#             "additional_details": additional_details
#         }
        
#     except Exception as e:
#         logger.error(f"Detailed health check failed: {e}")
#         return {"error": str(e), "timestamp": datetime.now()}


# @router.get("/readiness")
# async def readiness_check():
#     """
#     Kubernetes-style readiness probe
#     Returns 200 if system is ready to accept traffic
#     """
#     try:
#         health_status = check_system_health()
        
#         if health_status["overall_status"] in ["healthy", "degraded"]:
#             return {
#                 "status": "ready",
#                 "timestamp": datetime.now().isoformat()
#             }
#         else:
#             return {
#                 "status": "not_ready", 
#                 "reason": health_status["overall_status"],
#                 "timestamp": datetime.now().isoformat()
#             }
        
#     except Exception as e:
#         logger.error(f"Readiness check failed: {e}")
#         return {
#             "status": "not_ready",
#             "reason": "health_check_failed",
#             "error": str(e),
#             "timestamp": datetime.now().isoformat()
#         }


# @router.get("/liveness")
# async def liveness_check():
#     """
#     Kubernetes-style liveness probe
#     Returns 200 if application is running
#     """
#     return {
#         "status": "alive",
#         "timestamp": datetime.now().isoformat(),
#         "uptime_seconds": "dynamic"  # Would calculate actual uptime in production
#     }


# @router.get("/services/{service_name}")
# async def service_health_check(service_name: str):
#     """
#     Check health of specific service
#     """
#     try:
#         if service_name == "database":
#             try:
#                 db = db_manager.get_session()
#                 db.execute("SELECT 1")
#                 db.close()
                
#                 return {
#                     "service": service_name,
#                     "status": "healthy",
#                     "timestamp": datetime.now().isoformat(),
#                     "details": {
#                         "connection": "established",
#                         "type": "sqlite" if "sqlite" in str(db_manager.database_url) else "postgresql"
#                     }
#                 }
#             except Exception as e:
#                 return {
#                     "service": service_name,
#                     "status": "unhealthy",
#                     "timestamp": datetime.now().isoformat(),
#                     "error": str(e)
#                 }
        
#         elif service_name == "llm":
#             llm_service = get_optional_llm_service()
#             if llm_service:
#                 status = llm_service.get_service_status()
#                 return {
#                     "service": service_name,
#                     "status": status.get("status", "unknown"),
#                     "timestamp": datetime.now().isoformat(),
#                     "details": status
#                 }
#             else:
#                 return {
#                     "service": service_name,
#                     "status": "unavailable",
#                     "timestamp": datetime.now().isoformat(),
#                     "reason": "Service not initialized"
#                 }
        
#         elif service_name == "vector":
#             vector_service = get_optional_vector_service()
#             if vector_service and vector_service.is_available():
#                 status = vector_service.get_service_status()
#                 return {
#                     "service": service_name,
#                     "status": status.get("status", "unknown"),
#                     "timestamp": datetime.now().isoformat(),
#                     "details": status
#                 }
#             else:
#                 return {
#                     "service": service_name,
#                     "status": "unavailable",
#                     "timestamp": datetime.now().isoformat(),
#                     "reason": "Service not available"
#                 }
        
#         else:
#             return {
#                 "service": service_name,
#                 "status": "unknown",
#                 "timestamp": datetime.now().isoformat(),
#                 "error": f"Unknown service: {service_name}"
#             }
            
#     except Exception as e:
#         logger.error(f"Service health check failed for {service_name}: {e}")
#         return {
#             "service": service_name,
#             "status": "error",
#             "timestamp": datetime.now().isoformat(),
#             "error": str(e)
#         }


async def _get_system_stats() -> Dict[str, Any]:
    """Get basic system statistics"""
    try:
        db = db_manager.get_session()
        
        # Database stats
        from app.models.database import Portfolio, ComplianceRule, ComplianceBreach, PositionHistory
        
        portfolio_count = db.query(Portfolio).count()
        position_count = db.query(PositionHistory).count()
        rules_count = db.query(ComplianceRule).filter(ComplianceRule.is_active == True).count()
        open_breaches = db.query(ComplianceBreach).filter(ComplianceBreach.status == "OPEN").count()
        
        db.close()
        
        return {
            "portfolio_count": portfolio_count,
            "position_count": position_count,
            "active_compliance_rules": rules_count,
            "open_breaches": open_breaches,
            "system_load": "normal"
        }
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return {"error": str(e)}


# async def _get_performance_metrics() -> Dict[str, Any]:
#     """Get system performance metrics"""
#     try:
#         # In production, these would be real metrics from monitoring systems
#         return {
#             "avg_response_time_ms": 250,
#             "requests_per_minute": 45,
#             "error_rate_percent": 0.1,
#             "cache_hit_rate_percent": 85.5,
#             "database_connections": {
#                 "active": 3,
#                 "idle": 7,
#                 "max_pool_size": 20
#             }
#         }
        
#     except Exception as e:
#         return {"error": str(e)}


# async def _get_resource_usage() -> Dict[str, Any]:
#     """Get resource usage information"""
#     try:
#         # In production, implement actual resource monitoring
#         return {
#             "memory_usage_mb": 512,
#             "cpu_usage_percent": 15.5,
#             "disk_usage": {
#                 "database_size_mb": 128,
#                 "log_files_size_mb": 64,
#                 "vector_store_size_mb": 256
#             },
#             "network": {
#                 "active_connections": 12,
#                 "total_requests_today": 1250
#             }
#         }
        
#     except Exception as e:
#         return {"error": str(e)}


# async def _get_recent_errors() -> Dict[str, Any]:
#     """Get recent error information"""
#     try:
#         # In production, query actual error logs
#         return {
#             "last_24_hours": {
#                 "total_errors": 3,
#                 "error_types": {
#                     "validation_error": 2,
#                     "database_timeout": 1
#                 },
#                 "critical_errors": 0
#             },
#             "last_error": {
#                 "timestamp": (datetime.now()).isoformat(),
#                 "type": "validation_error",
#                 "message": "Invalid portfolio weight provided"
#             }
#         }
        
#     except Exception as e:
#         return {"error": str(e)}