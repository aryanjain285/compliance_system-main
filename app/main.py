"""
Main FastAPI Application
Production-ready compliance system with comprehensive API
"""
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid
from contextlib import asynccontextmanager

from app.config.settings import get_settings
from app.utils.logger import setup_logging, get_logger, compliance_logger
from app.utils.exceptions import ComplianceException, compliance_exception_handler
from app.models.database import db_manager
from app.api.routes import health, portfolio, compliance, rules, policies, analytics
from app.api.dependencies import check_system_health

settings = get_settings()

# Setup logging
setup_logging()
logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        
        # Log incoming request
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent")
            }
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time": round(process_time * 1000, 2)
                }
            )
            
            # Log to compliance logger for audit
            compliance_logger.log_api_request(
                method=request.method,
                path=str(request.url.path),
                status_code=response.status_code,
                duration_ms=round(process_time * 1000, 2),
                user_id=getattr(request.state, 'user_id', None)
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            logger.error(
                f"Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "process_time": round(process_time * 1000, 2)
                }
            )
            
            # Re-raise the exception
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("Starting Compliance System API")
    
    try:
        # Initialize database
        db_manager.create_tables()
        db_manager.init_demo_data()
        logger.info("Database initialized successfully")
        
        # Check system health
        health_status = check_system_health()
        logger.info(f"System health check: {health_status}")
        
        # Log startup completion
        logger.info("Compliance System API started successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Compliance System API")


# Create FastAPI application
app = FastAPI(
    title="Compliance System API",
    description="""
    Advanced compliance monitoring and rule management system for financial institutions.
    
    ## Features
    
    * **Portfolio Management**: Track and manage investment positions
    * **Compliance Rules**: Define and manage compliance rules with multiple control types
    * **Real-time Monitoring**: Continuous compliance monitoring and breach detection
    * **Policy Management**: Upload, parse, and manage policy documents
    * **LLM-Powered Analysis**: Intelligent rule extraction from policy documents
    * **Vector Search**: Semantic search across policy knowledge base
    * **Analytics & Reporting**: Comprehensive compliance analytics and reporting
    * **Audit Trail**: Complete audit trail for regulatory compliance
    
    ## Control Types Supported
    
    * **Quantitative Limits**: Portfolio concentration, exposure limits
    * **List Constraints**: Allowed/prohibited securities, ratings requirements
    * **Temporal Windows**: Holding periods, lock-up requirements
    * **Process Controls**: Approval workflows, documentation requirements
    * **Reporting & Disclosure**: Regulatory reporting obligations
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Health", "description": "System health and status endpoints"},
        {"name": "Portfolio", "description": "Portfolio management operations"},
        {"name": "Compliance", "description": "Compliance monitoring and evaluation"},
        {"name": "Rules", "description": "Compliance rule management"},
        {"name": "Policies", "description": "Policy document management"},
        {"name": "Analytics", "description": "Compliance analytics and reporting"},
    ],
    lifespan=lifespan
)

# Security middleware
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure with actual allowed hosts in production
    )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request logging middleware
app.add_middleware(RequestLoggingMiddleware)


# Exception handlers
@app.exception_handler(ComplianceException)
async def compliance_exception_handler_wrapper(request: Request, exc: ComplianceException):
    """Handle custom compliance exceptions"""
    http_exc = compliance_exception_handler(exc)
    return JSONResponse(
        status_code=http_exc.status_code,
        content=http_exc.detail,
        headers={"X-Request-ID": getattr(request.state, 'request_id', 'unknown')}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": {
                "errors": exc.errors(),
                "body": exc.body
            }
        },
        headers={"X-Request-ID": getattr(request.state, 'request_id', 'unknown')}
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": f"HTTP_{exc.status_code}",
            "message": exc.detail,
            "request_id": getattr(request.state, 'request_id', 'unknown')
        },
        headers={"X-Request-ID": getattr(request.state, 'request_id', 'unknown')}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(
        f"Unhandled exception",
        extra={
            "request_id": getattr(request.state, 'request_id', 'unknown'),
            "path": request.url.path,
            "method": request.method,
            "error": str(exc)
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred" if settings.is_production else str(exc),
            "request_id": getattr(request.state, 'request_id', 'unknown')
        },
        headers={"X-Request-ID": getattr(request.state, 'request_id', 'unknown')}
    )


# Root endpoint
@app.get("/", tags=["Health"])
async def root():
    """API root endpoint with basic information"""
    return {
        "message": "Compliance System API",
        "version": "2.0.0",
        "status": "operational",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": {
            "health": "/health",
            "portfolio": "/api/portfolio",
            "compliance": "/api/compliance",
            "rules": "/api/rules",
            "policies": "/api/policies",
            "analytics": "/api/analytics"
        },
        "timestamp": time.time()
    }


# Configuration endpoint
@app.get("/api/config", tags=["Health"])
async def get_system_config():
    """Get system configuration and feature flags"""
    return {
        "version": "2.0.0",
        "environment": "production" if settings.is_production else "development",
        "features": {
            "llm_enabled": bool(settings.openai_api_key or settings.anthropic_api_key),
            "vector_search_enabled": not settings.skip_vector_store,
            "file_upload_enabled": True,
            "real_time_monitoring": True,
            "analytics_enabled": True,
            "audit_logging": True
        },
        "limits": {
            "max_file_size_mb": settings.max_file_size_mb,
            "allowed_file_types": settings.allowed_file_types,
            "max_positions": 10000,
            "max_rules": 1000,
            "rate_limits": {
                "compliance_checks": "50/hour",
                "file_uploads": "20/hour",
                "llm_requests": "30/hour"
            }
        },
        "timestamp": time.time()
    }


# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["Portfolio"])
app.include_router(compliance.router, prefix="/api/compliance", tags=["Compliance"])
app.include_router(rules.router, prefix="/api/rules", tags=["Rules"])
app.include_router(policies.router, prefix="/api/policies", tags=["Policies"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
        access_log=settings.is_development
    )