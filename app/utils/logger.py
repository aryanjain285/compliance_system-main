"""
Logging Configuration and Utilities
"""
import logging
import logging.handlers
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
import structlog
from rich.logging import RichHandler
from rich.console import Console
from app.config.settings import get_settings

settings = get_settings()


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra"):
            log_obj.update(record.extra)
        
        return json.dumps(log_obj, default=str)


def setup_logging():
    """Setup application logging configuration"""
    
    # Create logs directory
    settings.create_log_dir()
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with Rich for development
    if settings.is_development:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
        )
        console_handler.setLevel(getattr(logging, settings.log_level))
        root_logger.addHandler(console_handler)
    else:
        # Simple console handler for production
        console_handler = logging.StreamHandler(sys.stdout)
        if settings.log_format == "json":
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
        console_handler.setLevel(getattr(logging, settings.log_level))
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    
    if settings.log_format == "json":
        file_handler.setFormatter(JSONFormatter())
    else:
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
    
    file_handler.setLevel(getattr(logging, settings.log_level))
    root_logger.addHandler(file_handler)
    
    # Configure specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    return structlog.get_logger()


def get_logger(name: str = None) -> structlog.BoundLogger:
    """Get a structured logger instance"""
    return structlog.get_logger(name)


class ComplianceLogger:
    """Enhanced logger for compliance-specific events"""
    
    def __init__(self, logger_name: str = "compliance"):
        self.logger = get_logger(logger_name)
    
    def log_rule_evaluation(self, rule_id: str, result: Dict[str, Any], 
                          execution_time_ms: float):
        """Log rule evaluation event"""
        self.logger.info(
            "Rule evaluation completed",
            rule_id=rule_id,
            status=result.get("status"),
            execution_time_ms=execution_time_ms,
            event_type="rule_evaluation"
        )
    
    def log_breach_detected(self, breach_id: str, rule_id: str, 
                          observed_value: float, threshold: float):
        """Log breach detection event"""
        self.logger.warning(
            "Compliance breach detected",
            breach_id=breach_id,
            rule_id=rule_id,
            observed_value=observed_value,
            threshold=threshold,
            event_type="breach_detected"
        )
    
    def log_breach_resolved(self, breach_id: str, resolved_by: str):
        """Log breach resolution event"""
        self.logger.info(
            "Compliance breach resolved",
            breach_id=breach_id,
            resolved_by=resolved_by,
            event_type="breach_resolved"
        )
    
    def log_policy_uploaded(self, policy_id: str, filename: str, 
                          chunks_count: int, rules_extracted: int):
        """Log policy upload event"""
        self.logger.info(
            "Policy document uploaded and processed",
            policy_id=policy_id,
            filename=filename,
            chunks_count=chunks_count,
            rules_extracted=rules_extracted,
            event_type="policy_uploaded"
        )
    
    def log_portfolio_update(self, symbol: str, changes: Dict[str, Any]):
        """Log portfolio update event"""
        self.logger.info(
            "Portfolio position updated",
            symbol=symbol,
            changes=changes,
            event_type="portfolio_update"
        )
    
    def log_api_request(self, method: str, path: str, status_code: int, 
                       duration_ms: float, user_id: Optional[str] = None):
        """Log API request event"""
        self.logger.info(
            "API request processed",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            user_id=user_id,
            event_type="api_request"
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error event with context"""
        self.logger.error(
            f"Error occurred: {str(error)}",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {},
            event_type="error",
            exc_info=True
        )
    
    def log_system_event(self, event_type: str, message: str, **kwargs):
        """Log generic system event"""
        self.logger.info(
            message,
            event_type=event_type,
            **kwargs
        )


# Global compliance logger instance
compliance_logger = ComplianceLogger()


def log_execution_time(func_name: str):
    """Decorator to log function execution time"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                compliance_logger.logger.debug(
                    f"Function {func_name} executed",
                    function=func_name,
                    execution_time_ms=round(execution_time, 2),
                    event_type="function_execution"
                )
                return result
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                compliance_logger.log_error(
                    e, 
                    {
                        "function": func_name,
                        "execution_time_ms": round(execution_time, 2)
                    }
                )
                raise
        return wrapper
    return decorator