"""
Custom Exception Classes for Compliance System
"""
from typing import Any, Dict, Optional
from fastapi import HTTPException, status


class ComplianceException(Exception):
    """Base exception class for compliance system"""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}


class DatabaseException(ComplianceException):
    """Database-related exceptions"""
    pass


class ValidationException(ComplianceException):
    """Data validation exceptions"""
    pass


class RuleEvaluationException(ComplianceException):
    """Rule evaluation exceptions"""
    pass


class PolicyParsingException(ComplianceException):
    """Policy document parsing exceptions"""
    pass


class LLMServiceException(ComplianceException):
    """LLM service exceptions"""
    pass


class VectorStoreException(ComplianceException):
    """Vector store exceptions"""
    pass


class FileUploadException(ComplianceException):
    """File upload exceptions"""
    pass


class ConfigurationException(ComplianceException):
    """Configuration-related exceptions"""
    pass


class AuthenticationException(ComplianceException):
    """Authentication exceptions"""
    pass


class AuthorizationException(ComplianceException):
    """Authorization exceptions"""
    pass


class RateLimitException(ComplianceException):
    """Rate limiting exceptions"""
    pass


# HTTP Exception mappings
def compliance_exception_handler(exc: ComplianceException) -> HTTPException:
    """Convert compliance exceptions to HTTP exceptions"""
    
    status_code_map = {
        ValidationException: status.HTTP_400_BAD_REQUEST,
        FileUploadException: status.HTTP_400_BAD_REQUEST,
        ConfigurationException: status.HTTP_500_INTERNAL_SERVER_ERROR,
        DatabaseException: status.HTTP_500_INTERNAL_SERVER_ERROR,
        RuleEvaluationException: status.HTTP_500_INTERNAL_SERVER_ERROR,
        PolicyParsingException: status.HTTP_422_UNPROCESSABLE_ENTITY,
        LLMServiceException: status.HTTP_503_SERVICE_UNAVAILABLE,
        VectorStoreException: status.HTTP_503_SERVICE_UNAVAILABLE,
        AuthenticationException: status.HTTP_401_UNAUTHORIZED,
        AuthorizationException: status.HTTP_403_FORBIDDEN,
        RateLimitException: status.HTTP_429_TOO_MANY_REQUESTS,
    }
    
    status_code = status_code_map.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    return HTTPException(
        status_code=status_code,
        detail={
            "message": exc.message,
            "error_code": exc.error_code,
            "details": exc.details,
        }
    )


# Specific business logic exceptions
class RuleNotFound(ComplianceException):
    """Rule not found exception"""
    
    def __init__(self, rule_id: str):
        super().__init__(
            message=f"Compliance rule '{rule_id}' not found",
            error_code="RULE_NOT_FOUND",
            details={"rule_id": rule_id}
        )


class BreachNotFound(ComplianceException):
    """Breach not found exception"""
    
    def __init__(self, breach_id: str):
        super().__init__(
            message=f"Compliance breach '{breach_id}' not found",
            error_code="BREACH_NOT_FOUND",
            details={"breach_id": breach_id}
        )


class PolicyNotFound(ComplianceException):
    """Policy not found exception"""
    
    def __init__(self, policy_id: str):
        super().__init__(
            message=f"Policy document '{policy_id}' not found",
            error_code="POLICY_NOT_FOUND",
            details={"policy_id": policy_id}
        )


class PositionNotFound(ComplianceException):
    """Portfolio position not found exception"""
    
    def __init__(self, symbol: str):
        super().__init__(
            message=f"Portfolio position '{symbol}' not found",
            error_code="POSITION_NOT_FOUND",
            details={"symbol": symbol}
        )


class InvalidRuleExpression(ValidationException):
    """Invalid rule expression exception"""
    
    def __init__(self, rule_id: str, errors: list):
        super().__init__(
            message=f"Invalid rule expression for rule '{rule_id}'",
            error_code="INVALID_RULE_EXPRESSION",
            details={"rule_id": rule_id, "validation_errors": errors}
        )


class UnsupportedFileType(FileUploadException):
    """Unsupported file type exception"""
    
    def __init__(self, file_type: str, allowed_types: list):
        super().__init__(
            message=f"Unsupported file type '{file_type}'. Allowed types: {', '.join(allowed_types)}",
            error_code="UNSUPPORTED_FILE_TYPE",
            details={"file_type": file_type, "allowed_types": allowed_types}
        )


class FileTooLarge(FileUploadException):
    """File too large exception"""
    
    def __init__(self, file_size: int, max_size: int):
        super().__init__(
            message=f"File size {file_size} bytes exceeds maximum allowed size {max_size} bytes",
            error_code="FILE_TOO_LARGE",
            details={"file_size": file_size, "max_size": max_size}
        )


class ServiceUnavailable(ComplianceException):
    """Service unavailable exception"""
    
    def __init__(self, service_name: str, reason: str = None):
        message = f"Service '{service_name}' is currently unavailable"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="SERVICE_UNAVAILABLE",
            details={"service": service_name, "reason": reason}
        )


class DuplicateResource(ValidationException):
    """Duplicate resource exception"""
    
    def __init__(self, resource_type: str, identifier: str):
        super().__init__(
            message=f"Duplicate {resource_type} with identifier '{identifier}' already exists",
            error_code="DUPLICATE_RESOURCE",
            details={"resource_type": resource_type, "identifier": identifier}
        )


class InsufficientData(ValidationException):
    """Insufficient data exception"""
    
    def __init__(self, requirement: str):
        super().__init__(
            message=f"Insufficient data: {requirement}",
            error_code="INSUFFICIENT_DATA",
            details={"requirement": requirement}
        )


class ConcurrencyError(DatabaseException):
    """Concurrency error exception"""
    
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            message=f"Concurrency conflict while updating {resource_type} '{resource_id}'",
            error_code="CONCURRENCY_ERROR",
            details={"resource_type": resource_type, "resource_id": resource_id}
        )


class ExternalServiceError(ComplianceException):
    """External service error exception"""
    
    def __init__(self, service: str, error_message: str, status_code: Optional[int] = None):
        super().__init__(
            message=f"External service '{service}' error: {error_message}",
            error_code="EXTERNAL_SERVICE_ERROR",
            details={
                "service": service,
                "error_message": error_message,
                "status_code": status_code
            }
        )


# Validation helpers
def validate_rule_expression(expression: Dict[str, Any], control_type: str) -> None:
    """Validate rule expression structure"""
    errors = []
    
    if control_type == "quant_limit":
        required_fields = ["metric", "operator", "threshold", "scope"]
        for field in required_fields:
            if field not in expression:
                errors.append(f"Missing required field: {field}")
        
        if "operator" in expression and expression["operator"] not in ["<=", ">=", "<", ">", "==", "!="]:
            errors.append(f"Invalid operator: {expression['operator']}")
        
        if "threshold" in expression and not isinstance(expression["threshold"], (int, float)):
            errors.append("Threshold must be a number")
    
    elif control_type == "list_constraint":
        required_fields = ["field", "scope"]
        for field in required_fields:
            if field not in expression:
                errors.append(f"Missing required field: {field}")
        
        if not ("allowed_values" in expression or "denied_values" in expression):
            errors.append("Must specify either allowed_values or denied_values")
    
    elif control_type == "temporal_window":
        required_fields = ["metric", "minimum_days", "scope"]
        for field in required_fields:
            if field not in expression:
                errors.append(f"Missing required field: {field}")
    
    elif control_type == "process_control":
        required_fields = ["approval_required", "evidence_required"]
        for field in required_fields:
            if field not in expression:
                errors.append(f"Missing required field: {field}")
    
    elif control_type == "reporting_disclosure":
        required_fields = ["report_type", "frequency"]
        for field in required_fields:
            if field not in expression:
                errors.append(f"Missing required field: {field}")
    
    if errors:
        raise InvalidRuleExpression("validation", errors)