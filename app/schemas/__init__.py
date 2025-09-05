"""
Pydantic Schemas for Request/Response Models
"""
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


# Enums
class DocumentStatusEnum(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    PENDING = "pending"


class RuleSeverityEnum(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ControlTypeEnum(str, Enum):
    QUANT_LIMIT = "quant_limit"
    LIST_CONSTRAINT = "list_constraint"
    TEMPORAL_WINDOW = "temporal_window"
    PROCESS_CONTROL = "process_control"
    REPORTING_DISCLOSURE = "reporting_disclosure"


class BreachStatusEnum(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"
    UNDER_REVIEW = "under_review"


# Add helper functions to convert database enums to API enums:
def convert_breach_status(db_status: str) -> str:
    """Convert database status to API status"""
    mapping = {
        "OPEN": "open",
        "RESOLVED": "resolved", 
        "FALSE_POSITIVE": "false_positive",
        "UNDER_REVIEW": "under_review"
    }
    return mapping.get(db_status, db_status.lower())

def convert_document_status(db_status: str) -> str:
    """Convert database status to API status"""
    mapping = {
        "UPLOADED": "active",
        "PARSED": "active", 
        "INDEXED": "active",
        "ERROR": "error"
    }
    return mapping.get(db_status, db_status.lower())


class CaseStatusEnum(str, Enum):
    OPEN = "open"
    IN_REVIEW = "in_review"
    CLOSED = "closed"
    ESCALATED = "escalated"


# Base Models
class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# Portfolio Schemas
class PositionBase(BaseModel):
    symbol: str = Field(..., description="Security symbol")
    name: Optional[str] = Field(None, description="Security name")
    weight: float = Field(..., ge=0, le=1, description="Portfolio weight (0-1)")
    market_value: float = Field(..., ge=0, description="Market value in base currency")
    quantity: Optional[float] = Field(None, ge=0, description="Number of shares/units")
    price: Optional[float] = Field(None, ge=0, description="Price per unit")
    sector: Optional[str] = Field(None, description="Sector classification")
    industry: Optional[str] = Field(None, description="Industry classification")
    country: Optional[str] = Field(None, description="Country of domicile")
    currency: str = Field(default="USD", description="Currency code")
    rating: Optional[str] = Field(None, description="Credit rating")
    rating_agency: Optional[str] = Field(None, description="Rating agency")
    instrument_type: Optional[str] = Field(None, description="Instrument type")
    exchange: Optional[str] = Field(None, description="Exchange")
    maturity_date: Optional[datetime] = Field(None, description="Maturity date for bonds")
    acquisition_date: Optional[datetime] = Field(None, description="Acquisition date")
    bloomberg_id: Optional[str] = Field(None, description="Bloomberg identifier")
    cusip: Optional[str] = Field(None, description="CUSIP identifier")
    isin: Optional[str] = Field(None, description="ISIN identifier")
    sedol: Optional[str] = Field(None, description="SEDOL identifier")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class PositionCreate(PositionBase):
    pass


class PositionUpdate(BaseModel):
    weight: Optional[float] = Field(None, ge=0, le=1)
    market_value: Optional[float] = Field(None, ge=0)
    quantity: Optional[float] = Field(None, ge=0)
    price: Optional[float] = Field(None, ge=0)
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    rating: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PositionResponse(PositionBase):
    position_id: str
    last_updated: datetime
    created_at: datetime
    
    @validator('weight', pre=True)
    def ensure_weight(cls, v):
        """Ensure weight is present"""
        return v if v is not None else 0.0
    
    @validator('market_value', pre=True) 
    def ensure_market_value(cls, v):
        """Ensure market_value is present"""
        return v if v is not None else 0.0
    
    class Config:
        from_attributes = True


class PortfolioSummary(BaseModel):
    total_positions: int
    total_market_value: float
    total_weight: float
    top_positions: Dict[str, float]
    sector_breakdown: Dict[str, float]
    country_breakdown: Dict[str, float]
    rating_breakdown: Dict[str, float]
    concentration_metrics: Dict[str, float]
    last_updated: datetime


# Policy Schemas
class PolicyDocumentBase(BaseModel):
    filename: str = Field(..., description="Original filename")
    document_type: Optional[str] = Field(None, description="Document type classification")
    jurisdiction: Optional[str] = Field(None, description="Jurisdiction")
    effective_date: Optional[datetime] = Field(None, description="Effective date")
    expiry_date: Optional[datetime] = Field(None, description="Expiry date")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class PolicyDocumentCreate(PolicyDocumentBase):
    content: Optional[str] = Field(None, description="Document content")
    uploaded_by: str = Field(default="api_user", description="Uploader identifier")


class PolicyDocumentResponse(PolicyDocumentBase):
    policy_id: str
    status: DocumentStatusEnum
    upload_date: datetime
    uploaded_by: str
    version: int
    content_stats: Dict[str, Any] = Field(default_factory=dict)
    chunk_stats: Dict[str, Any] = Field(default_factory=dict) 
    rules_generated: int = 0
    created_at: datetime
    updated_at: datetime
    
    @validator('status', pre=True)
    def convert_status(cls, v):
        """Convert database status to API status"""
        return convert_document_status(str(v))
    
    class Config:
        from_attributes = True


class PolicyUploadResponse(BaseResponse):
    policy_id: str
    filename: str
    chunks_count: int
    extracted_rules_count: int
    extracted_rules: List[Dict[str, Any]]
    vector_indexing: Dict[str, Any]
    requires_approval: bool = True


class PolicySearchRequest(BaseModel):
    search_terms: List[str] = Field(..., description="Terms to search for")
    case_sensitive: bool = Field(default=False, description="Case sensitive search")


class PolicySearchResult(BaseModel):
    chunk_id: str
    chunk_index: int
    section_title: Optional[str]
    page_number: Optional[int]
    matches: List[Dict[str, Any]]
    match_count: int


# Rule Schemas
class RuleExpressionQuantLimit(BaseModel):
    metric: str = Field(..., description="Metric to measure")
    operator: str = Field(..., description="Comparison operator")
    threshold: float = Field(..., description="Threshold value")
    scope: str = Field(..., description="Scope of application")
    group_by: Optional[str] = Field(None, description="Grouping dimension")
    filter: Optional[str] = Field(None, description="Filter value")


class RuleExpressionListConstraint(BaseModel):
    field: str = Field(..., description="Field to constrain")
    allowed_values: Optional[List[str]] = Field(None, description="Allowed values")
    denied_values: Optional[List[str]] = Field(None, description="Prohibited values")
    scope: str = Field(..., description="Scope of application")


class RuleExpressionTemporal(BaseModel):
    metric: str = Field(..., description="Temporal metric")
    minimum_days: int = Field(..., ge=0, description="Minimum days")
    maximum_days: Optional[int] = Field(None, ge=0, description="Maximum days")
    scope: str = Field(..., description="Scope of application")


class RuleExpressionProcess(BaseModel):
    approval_required: bool = Field(..., description="Approval required")
    approver_role: Optional[str] = Field(None, description="Required approver role")
    evidence_required: str = Field(..., description="Evidence requirements")
    sla_days: Optional[int] = Field(None, ge=1, description="SLA in days")


class RuleExpressionReporting(BaseModel):
    report_type: str = Field(..., description="Type of report")
    frequency: str = Field(..., description="Reporting frequency")
    deadline_days: Optional[int] = Field(None, ge=1, description="Deadline in days")
    recipient: Optional[str] = Field(None, description="Report recipient")


# Union type for rule expressions
RuleExpression = Union[
    RuleExpressionQuantLimit,
    RuleExpressionListConstraint,
    RuleExpressionTemporal,
    RuleExpressionProcess,
    RuleExpressionReporting,
    Dict[str, Any]  # Fallback for complex expressions
]


class ComplianceRuleBase(BaseModel):
    name: Optional[str] = Field(None, description="Rule name")
    description: str = Field(..., description="Rule description")
    control_type: ControlTypeEnum = Field(..., description="Control type")
    severity: RuleSeverityEnum = Field(..., description="Severity level")
    expression: RuleExpression = Field(..., description="Rule logic expression")
    materiality_bps: Optional[int] = Field(None, ge=0, description="Materiality threshold in basis points")
    source_section: Optional[str] = Field(None, description="Source policy section")
    effective_date: Optional[datetime] = Field(None, description="Effective date")
    expiry_date: Optional[datetime] = Field(None, description="Expiry date")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    @validator('expression')
    def validate_expression(cls, v, values):
        """Validate rule expression based on control type"""
        if not isinstance(v, dict):
            raise ValueError("Expression must be a dictionary")
        
        control_type = values.get('control_type')
        
        if control_type == ControlTypeEnum.QUANT_LIMIT:
            required = ['metric', 'operator', 'threshold', 'scope']
            if not all(field in v for field in required):
                raise ValueError(f"Quantitative limit requires: {required}")
        
        elif control_type == ControlTypeEnum.LIST_CONSTRAINT:
            required = ['field', 'scope']
            if not all(field in v for field in required):
                raise ValueError(f"List constraint requires: {required}")
            if not ('allowed_values' in v or 'denied_values' in v):
                raise ValueError("List constraint requires either allowed_values or denied_values")
        
        return v


class ComplianceRuleCreate(ComplianceRuleBase):
    source_policy_id: Optional[str] = Field(None, description="Source policy ID")
    created_by: str = Field(default="api_user", description="Rule creator")


class ComplianceRuleUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    expression: Optional[RuleExpression] = None
    severity: Optional[RuleSeverityEnum] = None
    materiality_bps: Optional[int] = Field(None, ge=0)
    is_active: Optional[bool] = None
    effective_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    modified_by: str = Field(default="api_user", description="Modifier")


class ComplianceRuleResponse(ComplianceRuleBase):
    rule_id: str
    source_policy_id: Optional[str]
    version: int
    is_active: bool
    created_by: str
    created_at: datetime
    modified_by: Optional[str]
    modified_at: Optional[datetime]  # Make this optional
    
    class Config:
        from_attributes = True


class RuleEvaluationResult(BaseModel):
    rule_id: str
    rule_name: Optional[str]
    rule_description: str
    control_type: str
    severity: str
    status: str = Field(..., description="compliant, breach, case_created, error, skipped")
    observed_value: Optional[float] = None
    threshold: Optional[float] = None
    violation_count: Optional[int] = None
    materiality_exceeded: Optional[bool] = None
    execution_time_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime


class ComplianceCheckResponse(BaseResponse):
    evaluation_id: str
    total_rules_checked: int
    compliant_rules: int
    breached_rules: int
    case_created_rules: int
    error_rules: int
    skipped_rules: int
    compliance_rate: float
    new_breaches: List[Dict[str, Any]]
    new_cases: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    portfolio_summary: Dict[str, Any]
    performance: Dict[str, Any]
    system_health: Dict[str, Any]
    results: List[RuleEvaluationResult]


# Breach Schemas
class ComplianceBreachResponse(BaseModel):
    breach_id: str
    rule_id: str
    rule_name: Optional[str]
    rule_description: str
    control_type: str
    severity: str
    observed_value: Optional[float]
    threshold: Optional[float]
    breach_magnitude: Optional[float]
    breach_timestamp: datetime
    status: BreachStatusEnum
    age_hours: Optional[float]
    age_category: Optional[str]
    sla_status: Optional[str]
    impact_assessment: Optional[Dict[str, Any]]  # This should be dict, not string
    external_reference: Optional[str]
    portfolio_snapshot_size: int

    @validator('impact_assessment', pre=True)
    def parse_impact_assessment(cls, v):
        """Parse impact assessment if it's a JSON string"""
        if isinstance(v, str):
            try:
                import json
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return {}
        return v or {}

    @validator('status', pre=True)
    def convert_status(cls, v):
        """Convert database status to API status"""
        return convert_breach_status(str(v))


class BreachResolutionRequest(BaseModel):
    resolved_by: str = Field(..., description="Who resolved the breach")
    notes: str = Field(default="", description="Resolution notes")
    resolution_type: str = Field(default="resolved", description="Type of resolution")

    @validator('resolution_type')
    def validate_resolution_type(cls, v):
        valid_types = ['resolved', 'false_positive', 'under_review']
        if v not in valid_types:
            raise ValueError(f"Resolution type must be one of: {valid_types}")
        return v


class BreachExplanationResponse(BaseModel):
    breach_id: str
    explanation: str
    generated_at: datetime


# Case Schemas
class ComplianceCaseResponse(BaseModel):
    case_id: str
    rule_id: str
    case_type: str
    status: CaseStatusEnum
    priority: str
    title: Optional[str]
    description: Optional[str]
    evidence_required: Optional[str]
    sla_deadline: Optional[datetime]
    assigned_to: Optional[str]
    created_at: datetime
    updated_at: datetime


# Analytics Schemas
class ComplianceAnalytics(BaseModel):
    period: Dict[str, Any]
    summary: Dict[str, Any]
    breach_analysis: Dict[str, Any]
    performance: Dict[str, Any]
    trends: Dict[str, Any]
    generated_at: datetime


# Vector Search Schemas
class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Search query")
    n_results: int = Field(default=10, ge=1, le=100, description="Number of results")
    policy_id: Optional[str] = Field(None, description="Filter by policy ID")
    keywords: Optional[List[str]] = Field(None, description="Keywords for hybrid search")
    min_relevance_score: float = Field(default=0.5, ge=0, le=1, description="Minimum relevance score")


class SemanticSearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    policy_id: Optional[str]
    section_title: Optional[str]
    page_number: Optional[int]
    chunk_type: Optional[str]
    rank: int


class SemanticSearchResponse(BaseModel):
    query: str
    results: List[SemanticSearchResult]
    total_found: int
    search_type: str
    filters_applied: Dict[str, Any]


class KnowledgeQueryRequest(BaseModel):
    question: str = Field(..., min_length=10, description="Natural language question")
    policy_id: Optional[str] = Field(None, description="Filter by specific policy")
    n_context: int = Field(default=5, ge=1, le=10, description="Number of context chunks")


class KnowledgeQueryResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    context_used: int


# Rule Extraction Schemas
class RuleExtractionRequest(BaseModel):
    policy_text: Optional[str] = Field(None, description="Raw policy text")
    policy_id: Optional[str] = Field(None, description="Policy document ID")


class ExtractedRule(BaseModel):
    rule_id: str
    description: str
    control_type: str
    severity: str
    expression: Dict[str, Any]
    materiality_bps: Optional[int] = None
    source_section: Optional[str] = None
    confidence: float = Field(..., ge=0, le=1)
    rationale: Optional[str] = None
    extraction_metadata: Optional[Dict[str, Any]] = None


class RuleExtractionResponse(BaseModel):
    proposed_rules: List[ExtractedRule]
    requires_approval: bool = True
    total_rules_extracted: int
    extraction_method: str
    source_text: Optional[str] = None


class RuleApprovalRequest(BaseModel):
    approved_rules: List[Dict[str, Any]] = Field(..., description="Rules to approve and create")
    policy_id: Optional[str] = Field(None, description="Source policy ID")
    approved_by: str = Field(default="api_user", description="Approver identifier")


class RuleApprovalResponse(BaseResponse):
    created_rule_ids: List[str]
    approved_by: str
    total_approved: int


# Health Check Schemas
class ServiceStatus(BaseModel):
    service: str
    status: str
    details: Optional[Dict[str, Any]] = None


class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    database: str
    services: Dict[str, ServiceStatus]
    stats: Optional[Dict[str, Any]] = None


# System Configuration
class SystemConfigResponse(BaseModel):
    version: str = "2.0.0"
    environment: str
    features: Dict[str, bool]
    limits: Dict[str, Any]
    timestamp: datetime