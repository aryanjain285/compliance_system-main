"""
Compliance-specific Pydantic schemas
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from app.schemas import BreachStatusEnum, CaseStatusEnum, RuleSeverityEnum


class ComplianceCheckRequest(BaseModel):
    """Request model for compliance check"""
    rule_ids: Optional[List[str]] = Field(None, description="Specific rule IDs to check")
    include_skipped: bool = Field(default=False, description="Include skipped rules in results")
    detailed_results: bool = Field(default=True, description="Include detailed evaluation results")


class RuleEvaluationResult(BaseModel):
    """Individual rule evaluation result"""
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


class ComplianceCheckResponse(BaseModel):
    """Response model for compliance check"""
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
    timestamp: datetime


class BreachSummary(BaseModel):
    """Summary of a compliance breach"""
    breach_id: str
    rule_id: str
    rule_name: Optional[str]
    rule_description: str
    control_type: str
    severity: RuleSeverityEnum
    observed_value: Optional[float]
    threshold: Optional[float]
    breach_magnitude: Optional[float]
    breach_timestamp: datetime
    status: BreachStatusEnum
    age_hours: Optional[float]
    age_category: Optional[str]
    sla_status: Optional[str]
    impact_assessment: Optional[Dict[str, Any]]
    external_reference: Optional[str]
    portfolio_snapshot_size: int


class BreachListResponse(BaseModel):
    """Response model for breach listing"""
    breaches: List[BreachSummary]
    total_count: int
    open_count: int
    resolved_count: int
    critical_count: int
    high_count: int
    filter_applied: Dict[str, Any]
    timestamp: datetime


class BreachResolutionRequest(BaseModel):
    """Request to resolve a breach"""
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
    """Response model for breach explanation"""
    breach_id: str
    explanation: str
    generated_at: datetime
    generation_method: str = "ai_generated"
    confidence: Optional[float] = None


class ComplianceCaseSummary(BaseModel):
    """Summary of a compliance case"""
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
    days_open: Optional[int]
    sla_status: Optional[str]


class CaseListResponse(BaseModel):
    """Response model for case listing"""
    cases: List[ComplianceCaseSummary]
    total_count: int
    open_count: int
    overdue_count: int
    high_priority_count: int
    filter_applied: Dict[str, Any]
    timestamp: datetime


class ComplianceAnalytics(BaseModel):
    """Comprehensive compliance analytics"""
    period: Dict[str, Any]
    summary: Dict[str, Any]
    breach_analysis: Dict[str, Any]
    performance: Dict[str, Any]
    trends: Dict[str, Any]
    generated_at: datetime


class ComplianceMetrics(BaseModel):
    """Key compliance metrics"""
    compliance_rate: float
    total_active_rules: int
    open_breaches: int
    resolved_breaches_30d: int
    average_resolution_time_hours: float
    critical_breaches: int
    high_breaches: int
    overdue_cases: int
    system_health_score: float
    last_evaluation: Optional[datetime]


class SystemHealthCheck(BaseModel):
    """System health check response"""
    overall_status: str
    database_status: str
    llm_service_status: str
    vector_store_status: str
    portfolio_data_freshness: str
    active_rules_count: int
    total_positions: int
    last_evaluation: Optional[datetime]
    errors: List[str]
    warnings: List[str]
    timestamp: datetime