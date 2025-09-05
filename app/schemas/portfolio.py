"""
Portfolio-specific Pydantic schemas
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator


class PositionBase(BaseModel):
    """Base position model"""
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
    """Create position request"""
    pass


class PositionUpdate(BaseModel):
    """Update position request"""
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
    """Position response model"""
    position_id: str
    last_updated: datetime
    created_at: datetime
    
    class Config:
        from_attributes = True


class PositionBulkUpdate(BaseModel):
    """Bulk position update request"""
    positions: List[PositionCreate] = Field(..., description="List of positions to create/update")
    update_mode: str = Field(default="replace", description="replace, merge, or append")
    
    @validator('update_mode')
    def validate_update_mode(cls, v):
        valid_modes = ['replace', 'merge', 'append']
        if v not in valid_modes:
            raise ValueError(f"Update mode must be one of: {valid_modes}")
        return v


class PositionBulkResponse(BaseModel):
    """Bulk position update response"""
    created_count: int
    updated_count: int
    deleted_count: int
    error_count: int
    total_positions: int
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    timestamp: datetime


class PortfolioSummary(BaseModel):
    """Portfolio summary statistics"""
    total_positions: int
    total_market_value: float
    total_weight: float
    top_positions: Dict[str, float]
    sector_breakdown: Dict[str, float]
    country_breakdown: Dict[str, float]
    rating_breakdown: Dict[str, float]
    currency_breakdown: Dict[str, float]
    instrument_type_breakdown: Dict[str, float]
    concentration_metrics: Dict[str, float]
    last_updated: datetime


class ConcentrationAnalysis(BaseModel):
    """Portfolio concentration analysis"""
    issuer_concentration: Dict[str, Any]
    sector_concentration: Dict[str, Any]
    country_concentration: Dict[str, Any]
    rating_concentration: Dict[str, Any]
    currency_concentration: Dict[str, Any]
    herfindahl_indices: Dict[str, float]
    concentration_warnings: List[Dict[str, Any]]
    analysis_timestamp: datetime


class PositionHistory(BaseModel):
    """Position history record"""
    history_id: str
    position_id: str
    symbol: str
    change_type: str  # insert, update, delete
    changed_fields: Dict[str, Any]
    previous_values: Dict[str, Any]
    new_values: Dict[str, Any]
    change_reason: Optional[str]
    changed_by: Optional[str]
    change_timestamp: datetime


class PositionHistoryResponse(BaseModel):
    """Position history response"""
    history: List[PositionHistory]
    total_records: int
    position_symbol: str
    date_range: Dict[str, datetime]
    change_summary: Dict[str, int]


class PortfolioAnalytics(BaseModel):
    """Comprehensive portfolio analytics"""
    summary: PortfolioSummary
    concentration: ConcentrationAnalysis
    performance_metrics: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    compliance_summary: Dict[str, Any]
    trends: Dict[str, Any]
    generated_at: datetime


class PortfolioValidationResult(BaseModel):
    """Portfolio data validation result"""
    is_valid: bool
    total_positions: int
    validation_errors: List[Dict[str, Any]]
    validation_warnings: List[Dict[str, Any]]
    weight_sum: float
    weight_tolerance: float
    missing_data_count: int
    duplicate_positions: List[str]
    validation_timestamp: datetime


class PositionSearchRequest(BaseModel):
    """Position search request"""
    symbols: Optional[List[str]] = Field(None, description="Filter by symbols")
    sectors: Optional[List[str]] = Field(None, description="Filter by sectors")
    countries: Optional[List[str]] = Field(None, description="Filter by countries")
    ratings: Optional[List[str]] = Field(None, description="Filter by ratings")
    min_weight: Optional[float] = Field(None, ge=0, le=1, description="Minimum weight")
    max_weight: Optional[float] = Field(None, ge=0, le=1, description="Maximum weight")
    min_market_value: Optional[float] = Field(None, ge=0, description="Minimum market value")
    max_market_value: Optional[float] = Field(None, ge=0, description="Maximum market value")
    instrument_types: Optional[List[str]] = Field(None, description="Filter by instrument types")
    sort_by: str = Field(default="weight", description="Sort field")
    sort_order: str = Field(default="desc", description="Sort order (asc/desc)")
    limit: int = Field(default=100, ge=1, le=1000, description="Result limit")

    @validator('sort_order')
    def validate_sort_order(cls, v):
        valid_orders = ['asc', 'desc']
        if v not in valid_orders:
            raise ValueError(f"Sort order must be one of: {valid_orders}")
        return v


class PositionSearchResponse(BaseModel):
    """Position search response"""
    positions: List[PositionResponse]
    total_found: int
    search_criteria: PositionSearchRequest
    summary_stats: Dict[str, Any]
    timestamp: datetime