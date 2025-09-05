from __future__ import annotations
"""
SQLAlchemy ORM models and DB session utilities for the Compliance System.

Exports:
- get_db (FastAPI dependency)
- db_manager (DBManager instance)
- ORM models: ComplianceRule, ComplianceBreach, ComplianceCase, Portfolio,
  RuleEvaluation, PositionHistory, PolicyDocument, PolicyChunk
- Enums: BreachStatus, CaseStatus, CaseType, RuleSeverity, ControlType, DocumentStatus
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Boolean,
    ForeignKey,
    Text,
    Enum,
    JSON,
    text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

from app.config.settings import get_settings

# ---------- Enums ----------
import enum

class ControlType(str, enum.Enum):
    QUANT_LIMIT = "QUANT_LIMIT"
    LIST_CONSTRAINT = "LIST_CONSTRAINT"
    TEMPORAL_WINDOW = "TEMPORAL_WINDOW"
    PROCESS_CONTROL = "PROCESS_CONTROL"
    REPORTING_DISCLOSURE = "REPORTING_DISCLOSURE"

class RuleSeverity(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class BreachStatus(str, enum.Enum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    FALSE_POSITIVE = "FALSE_POSITIVE"

class CaseStatus(str, enum.Enum):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"

class CaseType(str, enum.Enum):
    PROCESS_CONTROL = "PROCESS_CONTROL"
    REPORTING_DISCLOSURE = "REPORTING_DISCLOSURE"
    BREACH_REVIEW = "BREACH_REVIEW"

class DocumentStatus(str, enum.Enum):
    UPLOADED = "UPLOADED"
    PARSED = "PARSED"
    INDEXED = "INDEXED"
    ERROR = "ERROR"

# ---------- Engine / Session ----------
_settings = get_settings()

def _prefer_db_url() -> str:
    """Use DATABASE_URL directly, ignore Supabase for now."""
    url = _settings.database_url
    return url

DATABASE_URL = _prefer_db_url()

def _make_engine(url: str):
    if url.startswith("sqlite"):
        # e.g. sqlite:///./compliance.db
        db_file = url.replace("sqlite:///", "")
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            echo=False,
            future=True,
        )
    # Postgres (Supabase) or others
    return create_engine(
        url,
        pool_pre_ping=True,
        echo=False,
        future=True,
    )

engine = _make_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()

# ---------- Models ----------
class ComplianceRule(Base):
    __tablename__ = "compliance_rules"

    rule_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=True)  # Allow nullable
    description = Column(Text, nullable=False)
    expression = Column(JSON, nullable=False)  # Change to JSON for complex expressions
    control_type = Column(Enum(ControlType), nullable=False)
    severity = Column(Enum(RuleSeverity), nullable=False, default=RuleSeverity.MEDIUM)
    materiality_bps = Column(Integer, nullable=True)
    effective_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    expiry_date = Column(DateTime, nullable=True)
    source_section = Column(String(255), nullable=True)
    
    # Add the missing field:
    source_policy_id = Column(String, nullable=True)
    
    version = Column(Integer, default=1, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Add audit fields:
    created_by = Column(String(255), nullable=True)
    modified_by = Column(String(255), nullable=True)
    modified_at = Column(DateTime, nullable=True)
    rule_metadata = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    breaches = relationship("ComplianceBreach", back_populates="rule", cascade="all,delete-orphan")

class ComplianceBreach(Base):
    __tablename__ = "compliance_breaches"

    breach_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    rule_id = Column(String, ForeignKey("compliance_rules.rule_id", ondelete="CASCADE"), nullable=False)
    
    # Fix the enum - use lowercase to match your schema expectations
    status = Column(String(50), default="open", nullable=False)  # Change from Enum to String
    
    observed_value = Column(Float, nullable=True)
    threshold_value = Column(Float, nullable=True)
    breach_magnitude = Column(Float, nullable=True)
    breach_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    portfolio_snapshot = Column(JSON, nullable=True)
    
    # Fix this - store as JSON, not string
    impact_assessment = Column(JSON, nullable=True)  # This should be JSON not string
    
    resolution_notes = Column(Text, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(String(255), nullable=True)
    external_reference = Column(String(255), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    rule = relationship("ComplianceRule", back_populates="breaches")

class ComplianceCase(Base):
    __tablename__ = "compliance_cases"

    case_id = Column(String, primary_key=True)  # string/UUID
    rule_id = Column(String, ForeignKey("compliance_rules.rule_id", ondelete="SET NULL"), nullable=True)
    status = Column(Enum(CaseStatus), default=CaseStatus.OPEN, nullable=False)
    priority = Column(Integer, default=3, nullable=False)  # 1=high ... 5=low
    sla_deadline = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

class RuleEvaluation(Base):
    __tablename__ = "rule_evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rule_id = Column(String, ForeignKey("compliance_rules.rule_id", ondelete="CASCADE"), nullable=False)
    evaluation_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    execution_time_ms = Column(Float, nullable=True)
    result = Column(JSON, nullable=True)

class Portfolio(Base):
    __tablename__ = "portfolios"

    position_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))  # Add this
    portfolio_id = Column(String, nullable=True)  # Keep existing
    symbol = Column(String(64), nullable=False, unique=True)  # Add this
    name = Column(String(255), nullable=True)
    
    # Add ALL the missing fields that your API expects:
    weight = Column(Float, nullable=False, default=0.0)
    market_value = Column(Float, nullable=False, default=0.0)
    quantity = Column(Float, nullable=True)
    price = Column(Float, nullable=True)
    sector = Column(String(255), nullable=True)
    industry = Column(String(255), nullable=True)
    country = Column(String(255), nullable=True)
    currency = Column(String(10), default="USD")
    rating = Column(String(50), nullable=True)
    rating_agency = Column(String(100), nullable=True)
    instrument_type = Column(String(100), nullable=True)
    exchange = Column(String(100), nullable=True)
    maturity_date = Column(DateTime, nullable=True)
    acquisition_date = Column(DateTime, nullable=True)
    bloomberg_id = Column(String(50), nullable=True)
    cusip = Column(String(50), nullable=True)
    isin = Column(String(50), nullable=True)
    sedol = Column(String(50), nullable=True)
    position_metadata = Column(JSON, nullable=True)
    
    owner = Column(String(255), nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class PositionHistory(Base):
    __tablename__ = "position_history"

    # Add missing fields:
    history_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))  # Add this
    id = Column(Integer, autoincrement=True, unique=True)  # Keep for compatibility
    portfolio_id = Column(String, ForeignKey("portfolios.portfolio_id", ondelete="CASCADE"), nullable=True)
    position_id = Column(String, nullable=True)  # Add this
    symbol = Column(String(64), nullable=False)
    
    # Add the missing fields that your API expects:
    weight = Column(Float, nullable=False)
    market_value = Column(Float, nullable=False)
    quantity = Column(Float, nullable=True)
    price = Column(Float, nullable=True)
    change_type = Column(String(50), nullable=True)  # Add this
    changed_fields = Column(JSON, nullable=True)     # Add this  
    previous_values = Column(JSON, nullable=True)    # Add this
    change_reason = Column(String(255), nullable=True)  # Add this
    changed_by = Column(String(255), nullable=True)     # Add this
    history_metadata = Column(JSON, nullable=True)      # Add this
    
    change_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

class PolicyDocument(Base):
    __tablename__ = "policy_documents"

    policy_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=True)
    filename = Column(String(255), nullable=False)
    content_hash = Column(String(128), nullable=False, unique=True)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.UPLOADED, nullable=False)
    
    # Add missing fields:
    document_type = Column(String(100), nullable=True)  # Add this
    jurisdiction = Column(String(100), nullable=True)   # Add this
    upload_date = Column(DateTime, default=datetime.utcnow, nullable=False)  # Add this
    effective_date = Column(DateTime, nullable=True)    # Add this
    expiry_date = Column(DateTime, nullable=True)       # Add this
    version = Column(Integer, default=1, nullable=False)  # Add this
    document_metadata = Column(JSON, nullable=True)     # Add this
    
    uploaded_by = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # parsed content
    paragraphs = Column(JSON, nullable=True)
    tables = Column(JSON, nullable=True)

    chunks = relationship("PolicyChunk", back_populates="document", cascade="all,delete-orphan")

class PolicyChunk(Base):
    __tablename__ = "policy_chunks"

    chunk_id = Column(String, primary_key=True)  # string/UUID
    policy_id = Column(
        String,
        ForeignKey("policy_documents.policy_id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)
    section_title = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)
    word_count = Column(Integer, nullable=True)
    char_count = Column(Integer, nullable=True)
    # Keep name as chunk_metadata to match your latest schema
    chunk_metadata = Column(JSON, nullable=True)

    document = relationship("PolicyDocument", back_populates="chunks")


# ---------- FastAPI dependency ----------
def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------- Manager used by app.main ----------
class DBManager:
    def __init__(self):
        self.engine = engine
        self.Base = Base
        self._SessionLocal = SessionLocal

    def create_tables(self):
        self.Base.metadata.create_all(self.engine)

    def init_demo_data(self):
        # Keep for compatibility; seed here if needed
        pass

    # NEW: fix for health check calling db_manager.get_session()
    def get_session(self) -> Session:
        """Return a new SQLAlchemy session (caller is responsible to close())."""
        return self._SessionLocal()

    # Optional: a tiny self-check you can call if useful
    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

db_manager = DBManager()

__all__ = [
    "get_db",
    "db_manager",
    "ComplianceRule",
    "ComplianceBreach",
    "ComplianceCase",
    "Portfolio",
    "RuleEvaluation",
    "PositionHistory",
    "PolicyDocument",
    "PolicyChunk",
    "BreachStatus",
    "CaseStatus",
    "CaseType",
    "RuleSeverity",
    "ControlType",
    "DocumentStatus",
]
