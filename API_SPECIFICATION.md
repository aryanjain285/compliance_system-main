# Compliance System API Specification

## Overview

The Compliance System API provides comprehensive endpoints for managing investment compliance, portfolio monitoring, rule management, and regulatory reporting. The system includes LLM-powered policy analysis and vector-based document search capabilities.

**Base URL**: `http://localhost:8000`  
**API Version**: v1  
**Content-Type**: `application/json`

## Authentication

Currently, the API uses development mode. In production, implement JWT-based authentication.

## Core Endpoints

### Health & System Status

#### `GET /`
**Description**: Basic health check and system information
**Response**:
```json
{
  "message": "Compliance System API is running",
  "timestamp": "2025-09-05T12:30:00Z",
  "version": "1.0.0"
}
```

#### `GET /api/config`
**Description**: System configuration and feature flags
**Response**:
```json
{
  "features": {
    "llm_service": true,
    "vector_store": true,
    "file_upload": true,
    "email_notifications": true
  },
  "limits": {
    "max_file_size": 50,
    "max_results": 100
  }
}
```

---

## Portfolio Management (`/api/portfolio`)

### Portfolio Positions

#### `GET /api/portfolio`
**Description**: Get all portfolio positions
**Response**:
```json
{
  "AAPL": {
    "symbol": "AAPL",
    "quantity": 1000,
    "market_value": 150000.00,
    "weight": 15.0,
    "sector": "Technology",
    "last_updated": "2025-09-05T12:00:00Z"
  }
}
```

#### `POST /api/portfolio/{symbol}`
**Description**: Add or update a position
**Path Parameters**: 
- `symbol` (string): Stock symbol
**Request Body**:
```json
{
  "quantity": 1000,
  "purchase_price": 150.00,
  "sector": "Technology",
  "country": "US"
}
```

#### `PATCH /api/portfolio/{symbol}`
**Description**: Update existing position
**Path Parameters**:
- `symbol` (string): Stock symbol
**Request Body**:
```json
{
  "quantity": 1200
}
```

#### `DELETE /api/portfolio/{symbol}`
**Description**: Remove position from portfolio
**Path Parameters**:
- `symbol` (string): Stock symbol

#### `GET /api/portfolio/{symbol}`
**Description**: Get specific position details
**Path Parameters**:
- `symbol` (string): Stock symbol

#### `GET /api/portfolio/{symbol}/history`
**Description**: Get position history and changes
**Path Parameters**:
- `symbol` (string): Stock symbol

#### `GET /api/portfolio/summary/overview`
**Description**: Get portfolio summary and analytics
**Response**:
```json
{
  "total_value": 1000000.00,
  "total_positions": 25,
  "sector_breakdown": {
    "Technology": 30.0,
    "Healthcare": 25.0,
    "Finance": 20.0
  },
  "geographic_breakdown": {
    "US": 70.0,
    "Europe": 20.0,
    "Asia": 10.0
  }
}
```

#### `POST /api/portfolio/bulk-update`
**Description**: Update multiple positions at once
**Request Body**:
```json
{
  "updates": [
    {
      "symbol": "AAPL",
      "quantity": 1000,
      "market_price": 150.00
    },
    {
      "symbol": "GOOGL", 
      "quantity": 500,
      "market_price": 2800.00
    }
  ]
}
```

---

## Compliance Management (`/api/compliance`)

### Compliance Checks

#### `POST /api/compliance/check`
**Description**: Run comprehensive compliance check
**Request Body**:
```json
{
  "rule_ids": ["RULE_001", "RULE_002"],
  "portfolio_date": "2025-09-05"
}
```
**Response**:
```json
{
  "overall_status": "COMPLIANT",
  "total_rules_checked": 5,
  "passed": 5,
  "failed": 0,
  "warnings": 1,
  "results": [
    {
      "rule_id": "RULE_001",
      "status": "COMPLIANT",
      "message": "Single issuer limits satisfied"
    }
  ]
}
```

#### `GET /api/compliance/rules/{rule_id}/evaluate`
**Description**: Evaluate specific rule against current portfolio
**Path Parameters**:
- `rule_id` (string): Rule identifier

#### `GET /api/compliance/status`
**Description**: Get overall compliance status
**Response**:
```json
{
  "status": "COMPLIANT",
  "last_check": "2025-09-05T12:30:00Z",
  "active_breaches": 0,
  "total_rules": 5,
  "compliant_rules": 5
}
```

### Breach Management

#### `GET /api/compliance/breaches`
**Description**: Get all compliance breaches
**Query Parameters**:
- `status` (string, optional): Filter by status (ACTIVE, RESOLVED, PENDING)
- `severity` (string, optional): Filter by severity (LOW, MEDIUM, HIGH, CRITICAL)
**Response**:
```json
{
  "data": [
    {
      "breach_id": "BREACH_001",
      "rule_id": "RULE_001",
      "status": "ACTIVE",
      "severity": "HIGH",
      "detected_at": "2025-09-05T10:00:00Z",
      "description": "Single issuer concentration exceeded"
    }
  ]
}
```

#### `GET /api/compliance/breaches/{breach_id}`
**Description**: Get detailed breach information
**Path Parameters**:
- `breach_id` (string): Breach identifier

#### `POST /api/compliance/breaches/{breach_id}/resolve`
**Description**: Mark breach as resolved
**Path Parameters**:
- `breach_id` (string): Breach identifier
**Request Body**:
```json
{
  "resolution_notes": "Position reduced to comply with limits",
  "resolved_by": "admin"
}
```

#### `POST /api/compliance/breaches/{breach_id}/explain`
**Description**: Get LLM-powered explanation of breach
**Path Parameters**:
- `breach_id` (string): Breach identifier
**Response**:
```json
{
  "breach_id": "BREACH_001",
  "explanation": "This breach occurred because the concentration in AAPL (15.2%) exceeded the maximum single issuer limit of 15%. This violates diversification requirements designed to limit portfolio risk.",
  "recommendations": [
    "Reduce AAPL position to below 15% of portfolio value",
    "Consider rebalancing into other technology stocks"
  ],
  "severity_reasoning": "High severity due to significant risk concentration",
  "regulatory_context": "Violates SEC diversification requirements..."
}
```

### Case Management

#### `GET /api/compliance/cases`
**Description**: Get all compliance cases
**Response**:
```json
{
  "data": [
    {
      "case_id": "CASE_001",
      "title": "Concentration Risk Review",
      "status": "OPEN",
      "priority": "HIGH",
      "created_at": "2025-09-05T10:00:00Z"
    }
  ]
}
```

#### `POST /api/compliance/simulate`
**Description**: Simulate compliance check with hypothetical changes
**Request Body**:
```json
{
  "position_changes": [
    {
      "symbol": "AAPL",
      "new_quantity": 800
    }
  ],
  "rule_ids": ["RULE_001"]
}
```

---

## Rules Management (`/api/rules`)

### Rule CRUD Operations

#### `GET /api/rules`
**Description**: Get all compliance rules
**Query Parameters**:
- `category` (string, optional): Filter by category
- `active` (boolean, optional): Filter by active status
**Response**:
```json
{
  "data": [
    {
      "rule_id": "RULE_001",
      "name": "Single Issuer Concentration Limit",
      "description": "No single issuer should exceed 5% of portfolio value",
      "category": "CONCENTRATION",
      "severity": "HIGH",
      "active": true,
      "created_at": "2025-09-01T00:00:00Z"
    }
  ]
}
```

#### `POST /api/rules`
**Description**: Create new compliance rule
**Request Body**:
```json
{
  "name": "Sector Concentration Limit",
  "description": "No single sector should exceed 25% of portfolio",
  "category": "CONCENTRATION",
  "severity": "MEDIUM",
  "rule_expression": {
    "type": "percentage_limit",
    "field": "sector_weight",
    "operator": "<=",
    "threshold": 25.0
  },
  "active": true
}
```

#### `GET /api/rules/{rule_id}`
**Description**: Get specific rule details
**Path Parameters**:
- `rule_id` (string): Rule identifier

#### `PUT /api/rules/{rule_id}`
**Description**: Update existing rule
**Path Parameters**:
- `rule_id` (string): Rule identifier
**Request Body**: Same as POST /api/rules

#### `DELETE /api/rules/{rule_id}`
**Description**: Delete rule
**Path Parameters**:
- `rule_id` (string): Rule identifier

### Advanced Rule Operations

#### `POST /api/rules/extract`
**Description**: Extract rules from natural language using LLM
**Request Body**:
```json
{
  "policy_text": "The fund shall not invest more than 5% of its total assets in securities of any single issuer.",
  "context": "Investment Policy Statement Section 4.2"
}
```
**Response**:
```json
{
  "extracted_rules": [
    {
      "name": "Single Issuer Limit",
      "rule_expression": {
        "type": "percentage_limit",
        "field": "issuer_weight",
        "operator": "<=",
        "threshold": 5.0
      },
      "confidence": 0.95
    }
  ]
}
```

#### `POST /api/rules/approve`
**Description**: Approve extracted rules for activation
**Request Body**:
```json
{
  "rule_ids": ["EXTRACTED_001", "EXTRACTED_002"],
  "approved_by": "admin"
}
```

#### `GET /api/rules/{rule_id}/yaml`
**Description**: Get rule in YAML format
**Path Parameters**:
- `rule_id` (string): Rule identifier

#### `PUT /api/rules/{rule_id}/yaml`
**Description**: Update rule from YAML
**Path Parameters**:
- `rule_id` (string): Rule identifier
**Request Body**:
```yaml
name: "Updated Rule Name"
description: "Updated description"
rule_expression:
  type: "percentage_limit"
  field: "issuer_weight"
  operator: "<="
  threshold: 4.0
```

#### `POST /api/rules/{rule_id}/clone`
**Description**: Clone existing rule
**Path Parameters**:
- `rule_id` (string): Rule identifier to clone
**Request Body**:
```json
{
  "new_name": "Cloned Rule Name",
  "modifications": {
    "threshold": 3.0
  }
}
```

#### `GET /api/rules/templates/control-types`
**Description**: Get available rule template types
**Response**:
```json
{
  "templates": [
    "percentage_limit",
    "absolute_limit", 
    "ratio_check",
    "categorical_restriction"
  ]
}
```

---

## Policy Management (`/api/policies`)

### Document Management

#### `POST /api/policies/upload`
**Description**: Upload policy document for processing
**Content-Type**: `multipart/form-data`
**Request Body**:
- `file`: PDF file
- `title`: Document title
- `category`: Document category
- `metadata`: JSON metadata (optional)

#### `GET /api/policies`
**Description**: Get all policy documents
**Response**:
```json
{
  "data": [
    {
      "policy_id": "POL_001",
      "title": "Investment Policy Statement",
      "category": "INVESTMENT",
      "upload_date": "2025-09-01T00:00:00Z",
      "status": "PROCESSED",
      "chunk_count": 25
    }
  ]
}
```

#### `GET /api/policies/{policy_id}`
**Description**: Get specific policy document
**Path Parameters**:
- `policy_id` (string): Policy identifier

#### `PUT /api/policies/{policy_id}/metadata`
**Description**: Update policy metadata
**Path Parameters**:
- `policy_id` (string): Policy identifier
**Request Body**:
```json
{
  "title": "Updated Policy Title",
  "category": "COMPLIANCE",
  "tags": ["risk", "concentration"]
}
```

#### `DELETE /api/policies/{policy_id}`
**Description**: Delete policy document
**Path Parameters**:
- `policy_id` (string): Policy identifier

### Search and Analysis

#### `POST /api/policies/{policy_id}/search`
**Description**: Search within specific policy document
**Path Parameters**:
- `policy_id` (string): Policy identifier
**Request Body**:
```json
{
  "query": "concentration limits",
  "max_results": 5
}
```

#### `POST /api/policies/semantic-search`
**Description**: Semantic search across all policy documents
**Request Body**:
```json
{
  "query": "issuer concentration risk limits",
  "max_results": 10,
  "min_relevance": 0.5
}
```
**Response**:
```json
{
  "query": "issuer concentration risk limits",
  "results": [
    {
      "policy_id": "POL_001",
      "chunk_id": "CHUNK_001",
      "content": "No single issuer shall represent more than 5% of total portfolio value...",
      "relevance_score": 0.85,
      "section": "Section 4.2.1"
    }
  ],
  "total_results": 5
}
```

#### `POST /api/policies/ask` 🤖
**Description**: Natural language question answering using LLM
**Request Body**:
```json
{
  "question": "What are the issuer concentration limits and why are they important?",
  "max_results": 3,
  "include_context": true
}
```
**Response**:
```json
{
  "question": "What are the issuer concentration limits?",
  "answer": "The issuer concentration limits are as follows: No single issuer should represent more than five percent (5%) of the Fund's total net asset value. These limits are important for risk management as they ensure diversification...",
  "confidence": 0.85,
  "sources": [
    {
      "policy_id": "POL_001",
      "section": "Risk Management",
      "relevance_score": 0.92
    }
  ],
  "context_used": 3
}
```

### Document Processing

#### `GET /api/policies/{policy_id}/chunks`
**Description**: Get document chunks for a policy
**Path Parameters**:
- `policy_id` (string): Policy identifier

#### `GET /api/policies/stats/processing`
**Description**: Get document processing statistics
**Response**:
```json
{
  "total_documents": 5,
  "processed": 5,
  "processing": 0,
  "failed": 0,
  "total_chunks": 125,
  "average_chunk_size": 500
}
```

---

## Analytics & Reporting (`/api/analytics`)

### Compliance Analytics

#### `GET /api/analytics/compliance-summary`
**Description**: Get compliance summary analytics
**Query Parameters**:
- `period` (string): Time period (daily, weekly, monthly)
- `start_date` (string, optional): Start date (ISO format)
- `end_date` (string, optional): End date (ISO format)

#### `GET /api/analytics/breach-analysis`
**Description**: Detailed breach analysis and trends
**Response**:
```json
{
  "summary": {
    "total_breaches": 15,
    "resolved": 12,
    "active": 3,
    "average_resolution_time": 2.5
  },
  "by_severity": {
    "CRITICAL": 2,
    "HIGH": 5,
    "MEDIUM": 6,
    "LOW": 2
  },
  "trends": {
    "weekly_change": -20.0,
    "most_common_category": "CONCENTRATION"
  }
}
```

#### `GET /api/analytics/performance-metrics`
**Description**: System performance and processing metrics

#### `GET /api/analytics/portfolio-analytics`
**Description**: Advanced portfolio analytics
**Response**:
```json
{
  "risk_metrics": {
    "var_95": 0.025,
    "max_drawdown": 0.15,
    "sharpe_ratio": 1.2
  },
  "concentration_analysis": {
    "herfindahl_index": 0.08,
    "top_5_holdings": 0.45,
    "sector_concentration": 0.35
  },
  "diversification_metrics": {
    "effective_stocks": 25.5,
    "correlation_avg": 0.3
  }
}
```

### Regulatory Reporting

#### `GET /api/analytics/regulatory-report`
**Description**: Generate regulatory compliance report
**Query Parameters**:
- `report_type` (string): Type of report (monthly, quarterly, annual)
- `format` (string): Output format (json, pdf, excel)

#### `POST /api/analytics/custom-report`
**Description**: Generate custom analytics report
**Request Body**:
```json
{
  "metrics": ["breach_count", "compliance_ratio", "risk_metrics"],
  "filters": {
    "date_range": {
      "start": "2025-08-01",
      "end": "2025-09-01"
    },
    "categories": ["CONCENTRATION", "LIQUIDITY"]
  },
  "format": "json"
}
```

---

## Response Formats

### Standard Response Structure
```json
{
  "success": true,
  "data": {}, 
  "message": "Operation completed successfully",
  "timestamp": "2025-09-05T12:30:00Z",
  "request_id": "req-123456"
}
```

### Error Response Structure
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "quantity",
      "issue": "Must be greater than 0"
    }
  },
  "timestamp": "2025-09-05T12:30:00Z",
  "request_id": "req-123456"
}
```

---

## Status Codes

- **200**: Success
- **201**: Created
- **400**: Bad Request
- **401**: Unauthorized  
- **404**: Not Found
- **422**: Validation Error
- **500**: Internal Server Error

---

## Rate Limits

- Standard endpoints: 1000 requests/hour
- LLM endpoints (/api/policies/ask): 100 requests/hour  
- File upload: 10 uploads/hour

---

## Data Models

### Key Entities

**Position**:
```json
{
  "symbol": "AAPL",
  "quantity": 1000,
  "market_price": 150.00,
  "market_value": 150000.00,
  "weight": 15.0,
  "sector": "Technology",
  "country": "US",
  "currency": "USD"
}
```

**Compliance Rule**:
```json
{
  "rule_id": "RULE_001", 
  "name": "Single Issuer Limit",
  "description": "Maximum 5% per issuer",
  "category": "CONCENTRATION",
  "severity": "HIGH",
  "rule_expression": {
    "type": "percentage_limit",
    "field": "issuer_weight", 
    "operator": "<=",
    "threshold": 5.0
  },
  "active": true
}
```

**Breach**:
```json
{
  "breach_id": "BREACH_001",
  "rule_id": "RULE_001", 
  "status": "ACTIVE",
  "severity": "HIGH",
  "current_value": 15.2,
  "threshold_value": 5.0,
  "affected_positions": ["AAPL"],
  "detected_at": "2025-09-05T10:00:00Z"
}
```

---

## WebSocket Endpoints (Future)

Real-time updates for:
- `/ws/compliance-status` - Real-time compliance monitoring
- `/ws/breach-alerts` - Live breach notifications
- `/ws/portfolio-updates` - Portfolio change notifications

---

*Generated on: 2025-09-05*  
*Version: 1.0.0*