# Compliance System API

A comprehensive AI-powered compliance monitoring and rule management system for financial institutions. Built with FastAPI, featuring real-time portfolio monitoring, intelligent rule extraction, and LLM-powered policy analysis.

## =� Features

### Core Capabilities
- **Portfolio Management**: Track and manage investment positions with real-time compliance monitoring
- **Intelligent Rule Engine**: Define and evaluate complex compliance rules with multiple control types
- **AI-Powered Policy Analysis**: Extract compliance rules from natural language policy documents
- **Vector-Based Document Search**: Semantic search across policy knowledge base using ChromaDB
- **Real-time Breach Detection**: Continuous monitoring with automatic breach notifications
- **Comprehensive Analytics**: Advanced compliance reporting and performance metrics

### AI/LLM Integration
- **OpenAI GPT Integration**: Real LLM-powered policy analysis and rule extraction
- **Natural Language Q&A**: Ask questions about policies in plain English
- **Smart Rule Extraction**: Convert policy text into structured compliance rules
- **Semantic Document Search**: Find relevant policy sections using AI embeddings

### Security & Production Features
- **JWT Authentication**: Secure token-based authentication system
- **Rate Limiting**: IP-based rate limiting (60 requests/minute in production)
- **CORS Protection**: Configurable cross-origin resource sharing
- **Structured Logging**: Comprehensive audit trail and monitoring
- **Exception Handling**: Robust error handling with detailed logging

## =� Requirements

- Python 3.8+
- FastAPI
- SQLAlchemy
- ChromaDB (for vector search)
- OpenAI API key (for LLM features)

## =� Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd compliance_system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the environment template and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Security
SECRET_KEY=your-secure-secret-key-here
ALGORITHM=HS256

# LLM Configuration (Required for AI features)
OPENAI_API_KEY=your-openai-api-key-here
LLM_MODEL=gpt-4o-mini

# Database
DATABASE_URL=sqlite:///./compliance.db

# Vector Database
CHROMA_PERSIST_DIR=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 3. Setup Test Data

```bash
# Initialize database and create sample data
python scripts/setup_test_data.py
```

### 4. Run the Server

**Development Mode** (with mocks and debugging):
```bash
python simple_run.py
```

**Production Mode** (with real LLM services):
```bash
python run.py
```

The API will be available at: http://localhost:8000

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## =� API Examples

### Health Check

```bash
curl -X GET "http://localhost:8000/"
```

**Response**:
```json
{
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
  }
}
```

### Portfolio Management

**Get All Positions** (Working Endpoint):
```bash
curl -X GET "http://localhost:8000/api/portfolio/debug"
```

**Response**:
```json
{
  "AAPL": {
    "symbol": "AAPL",
    "weight": 0.078,
    "market_value": 150000.0,
    "portfolio_id": "PORT_001"
  },
  "MSFT": {
    "symbol": "MSFT", 
    "weight": 0.078,
    "market_value": 150000.0,
    "portfolio_id": "PORT_001"
  },
  "BRK.A": {
    "symbol": "BRK.A",
    "weight": 0.83,
    "market_value": 5000000.0,
    "portfolio_id": "PORT_002"
  }
}
```

**Add New Position**:
```bash
curl -X POST "http://localhost:8000/api/portfolio/DEMO123" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "DEMO123",
    "weight": 0.05,
    "market_value": 50000,
    "quantity": 1000,
    "price": 50.0,
    "sector": "Technology",
    "country": "US"
  }'
```

**Response**:
```json
{
  "success": true,
  "message": "Position DEMO123 created successfully",
  "timestamp": "2025-09-06T01:33:34.453868"
}
```

**Bulk Position Update**:
```bash
curl -X POST "http://localhost:8000/api/portfolio/bulk-update/debug" \
  -H "Content-Type: application/json" \
  -d '{"test": "data"}'
```

**Response**:
```json
{
  "success": true,
  "message": "Bulk update debug endpoint working",
  "received_data": {"test": "data"}
}
```

### AI-Powered Policy Analysis

**Ask Questions About Policies** (Natural Language Q&A):
```bash
curl -X POST "http://localhost:8000/api/policies/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the concentration limits for single issuers?",
    "max_results": 3
  }' | jq
```

**Example Response**:
```json
{
  "question": "What are the concentration limits for single issuers?",
  "answer": "The concentration limit for single issuers is that no single issuer shall represent more than five percent (5%) of the Fund's total net asset value.",
  "confidence": 0.85,
  "sources": [
    {
      "policy_id": "POL_001",
      "section": "Investment Policy Statement - Risk Management",
      "relevance_score": 0.92,
      "content_preview": "SECTION 4.2.1 - CONCENTRATION RISK MANAGEMENT..."
    }
  ]
}
```

**Extract Rules from Natural Language**:
```bash
curl -X POST "http://localhost:8000/api/rules/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "policy_text": "The fund must maintain at least 10% cash reserves and cannot invest more than 25% in technology stocks.",
    "context": "Investment Policy Section 3.1"
  }' | jq
```

**Example Response**:
```json
{
  "proposed_rules": [
    {
      "rule_id": "cash-reserve-minimum",
      "description": "The fund must maintain at least 10% cash reserves.",
      "control_type": "quant_limit",
      "severity": "high",
      "expression": {
        "metric": "cash_percentage",
        "operator": ">=",
        "threshold": 0.1
      },
      "confidence": 0.9
    }
  ],
  "total_rules_extracted": 2
}
```

### Semantic Document Search

```bash
curl -X POST "http://localhost:8000/api/policies/semantic-search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "concentration risk management",
    "max_results": 3
  }' | jq
```

### Compliance Monitoring

**Get Compliance Breaches** (Working Endpoint):
```bash
curl -X GET "http://localhost:8000/api/compliance/breaches"
```

**Response**:
```json
[
  {
    "breach_id": "dd98abb2-ec09-45cb-b8dd-8edb48ddc08c",
    "rule_id": "RULE_001",
    "rule_name": "Single Issuer Concentration Limit",
    "rule_description": "No single issuer shall exceed 5% of total portfolio NAV",
    "control_type": "ControlType.QUANT_LIMIT",
    "severity": "RuleSeverity.HIGH",
    "observed_value": 0.078,
    "threshold": 0.05,
    "breach_magnitude": 0.028,
    "breach_timestamp": "2025-09-05T23:05:58.568404",
    "status": "open",
    "age_hours": 2.4
  }
]
```

**Run Compliance Check** (Working Endpoint):
```bash
curl -X POST "http://localhost:8000/api/compliance/check/debug"
```

**Response**:
```json
{
  "success": true,
  "evaluation_id": "debug_check",
  "total_rules_checked": 6,
  "open_breaches": 4,
  "total_positions": 12,
  "timestamp": "2025-09-06T01:00:00Z"
}
```

### Rules Management

**Get All Rules** (Working Endpoint):
```bash
curl -X GET "http://localhost:8000/api/rules/debug"
```

**Response**:
```json
[
  {
    "rule_id": "RULE_001",
    "name": "Single Issuer Concentration Limit",
    "description": "No single issuer shall exceed 5% of total portfolio NAV",
    "control_type": "ControlType.QUANT_LIMIT",
    "severity": "RuleSeverity.HIGH",
    "is_active": true
  },
  {
    "rule_id": "RULE_002", 
    "name": "Sector Concentration Limit",
    "description": "No single sector shall exceed 25% of total portfolio NAV",
    "control_type": "ControlType.QUANT_LIMIT",
    "severity": "RuleSeverity.MEDIUM",
    "is_active": true
  }
]
```

**Create New Rule** (Working Endpoint):
```bash
curl -X POST "http://localhost:8000/api/rules/debug" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Demo Rule",
    "description": "Demo rule for testing POST endpoint",
    "expression": {
      "metric": "demo_metric",
      "operator": "<=",
      "threshold": 10.0,
      "scope": "portfolio"
    },
    "materiality_bps": 100
  }'
```

**Response**:
```json
{
  "success": true,
  "rule_id": "DEMO_BE4112B6",
  "message": "Rule DEMO_BE4112B6 created successfully"
}
```

### Analytics & Reporting

**Compliance Summary**:
```bash
curl -X GET "http://localhost:8000/api/analytics/compliance-summary" | jq
```

**Portfolio Analytics**:
```bash
curl -X GET "http://localhost:8000/api/analytics/portfolio-analytics" | jq
```

**Custom Report**:
```bash
curl -X POST "http://localhost:8000/api/analytics/custom-report" \
  -H "Content-Type: application/json" \
  -d '{
    "metrics": ["breach_count", "compliance_ratio"],
    "filters": {
      "date_range": {
        "start": "2025-08-01",
        "end": "2025-09-05"
      },
      "categories": ["CONCENTRATION"]
    },
    "format": "json"
  }' | jq
```

## <� Architecture

### Core Components

- **FastAPI Application** (`app/main.py`): Main API server with middleware
- **Database Models** (`app/models/`): SQLAlchemy ORM models for data persistence
- **API Routes** (`app/api/routes/`): RESTful endpoints organized by domain
- **Business Logic** (`app/services/`): Core compliance engine and LLM services
- **Configuration** (`app/config/`): Environment-based configuration management

### Key Services

- **Compliance Engine** (`app/services/compliance_engine.py`): Rule evaluation and breach detection
- **LLM Service** (`app/services/llm_service.py`): OpenAI integration for AI features
- **Vector Store** (`app/services/vector_store.py`): ChromaDB integration for semantic search
- **Policy Parser** (`app/services/policy_parser.py`): Document processing and analysis

### Database Schema

- **Portfolios**: Investment positions and metadata
- **Compliance Rules**: Configurable compliance rules with expressions
- **Compliance Breaches**: Detected violations with resolution tracking
- **Policy Documents**: Uploaded policy documents with vector embeddings
- **Rule Evaluations**: Historical rule evaluation results

## >� Testing

**Run the comprehensive test suite**:
```bash
./test.sh
```

This will test all API endpoints with realistic scenarios and generate detailed reports in `./test-results/`.

**Manual testing with specific endpoints**:
```bash
# Test health endpoints
curl -X GET "http://localhost:8000/health"

# Test LLM integration
curl -X POST "http://localhost:8000/api/policies/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the risk management policies?"}'
```

## =' Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `API_HOST` | Server bind address | `0.0.0.0` | No |
| `API_PORT` | Server port | `8000` | No |
| `DEBUG` | Enable debug mode | `false` | No |
| `SECRET_KEY` | JWT signing key | - | **Yes** |
| `OPENAI_API_KEY` | OpenAI API key for LLM features | - | **Yes** |
| `DATABASE_URL` | Database connection string | `sqlite:///./compliance.db` | No |
| `CHROMA_PERSIST_DIR` | ChromaDB storage directory | `./chroma_db` | No |

### Control Types Supported

- **QUANT_LIMIT**: Quantitative limits (concentration, exposure)
- **LIST_CONSTRAINT**: Allowed/prohibited securities, ratings
- **TEMPORAL_WINDOW**: Holding periods, lock-up requirements
- **PROCESS_CONTROL**: Approval workflows, documentation
- **REPORTING_DISCLOSURE**: Regulatory reporting obligations

## =� Monitoring & Logging

The system provides comprehensive logging and monitoring:

- **Structured Logging**: JSON-formatted logs in `./logs/compliance.log`
- **API Request Logging**: All requests logged with timing and status
- **Compliance Events**: Breach detection and resolution tracking
- **Performance Metrics**: Response times and system health metrics

**Log example**:
```json
{
  "timestamp": "2025-09-05T14:30:00Z",
  "level": "INFO",
  "event": "api_request",
  "method": "POST",
  "path": "/api/policies/ask",
  "status_code": 200,
  "duration_ms": 1250,
  "user_id": "user_12345"
}
```

## = Security

### Authentication

The system uses JWT (JSON Web Tokens) for authentication:

```bash
# Login would typically return a JWT token
# Use the token in subsequent requests:
curl -X GET "http://localhost:8000/api/portfolio" \
  -H "Authorization: Bearer your-jwt-token-here"
```

### Rate Limiting

- **Production**: 60 requests per minute per IP address
- **Development**: Rate limiting disabled for easier testing

### CORS Policy

- **Development**: Allows localhost origins (ports 3000, 8080, 5173)
- **Production**: Configure specific allowed origins in `app/main.py`

## =� Production Deployment

### Prerequisites

1. **Secure Environment Variables**: Never commit API keys or secrets
2. **Database**: Consider PostgreSQL for production workloads
3. **HTTPS**: Use reverse proxy (nginx/Apache) with SSL certificates
4. **Monitoring**: Set up log aggregation and alerting

### Production Checklist

- [ ] Set `DEBUG=false` and `API_RELOAD=false`
- [ ] Configure real frontend domains in CORS settings
- [ ] Set up proper database (PostgreSQL recommended)
- [ ] Configure log rotation and monitoring
- [ ] Set up backup procedures for database and ChromaDB
- [ ] Configure rate limiting based on your needs
- [ ] Set up health check monitoring

### Docker Deployment (Optional)

```dockerfile
# Basic Dockerfile structure
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "run.py"]
```

## =� Support

### Getting Help

- **API Documentation**: Visit `/docs` when server is running
- **Health Status**: Check `/health` endpoint for system status
- **Logs**: Check `./logs/compliance.log` for detailed error information

### Common Issues

**"LLM Service not available"**: Ensure `OPENAI_API_KEY` is set correctly
**"Database connection failed"**: Check `DATABASE_URL` and permissions
**"Vector store initialization failed"**: Verify ChromaDB directory permissions

## =� License

[Your License Here]

## > Contributing

[Your Contributing Guidelines Here]

---

**Built with FastAPI, SQLAlchemy, ChromaDB, and OpenAI GPT** =�