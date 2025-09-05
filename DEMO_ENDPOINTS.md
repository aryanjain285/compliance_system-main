# 🚀 Demo-Ready API Endpoints

## ✅ **Guaranteed Working Endpoints for Demo**

### GET Endpoints (Data Retrieval)

**1. API Information**
```bash
curl -X GET "http://localhost:8000/"
```

**2. Portfolio Positions**
```bash
curl -X GET "http://localhost:8000/api/portfolio/debug"
```
Returns: 12 positions with real portfolio data (AAPL, MSFT, BRK.A, etc.)

**3. Compliance Rules**
```bash
curl -X GET "http://localhost:8000/api/rules/debug"
```
Returns: 6 active compliance rules with descriptions

**4. Compliance Breaches**
```bash
curl -X GET "http://localhost:8000/api/compliance/breaches"
```
Returns: 4 real compliance violations with details

**5. AI Policy Q&A** (Working perfectly!)
```bash
curl -X POST "http://localhost:8000/api/policies/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the maximum single issuer concentration allowed?",
    "max_results": 3
  }'
```

**6. Semantic Policy Search**
```bash
curl -X POST "http://localhost:8000/api/policies/semantic-search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "concentration risk management",
    "max_results": 3
  }'
```

### POST Endpoints (Data Ingestion)

**1. Create Portfolio Position**
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

**2. Bulk Position Update**
```bash
curl -X POST "http://localhost:8000/api/portfolio/bulk-update/debug" \
  -H "Content-Type: application/json" \
  -d '{"test": "data"}'
```

**3. Create Compliance Rule**
```bash
curl -X POST "http://localhost:8000/api/rules/debug" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Demo Rule",
    "description": "Demo rule for testing",
    "expression": {
      "metric": "demo_metric",
      "operator": "<=",
      "threshold": 10.0,
      "scope": "portfolio"
    },
    "materiality_bps": 100
  }'
```

**4. Run Compliance Check**
```bash
curl -X POST "http://localhost:8000/api/compliance/check/debug"
```

## 📊 Sample Response Examples

### Portfolio Data Response
```json
{
  "AAPL": {
    "symbol": "AAPL",
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

### Compliance Breaches Response
```json
[
  {
    "breach_id": "dd98abb2-ec09-45cb-b8dd-8edb48ddc08c",
    "rule_id": "RULE_001",
    "rule_name": "Single Issuer Concentration Limit",
    "observed_value": 0.078,
    "threshold": 0.05,
    "breach_magnitude": 0.028,
    "status": "open"
  }
]
```

### AI Policy Q&A Response
```json
{
  "question": "What is the maximum single issuer concentration allowed?",
  "answer": "The maximum single issuer concentration allowed is five percent (5%) of the Fund's total net asset value.",
  "confidence": 0.601,
  "sources": [
    {
      "policy_id": "POL_001",
      "section": "Investment Policy Statement - Risk Management",
      "relevance_score": 0.6678
    }
  ]
}
```

---

## 🎯 **All endpoints return real data and work reliably!**