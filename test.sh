#!/bin/bash

# Compliance System API Test Suite
# Comprehensive testing of all API endpoints

set -e

# Configuration
BASE_URL="${BASE_URL:-http://localhost:8000}"
VERBOSE="${VERBOSE:-false}"
OUTPUT_DIR="./test-results"
TEST_FILE_PATH="./test-sample.pdf"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED_TESTS++))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED_TESTS++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Test execution function
run_test() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"
    local expected_status="$5"
    local description="$6"
    
    ((TOTAL_TESTS++))
    
    log_info "Testing: $name - $description"
    
    local cmd="curl -s -w 'HTTP_STATUS:%{http_code}\\nTIME:%{time_total}\\n' -X $method '$BASE_URL$endpoint'"
    
    if [ "$method" = "POST" ] || [ "$method" = "PUT" ] || [ "$method" = "PATCH" ]; then
        if [ -n "$data" ]; then
            cmd="$cmd -H 'Content-Type: application/json' -d '$data'"
        fi
    fi
    
    if [ "$VERBOSE" = "true" ]; then
        log_info "Command: $cmd"
    fi
    
    local response=$(eval "$cmd" 2>&1)
    local status=$(echo "$response" | grep "HTTP_STATUS:" | cut -d: -f2)
    local time=$(echo "$response" | grep "TIME:" | cut -d: -f2)
    local body=$(echo "$response" | grep -v "HTTP_STATUS:" | grep -v "TIME:")
    
    # Save response to file
    local safe_name=$(echo "$name" | tr ' ' '_' | tr -cd '[:alnum:]_-')
    echo "$body" > "$OUTPUT_DIR/${safe_name}_response.json"
    
    if [ "$status" = "$expected_status" ]; then
        log_success "$name (${status}) - ${time}s"
    else
        log_error "$name - Expected: $expected_status, Got: $status"
        if [ "$VERBOSE" = "true" ]; then
            echo "Response: $body"
        fi
    fi
}

# Create a sample test file for upload tests
create_test_file() {
    if [ ! -f "$TEST_FILE_PATH" ]; then
        log_info "Creating test file for upload tests..."
        echo "Test policy document content for compliance system testing." > "$TEST_FILE_PATH"
        echo "This file contains sample policy text for testing purposes." >> "$TEST_FILE_PATH"
    fi
}

# Wait for server to be ready
wait_for_server() {
    log_info "Waiting for server at $BASE_URL to be ready..."
    local retries=30
    while [ $retries -gt 0 ]; do
        if curl -s "$BASE_URL/" > /dev/null 2>&1; then
            log_success "Server is ready!"
            break
        fi
        ((retries--))
        sleep 2
    done
    
    if [ $retries -eq 0 ]; then
        log_error "Server not responding after 60 seconds"
        exit 1
    fi
}

# Header
echo "============================================="
echo "  Compliance System API Test Suite"
echo "============================================="
echo "Base URL: $BASE_URL"
echo "Output Dir: $OUTPUT_DIR"
echo "Verbose: $VERBOSE"
echo "============================================="

# Wait for server
wait_for_server

# Create test file
create_test_file

echo ""
log_info "Starting API tests..."

# =============================================================================
# HEALTH & SYSTEM STATUS TESTS
# =============================================================================

echo ""
echo "=== HEALTH & SYSTEM STATUS ==="

run_test "Root Health Check" "GET" "/" "" "200" "Basic system health"
run_test "API Config" "GET" "/api/config" "" "200" "System configuration"

# =============================================================================
# PORTFOLIO MANAGEMENT TESTS  
# =============================================================================

echo ""
echo "=== PORTFOLIO MANAGEMENT ==="

# Portfolio Overview
run_test "Get All Positions" "GET" "/api/portfolio" "" "200" "Retrieve all portfolio positions"
run_test "Portfolio Summary" "GET" "/api/portfolio/summary/overview" "" "200" "Portfolio summary analytics"

# Position Management
run_test "Add Position" "POST" "/api/portfolio/AAPL" '{
    "quantity": 1000,
    "purchase_price": 150.00,
    "sector": "Technology",
    "country": "US"
}' "200" "Add new AAPL position"

run_test "Update Position" "PATCH" "/api/portfolio/AAPL" '{
    "quantity": 1200
}' "200" "Update AAPL quantity"

run_test "Get Specific Position" "GET" "/api/portfolio/AAPL" "" "200" "Get AAPL position details"
run_test "Get Position History" "GET" "/api/portfolio/AAPL/history" "" "200" "Get AAPL position history"

# Bulk Operations
run_test "Bulk Update Positions" "POST" "/api/portfolio/bulk-update" '{
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
}' "200" "Bulk update multiple positions"

# Add more test positions for compliance testing
run_test "Add GOOGL Position" "POST" "/api/portfolio/GOOGL" '{
    "quantity": 500,
    "purchase_price": 2800.00,
    "sector": "Technology",
    "country": "US"
}' "200" "Add GOOGL position"

run_test "Add MSFT Position" "POST" "/api/portfolio/MSFT" '{
    "quantity": 800,
    "purchase_price": 300.00,
    "sector": "Technology", 
    "country": "US"
}' "200" "Add MSFT position"

# =============================================================================
# RULES MANAGEMENT TESTS
# =============================================================================

echo ""
echo "=== RULES MANAGEMENT ==="

# Rule CRUD
run_test "Get All Rules" "GET" "/api/rules" "" "200" "Retrieve all compliance rules"
run_test "Get Rules by Category" "GET" "/api/rules?category=CONCENTRATION" "" "200" "Filter rules by category"

run_test "Create New Rule" "POST" "/api/rules" '{
    "name": "Test Sector Concentration Limit",
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
}' "200" "Create new sector limit rule"

# Let the system process the rule creation
sleep 2

# Get the created rule (assuming RULE_TEST_001 or similar ID)
run_test "Get Specific Rule" "GET" "/api/rules/RULE_001" "" "200" "Get specific rule details"
run_test "Get Rule YAML" "GET" "/api/rules/RULE_001/yaml" "" "200" "Get rule in YAML format"

# Rule extraction from natural language
run_test "Extract Rules from Text" "POST" "/api/rules/extract" '{
    "policy_text": "The fund shall not invest more than 10% of its total assets in securities of any single issuer.",
    "context": "Investment Policy Statement Section 4.2"
}' "200" "Extract rules from natural language"

run_test "Get Rule Templates" "GET" "/api/rules/templates/control-types" "" "200" "Get available rule templates"

# =============================================================================
# COMPLIANCE MANAGEMENT TESTS
# =============================================================================

echo ""
echo "=== COMPLIANCE MANAGEMENT ==="

# Compliance checks
run_test "Compliance Status" "GET" "/api/compliance/status" "" "200" "Get overall compliance status"

run_test "Run Compliance Check" "POST" "/api/compliance/check" '{
    "rule_ids": [],
    "portfolio_date": "2025-09-05"
}' "200" "Run comprehensive compliance check"

run_test "Evaluate Specific Rule" "GET" "/api/compliance/rules/RULE_001/evaluate" "" "200" "Evaluate single rule"

# Breach management  
run_test "Get All Breaches" "GET" "/api/compliance/breaches" "" "200" "Retrieve all compliance breaches"
run_test "Get Active Breaches" "GET" "/api/compliance/breaches?status=ACTIVE" "" "200" "Filter active breaches"

# Case management
run_test "Get Compliance Cases" "GET" "/api/compliance/cases" "" "200" "Retrieve compliance cases"

# Compliance simulation
run_test "Simulate Position Changes" "POST" "/api/compliance/simulate" '{
    "position_changes": [
        {
            "symbol": "AAPL",
            "new_quantity": 800
        }
    ],
    "rule_ids": ["RULE_001"]
}' "200" "Simulate compliance with position changes"

# =============================================================================
# POLICY MANAGEMENT TESTS
# =============================================================================

echo ""
echo "=== POLICY MANAGEMENT ==="

# Document management
run_test "Get All Policies" "GET" "/api/policies" "" "200" "Retrieve all policy documents"
run_test "Policy Processing Stats" "GET" "/api/policies/stats/processing" "" "200" "Get document processing stats"

# Search functionality
run_test "Semantic Search" "POST" "/api/policies/semantic-search" '{
    "query": "issuer concentration risk limits",
    "max_results": 5,
    "min_relevance": 0.3
}' "200" "Semantic search across policies"

# Natural language Q&A (LLM endpoint)
run_test "Policy Question Answering" "POST" "/api/policies/ask" '{
    "question": "What are the issuer concentration limits and why are they important for risk management?",
    "max_results": 3,
    "include_context": true
}' "200" "LLM-powered policy question answering"

run_test "Another Policy Query" "POST" "/api/policies/ask" '{
    "question": "What are the sector diversification requirements?",
    "max_results": 3
}' "200" "Query about sector diversification"

# Document-specific operations (test with existing policy)
run_test "Get Specific Policy" "GET" "/api/policies/POL_001" "" "200" "Get specific policy document"
run_test "Search Within Policy" "POST" "/api/policies/POL_001/search" '{
    "query": "concentration limits",
    "max_results": 5
}' "200" "Search within specific policy"

run_test "Get Policy Chunks" "GET" "/api/policies/POL_001/chunks" "" "200" "Get policy document chunks"

# =============================================================================
# ANALYTICS & REPORTING TESTS
# =============================================================================

echo ""
echo "=== ANALYTICS & REPORTING ==="

run_test "Compliance Summary" "GET" "/api/analytics/compliance-summary" "" "200" "Compliance summary analytics"
run_test "Compliance Summary Weekly" "GET" "/api/analytics/compliance-summary?period=weekly" "" "200" "Weekly compliance summary"

run_test "Breach Analysis" "GET" "/api/analytics/breach-analysis" "" "200" "Detailed breach analysis"
run_test "Performance Metrics" "GET" "/api/analytics/performance-metrics" "" "200" "System performance metrics"
run_test "Portfolio Analytics" "GET" "/api/analytics/portfolio-analytics" "" "200" "Advanced portfolio analytics"

run_test "Regulatory Report" "GET" "/api/analytics/regulatory-report?report_type=monthly&format=json" "" "200" "Generate regulatory report"

run_test "Custom Report" "POST" "/api/analytics/custom-report" '{
    "metrics": ["breach_count", "compliance_ratio"],
    "filters": {
        "date_range": {
            "start": "2025-08-01",
            "end": "2025-09-05"
        },
        "categories": ["CONCENTRATION"]
    },
    "format": "json"
}' "200" "Generate custom analytics report"

# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

echo ""
echo "=== ERROR HANDLING ==="

run_test "Invalid Endpoint" "GET" "/api/nonexistent" "" "404" "Test 404 handling"
run_test "Invalid Position Symbol" "GET" "/api/portfolio/INVALID123" "" "404" "Invalid position lookup"
run_test "Invalid Rule ID" "GET" "/api/rules/NONEXISTENT" "" "404" "Invalid rule lookup"
run_test "Invalid Policy ID" "GET" "/api/policies/NONEXISTENT" "" "404" "Invalid policy lookup"

run_test "Invalid JSON Data" "POST" "/api/rules" '{invalid json}' "422" "Invalid JSON handling"
run_test "Missing Required Fields" "POST" "/api/portfolio/TEST" '{}' "422" "Missing required field validation"

# =============================================================================
# BREACH TESTING (if breaches exist)
# =============================================================================

echo ""
echo "=== BREACH MANAGEMENT (if breaches exist) ==="

# Get breaches first to test breach-specific endpoints
BREACHES_RESPONSE=$(curl -s "$BASE_URL/api/compliance/breaches")
BREACH_ID=$(echo "$BREACHES_RESPONSE" | grep -o '"breach_id":"[^"]*"' | head -1 | cut -d'"' -f4)

if [ -n "$BREACH_ID" ]; then
    log_info "Found breach ID: $BREACH_ID"
    run_test "Get Specific Breach" "GET" "/api/compliance/breaches/$BREACH_ID" "" "200" "Get specific breach details"
    run_test "Explain Breach" "POST" "/api/compliance/breaches/$BREACH_ID/explain" "" "200" "LLM explanation of breach"
else
    log_warning "No breaches found for breach-specific testing"
fi

# =============================================================================
# CLEANUP TESTS (Optional)
# =============================================================================

echo ""
echo "=== CLEANUP (Optional) ==="

# Clean up test positions (optional - comment out to preserve test data)
# run_test "Delete AAPL Position" "DELETE" "/api/portfolio/AAPL" "" "200" "Remove AAPL position"
# run_test "Delete GOOGL Position" "DELETE" "/api/portfolio/GOOGL" "" "200" "Remove GOOGL position"  
# run_test "Delete MSFT Position" "DELETE" "/api/portfolio/MSFT" "" "200" "Remove MSFT position"

# =============================================================================
# RESULTS SUMMARY
# =============================================================================

echo ""
echo "============================================="
echo "           TEST RESULTS SUMMARY"
echo "============================================="
echo "Total Tests:  $TOTAL_TESTS"
echo -e "Passed Tests: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed Tests: ${RED}$FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
    SUCCESS_RATE="100%"
else
    SUCCESS_RATE=$(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l)
    echo -e "${YELLOW}Success Rate: $SUCCESS_RATE%${NC}"
fi

echo "============================================="
echo "Test results saved to: $OUTPUT_DIR"
echo "Server tested: $BASE_URL"
echo "Test completed at: $(date)"
echo "============================================="

# Exit with error code if tests failed
if [ $FAILED_TESTS -gt 0 ]; then
    exit 1
else
    exit 0
fi