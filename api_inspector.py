#!/usr/bin/env python3
"""
API Route Inspector for Compliance System
This script helps debug what endpoints are actually available
"""

import requests
import json
import sys
from urllib.parse import urljoin

def test_endpoint(base_url, method, endpoint, description, data=None):
    """Test a single endpoint and return results"""
    url = urljoin(base_url, endpoint)
    
    print(f"\n🧪 Testing: {method} {endpoint}")
    print(f"   Description: {description}")
    print(f"   URL: {url}")
    
    try:
        if method.upper() == 'GET':
            response = requests.get(url, timeout=10)
        elif method.upper() == 'POST':
            headers = {'Content-Type': 'application/json'} if data else {}
            response = requests.post(url, json=data, headers=headers, timeout=10)
        elif method.upper() == 'PUT':
            headers = {'Content-Type': 'application/json'} if data else {}
            response = requests.put(url, json=data, headers=headers, timeout=10)
        elif method.upper() == 'PATCH':
            headers = {'Content-Type': 'application/json'} if data else {}
            response = requests.patch(url, json=data, headers=headers, timeout=10)
        elif method.upper() == 'DELETE':
            response = requests.delete(url, timeout=10)
        else:
            print(f"   ❌ Unsupported method: {method}")
            return False
            
        print(f"   ✅ Status: {response.status_code}")
        print(f"   ⏱️  Response Time: {response.elapsed.total_seconds():.3f}s")
        
        # Try to parse JSON response
        try:
            json_response = response.json()
            print(f"   📄 Response Type: JSON")
            if len(str(json_response)) > 200:
                print(f"   📝 Response Preview: {str(json_response)[:200]}...")
            else:
                print(f"   📝 Response: {json_response}")
        except:
            text_response = response.text
            if len(text_response) > 200:
                print(f"   📝 Response Preview: {text_response[:200]}...")
            else:
                print(f"   📝 Response: {text_response}")
                
        return response.status_code < 400
        
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connection Error: Could not connect to {url}")
        return False
    except requests.exceptions.Timeout:
        print(f"   ❌ Timeout: Request timed out after 10 seconds")
        return False
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False

def main():
    base_url = "http://localhost:8000"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print("=" * 60)
    print("🔍 Compliance System API Inspector")
    print("=" * 60)
    print(f"Base URL: {base_url}")
    print()
    
    success_count = 0
    total_count = 0
    
    # Test basic connectivity
    print("🌐 BASIC CONNECTIVITY")
    tests = [
        ("GET", "/", "Root endpoint"),
        ("GET", "/health", "Health check"),
        ("GET", "/docs", "FastAPI documentation"),
        ("GET", "/redoc", "ReDoc documentation"),
        ("GET", "/openapi.json", "OpenAPI schema"),
    ]
    
    for method, endpoint, desc in tests:
        total_count += 1
        if test_endpoint(base_url, method, endpoint, desc):
            success_count += 1
    
    # Test API endpoints from your script
    print("\n" + "=" * 60)
    print("🏢 PORTFOLIO ENDPOINTS")
    portfolio_tests = [
        ("GET", "/api/portfolio", "Get all positions"),
        ("GET", "/api/portfolio/summary/overview", "Portfolio summary"),
        ("POST", "/api/portfolio/TEST", "Add test position", {
            "quantity": 100,
            "purchase_price": 50.00,
            "sector": "Technology", 
            "country": "US"
        }),
    ]
    
    for test in portfolio_tests:
        total_count += 1
        if test_endpoint(base_url, *test):
            success_count += 1
    
    print("\n" + "=" * 60)
    print("📋 RULES ENDPOINTS")
    rules_tests = [
        ("GET", "/api/rules", "Get all rules"),
        ("GET", "/api/rules/templates/control-types", "Get rule templates"),
    ]
    
    for test in rules_tests:
        total_count += 1
        if test_endpoint(base_url, *test):
            success_count += 1
    
    print("\n" + "=" * 60)  
    print("✅ COMPLIANCE ENDPOINTS")
    compliance_tests = [
        ("GET", "/api/compliance/status", "Compliance status"),
        ("GET", "/api/compliance/breaches", "Get all breaches"),
    ]
    
    for test in compliance_tests:
        total_count += 1
        if test_endpoint(base_url, *test):
            success_count += 1
    
    print("\n" + "=" * 60)
    print("📚 POLICY ENDPOINTS")
    policy_tests = [
        ("GET", "/api/policies", "Get all policies"),
        ("GET", "/api/policies/stats/processing", "Processing stats"),
    ]
    
    for test in policy_tests:
        total_count += 1
        if test_endpoint(base_url, *test):
            success_count += 1
    
    print("\n" + "=" * 60)
    print("📊 ANALYTICS ENDPOINTS")
    analytics_tests = [
        ("GET", "/api/analytics/compliance-summary", "Compliance summary"),
        ("GET", "/api/analytics/performance-metrics", "Performance metrics"),
    ]
    
    for test in analytics_tests:
        total_count += 1
        if test_endpoint(base_url, *test):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    print(f"Success Rate: {(success_count/total_count)*100:.1f}%")
    
    if success_count == 0:
        print("\n❌ NO ENDPOINTS RESPONDING!")
        print("🔧 DEBUGGING SUGGESTIONS:")
        print("   1. Check if your server is actually running")
        print("   2. Verify the correct port (default: 8000)")
        print("   3. Check for any startup errors in server logs")
        print("   4. Make sure your FastAPI routes are properly registered")
        print("   5. Try visiting http://localhost:8000/docs in a browser")
    elif success_count < total_count:
        print(f"\n⚠️  SOME ENDPOINTS FAILING!")
        print("🔧 POSSIBLE ISSUES:")
        print("   1. Database connection problems (check your logs)")
        print("   2. Missing route implementations")
        print("   3. Authentication/authorization issues")
        print("   4. Server configuration problems")
    else:
        print(f"\n🎉 ALL ENDPOINTS WORKING!")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()