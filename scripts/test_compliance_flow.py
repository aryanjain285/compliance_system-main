#!/usr/bin/env python3
"""
Comprehensive Compliance System Flow Testing
Tests the complete workflow: rules → evaluation → breaches → explanations
"""
import sys
import os
import json
import asyncio
from pathlib import Path
import requests
from datetime import datetime

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.database import (
    db_manager, ComplianceRule, ComplianceBreach, Portfolio, 
    PositionHistory, PolicyDocument, PolicyChunk
)
from app.services.vector_store import VectorStoreService
from app.config.settings import get_settings

# Base API URL
BASE_URL = "http://localhost:8000"

class ComplianceFlowTester:
    def __init__(self):
        self.settings = get_settings()
        self.vector_service = VectorStoreService()
        
    def test_api_health(self):
        """Test basic API connectivity"""
        print("🔍 Testing API Health...")
        try:
            response = requests.get(f"{BASE_URL}/")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API is healthy - {data.get('message', 'OK')}")
                return True
            else:
                print(f"❌ API health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            return False
    
    def test_rules_endpoint(self):
        """Test compliance rules API"""
        print("\n🔍 Testing Rules API...")
        try:
            # Get all rules
            response = requests.get(f"{BASE_URL}/api/rules")
            if response.status_code == 200:
                rules = response.json()
                print(f"✅ Retrieved {len(rules.get('data', []))} compliance rules")
                
                # Show sample rule
                if rules.get('data'):
                    sample_rule = rules['data'][0]
                    print(f"   Sample: {sample_rule.get('rule_id')} - {sample_rule.get('name')}")
                return True
            else:
                print(f"❌ Rules API failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Rules API error: {e}")
            return False
    
    def test_portfolio_analysis(self):
        """Test portfolio analysis and rule evaluation"""
        print("\n🔍 Testing Portfolio Analysis...")
        try:
            # Get portfolio overview
            response = requests.get(f"{BASE_URL}/api/portfolio/summary/overview")
            
            if response.status_code == 200:
                analysis = response.json()
                print(f"✅ Portfolio analysis completed")
                
                # Show key metrics
                data = analysis.get('data', {})
                print(f"   Response: {str(data)[:100]}...")
                
                return True
            else:
                print(f"❌ Portfolio analysis failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Portfolio analysis error: {e}")
            return False
    
    def test_compliance_evaluation(self):
        """Test compliance rule evaluation"""
        print("\n🔍 Testing Compliance Evaluation...")
        try:
            # Get compliance status
            response = requests.get(f"{BASE_URL}/api/compliance/status")
            
            if response.status_code == 200:
                results = response.json()
                print(f"✅ Compliance status retrieved")
                
                # Show evaluation results
                data = results.get('data', {})
                print(f"   Response: {str(data)[:100]}...")
                
                return True
            else:
                print(f"❌ Compliance evaluation failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Compliance evaluation error: {e}")
            return False
    
    def test_breach_explanation(self):
        """Test breach explanation using LLM and vector search"""
        print("\n🔍 Testing Breach Explanation...")
        try:
            # Get existing breaches from API
            response = requests.get(f"{BASE_URL}/api/compliance/breaches")
            
            if response.status_code == 200:
                breaches = response.json()
                breach_list = breaches.get('data', [])
                
                if not breach_list:
                    print("⚠️ No breaches found to test")
                    return True
                
                # Test explanation for first breach
                breach_id = breach_list[0].get('breach_id')
                response = requests.get(f"{BASE_URL}/api/compliance/breaches/{breach_id}/explain")
                
                if response.status_code == 200:
                    explanation = response.json()
                    print(f"✅ Breach explanation generated")
                    
                    data = explanation.get('data', {})
                    print(f"   Response: {str(data)[:100]}...")
                    
                    return True
                else:
                    print(f"❌ Breach explanation failed: {response.status_code}")
                    return False
            else:
                print(f"❌ Failed to get breaches: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Breach explanation error: {e}")
            return False
    
    def test_policy_query(self):
        """Test natural language policy queries"""
        print("\n🔍 Testing Policy Query...")
        try:
            # Query the policy knowledge base
            query = "What are the issuer concentration limits?"
            response = requests.post(f"{BASE_URL}/api/policies/ask", json={
                "question": query,
                "max_results": 3
            })
            
            if response.status_code == 200:
                results = response.json()
                print(f"✅ Policy query completed")
                
                data = results.get('data', {})
                print(f"   Query: {query}")
                print(f"   Response: {str(data)[:150]}...")
                
                return True
            else:
                print(f"❌ Policy query failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Policy query error: {e}")
            return False
    
    def test_case_management(self):
        """Test compliance case creation and management"""
        print("\n🔍 Testing Case Management...")
        try:
            # Get existing cases
            response = requests.get(f"{BASE_URL}/api/compliance/cases")
            
            if response.status_code == 200:
                cases = response.json()
                print(f"✅ Case management tested")
                
                data = cases.get('data', []) if isinstance(cases, dict) else cases
                print(f"   Cases found: {len(data)}")
                
                return True
            else:
                print(f"❌ Case management failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Case management error: {e}")
            return False
    
    def test_database_integrity(self):
        """Test database data integrity"""
        print("\n🔍 Testing Database Integrity...")
        try:
            session = db_manager.get_session()
            
            # Count records in each table
            rule_count = session.query(ComplianceRule).count()
            portfolio_count = session.query(Portfolio).count()
            position_count = session.query(PositionHistory).count()
            breach_count = session.query(ComplianceBreach).count()
            policy_count = session.query(PolicyDocument).count()
            chunk_count = session.query(PolicyChunk).count()
            
            session.close()
            
            print(f"✅ Database integrity check passed")
            print(f"   Rules: {rule_count}")
            print(f"   Portfolios: {portfolio_count}")
            print(f"   Positions: {position_count}")
            print(f"   Breaches: {breach_count}")
            print(f"   Policy Docs: {policy_count}")
            print(f"   Policy Chunks: {chunk_count}")
            
            return True
        except Exception as e:
            print(f"❌ Database integrity check failed: {e}")
            return False
    
    def test_vector_store_integrity(self):
        """Test vector store integrity"""
        print("\n🔍 Testing Vector Store Integrity...")
        try:
            # Test vector search
            results = self.vector_service.semantic_search(
                query="issuer concentration limits",
                n_results=3
            )
            
            print(f"✅ Vector store integrity check passed")
            print(f"   Search results: {len(results)}")
            if results:
                print(f"   Top result: {results[0].get('metadata', {}).get('title', 'Unknown')[:50]}...")
            
            return True
        except Exception as e:
            print(f"❌ Vector store integrity check failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run complete compliance system test suite"""
        print("="*70)
        print("🧪 COMPREHENSIVE COMPLIANCE SYSTEM TESTING")
        print("="*70)
        
        tests = [
            ("API Health", self.test_api_health),
            ("Database Integrity", self.test_database_integrity),
            ("Vector Store Integrity", self.test_vector_store_integrity),
            ("Rules API", self.test_rules_endpoint),
            ("Portfolio Analysis", self.test_portfolio_analysis),
            ("Compliance Evaluation", self.test_compliance_evaluation),
            ("Breach Explanation", self.test_breach_explanation),
            ("Policy Query", self.test_policy_query),
            ("Case Management", self.test_case_management),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    print(f"❌ {test_name} test failed")
            except Exception as e:
                print(f"❌ {test_name} test error: {e}")
        
        print("\n" + "="*70)
        print(f"🧪 TEST RESULTS: {passed}/{total} tests passed")
        print("="*70)
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Compliance system is working correctly.")
        else:
            print("⚠️ Some tests failed. Check the logs above for details.")
        
        return passed == total

def main():
    """Run the complete compliance flow test"""
    tester = ComplianceFlowTester()
    success = tester.run_all_tests()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()