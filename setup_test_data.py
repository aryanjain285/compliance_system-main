#!/usr/bin/env python3
"""
Comprehensive Test Data Setup for Compliance System
Creates mock rules, portfolios, policies, and breaches for testing
"""
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import json

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.database import (
    db_manager, 
    ComplianceRule, ComplianceBreach, ComplianceCase, Portfolio, 
    PositionHistory, PolicyDocument, PolicyChunk,
    ControlType, RuleSeverity, BreachStatus, DocumentStatus
)
from app.services.vector_store import VectorStoreService
from app.config.settings import get_settings

def create_mock_compliance_rules():
    """Create comprehensive mock compliance rules"""
    print("Creating mock compliance rules...")
    
    rules = [
        {
            "rule_id": "RULE_001",
            "name": "Single Issuer Concentration Limit",
            "description": "No single issuer shall exceed 5% of total portfolio NAV",
            "expression": json.dumps({
                "type": "QUANT_LIMIT",
                "metric": "issuer_weight",
                "operator": "<=",
                "threshold": 0.05,
                "aggregation": "max",
                "scope": "portfolio"
            }),
            "control_type": ControlType.QUANT_LIMIT,
            "severity": RuleSeverity.HIGH,
            "materiality_bps": 100,
            "source_section": "Section 4.2.1 - Concentration Risk",
            "version": 1,
            "is_active": True
        },
        {
            "rule_id": "RULE_002", 
            "name": "Sector Concentration Limit",
            "description": "No single sector shall exceed 25% of total portfolio NAV",
            "expression": json.dumps({
                "type": "QUANT_LIMIT",
                "metric": "sector_weight",
                "operator": "<=",
                "threshold": 0.25,
                "aggregation": "max",
                "scope": "portfolio"
            }),
            "control_type": ControlType.QUANT_LIMIT,
            "severity": RuleSeverity.MEDIUM,
            "materiality_bps": 200,
            "source_section": "Section 4.3.1 - Sector Diversification",
            "version": 1,
            "is_active": True
        },
        {
            "rule_id": "RULE_003",
            "name": "Credit Rating Minimum",
            "description": "All corporate bonds must have minimum BBB rating",
            "expression": json.dumps({
                "type": "LIST_CONSTRAINT",
                "metric": "credit_rating",
                "operator": "in",
                "allowed_values": ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"],
                "scope": "position",
                "asset_class": "corporate_bond"
            }),
            "control_type": ControlType.LIST_CONSTRAINT,
            "severity": RuleSeverity.CRITICAL,
            "materiality_bps": 50,
            "source_section": "Section 3.1.2 - Credit Quality Standards",
            "version": 1,
            "is_active": True
        },
        {
            "rule_id": "RULE_004",
            "name": "Country Exposure Limit",
            "description": "No single country (ex-US) shall exceed 15% of total portfolio NAV",
            "expression": json.dumps({
                "type": "QUANT_LIMIT",
                "metric": "country_weight",
                "operator": "<=",
                "threshold": 0.15,
                "aggregation": "max",
                "scope": "portfolio",
                "exclude": ["US"]
            }),
            "control_type": ControlType.QUANT_LIMIT,
            "severity": RuleSeverity.MEDIUM,
            "materiality_bps": 150,
            "source_section": "Section 4.4.1 - Geographic Diversification",
            "version": 1,
            "is_active": True
        },
        {
            "rule_id": "RULE_005",
            "name": "Cash Reserve Minimum",
            "description": "Portfolio must maintain minimum 2% cash reserves",
            "expression": json.dumps({
                "type": "QUANT_LIMIT",
                "metric": "cash_weight",
                "operator": ">=",
                "threshold": 0.02,
                "scope": "portfolio"
            }),
            "control_type": ControlType.QUANT_LIMIT,
            "severity": RuleSeverity.LOW,
            "materiality_bps": 200,
            "source_section": "Section 5.1.1 - Liquidity Management",
            "version": 1,
            "is_active": True
        }
    ]
    
    session = db_manager.get_session()
    try:
        for rule_data in rules:
            existing_rule = session.query(ComplianceRule).filter_by(rule_id=rule_data["rule_id"]).first()
            if existing_rule:
                print(f"Rule {rule_data['rule_id']} already exists, updating...")
                for key, value in rule_data.items():
                    setattr(existing_rule, key, value)
            else:
                rule = ComplianceRule(**rule_data)
                session.add(rule)
                print(f"Created rule: {rule_data['rule_id']} - {rule_data['name']}")
        
        session.commit()
        print(f"Successfully created/updated {len(rules)} compliance rules")
        
    except Exception as e:
        session.rollback()
        print(f"Error creating rules: {e}")
        raise
    finally:
        session.close()

def create_mock_portfolios():
    """Create mock portfolio data"""
    print("Creating mock portfolios...")
    
    portfolios = [
        {
            "portfolio_id": "PORT_001",
            "symbol": "BGF001",
            "name": "Balanced Growth Fund",
            "owner": "John Smith"
        },
        {
            "portfolio_id": "PORT_002",
            "symbol": "CIF002", 
            "name": "Conservative Income Fund",
            "owner": "Sarah Johnson"
        }
    ]
    
    session = db_manager.get_session()
    try:
        for port_data in portfolios:
            existing_portfolio = session.query(Portfolio).filter_by(portfolio_id=port_data["portfolio_id"]).first()
            if not existing_portfolio:
                portfolio = Portfolio(**port_data)
                session.add(portfolio)
                print(f"Created portfolio: {port_data['portfolio_id']} - {port_data['name']}")
        
        session.commit()
        
    except Exception as e:
        session.rollback()
        print(f"Error creating portfolios: {e}")
        raise
    finally:
        session.close()

def create_mock_positions():
    """Create mock position history to test rules"""
    print("Creating mock positions...")
    
    # Positions that will trigger some rule violations
    positions = [
        # PORT_001 positions - will trigger issuer concentration breach
        {"portfolio_id": "PORT_001", "symbol": "AAPL", "quantity": 1000, "price": 150.0, "market_value": 150000.0, "weight": 0.078},  # High concentration
        {"portfolio_id": "PORT_001", "symbol": "MSFT", "quantity": 500, "price": 300.0, "market_value": 150000.0, "weight": 0.078},
        {"portfolio_id": "PORT_001", "symbol": "GOOGL", "quantity": 200, "price": 120.0, "market_value": 24000.0, "weight": 0.012},
        {"portfolio_id": "PORT_001", "symbol": "TSLA", "quantity": 300, "price": 200.0, "market_value": 60000.0, "weight": 0.031},
        {"portfolio_id": "PORT_001", "symbol": "NVDA", "quantity": 100, "price": 400.0, "market_value": 40000.0, "weight": 0.021},
        
        # PORT_002 positions - more conservative
        {"portfolio_id": "PORT_002", "symbol": "BRK.A", "quantity": 10, "price": 500000.0, "market_value": 5000000.0, "weight": 0.83},
        {"portfolio_id": "PORT_002", "symbol": "JNJ", "quantity": 200, "price": 160.0, "market_value": 32000.0, "weight": 0.053},
        {"portfolio_id": "PORT_002", "symbol": "PG", "quantity": 150, "price": 140.0, "market_value": 21000.0, "weight": 0.035},
        {"portfolio_id": "PORT_002", "symbol": "KO", "quantity": 300, "price": 55.0, "market_value": 16500.0, "weight": 0.027},
        {"portfolio_id": "PORT_002", "symbol": "CASH", "quantity": 50000, "price": 1.0, "market_value": 50000.0, "weight": 0.083},  # 8.3% cash
    ]
    
    session = db_manager.get_session()
    try:
        # Clear existing positions for test portfolios
        session.query(PositionHistory).filter(
            PositionHistory.portfolio_id.in_(["PORT_001", "PORT_002"])
        ).delete()
        
        for pos_data in positions:
            position = PositionHistory(**pos_data)
            session.add(position)
            print(f"Created position: {pos_data['portfolio_id']} - {pos_data['symbol']}")
        
        session.commit()
        print(f"Successfully created {len(positions)} positions")
        
    except Exception as e:
        session.rollback()
        print(f"Error creating positions: {e}")
        raise
    finally:
        session.close()

def create_mock_policy_documents():
    """Create mock policy documents and chunks"""
    print("Creating mock policy documents...")
    
    # Mock policy content
    policy_sections = [
        {
            "title": "Investment Policy Statement - Risk Management",
            "content": """
            SECTION 4.2.1 - CONCENTRATION RISK MANAGEMENT
            
            To mitigate concentration risk, the Fund shall observe the following limits:
            
            4.2.1.1 Single Issuer Exposure: No single issuer shall represent more than five percent (5%) 
            of the Fund's total net asset value. This limit applies to all securities of the same issuer, 
            including but not limited to common stock, preferred stock, corporate bonds, and convertible securities.
            
            4.2.1.2 Materiality Threshold: Breaches of issuer concentration limits shall be considered 
            material if they exceed 100 basis points (1.00%) above the stated limit.
            
            4.2.1.3 Monitoring and Reporting: Portfolio managers must monitor issuer concentrations daily 
            and report any breaches to the Risk Management Committee within 24 hours.
            """
        },
        {
            "title": "Investment Policy Statement - Sector Allocation",
            "content": """
            SECTION 4.3.1 - SECTOR DIVERSIFICATION REQUIREMENTS
            
            4.3.1.1 Sector Concentration Limits: No single Global Industry Classification Standard (GICS) 
            sector shall exceed twenty-five percent (25%) of the Fund's total net asset value.
            
            4.3.1.2 Sector Allocation Guidelines: The Fund should maintain diversified exposure across 
            at least six (6) different sectors to ensure adequate risk distribution.
            
            4.3.1.3 Technology Sector Special Provisions: Given the volatility of technology securities, 
            exposure to the Technology sector should not exceed 20% under normal market conditions.
            """
        },
        {
            "title": "Investment Policy Statement - Credit Quality Standards", 
            "content": """
            SECTION 3.1.2 - CREDIT QUALITY AND RATING REQUIREMENTS
            
            3.1.2.1 Minimum Credit Rating: All corporate bond investments must carry a minimum credit 
            rating of BBB- (or equivalent) from at least one major rating agency (S&P, Moody's, or Fitch).
            
            3.1.2.2 Rating Methodology: For securities with split ratings, the second-highest rating 
            shall be used for compliance purposes.
            
            3.1.2.3 Fallen Angels: Securities that are downgraded below the minimum rating after purchase 
            must be sold within 90 days unless specifically approved by the Investment Committee.
            
            3.1.2.4 Unrated Securities: Unrated securities may comprise no more than 5% of the portfolio 
            and require special approval from the Chief Investment Officer.
            """
        },
        {
            "title": "Investment Policy Statement - Geographic Diversification",
            "content": """
            SECTION 4.4.1 - COUNTRY AND REGIONAL EXPOSURE LIMITS
            
            4.4.1.1 Non-US Country Limits: Exposure to any single country (excluding the United States) 
            shall not exceed fifteen percent (15%) of the Fund's total net asset value.
            
            4.4.1.2 Emerging Markets Allocation: Total exposure to emerging markets shall not exceed 
            twenty percent (20%) of the Fund's net asset value.
            
            4.4.1.3 Currency Hedging: Foreign currency exposure exceeding 10% of NAV should be 
            appropriately hedged unless specifically approved by the Risk Committee.
            
            4.4.1.4 Regional Diversification: The Fund should maintain exposure across at least 
            three major geographic regions to ensure global diversification.
            """
        },
        {
            "title": "Investment Policy Statement - Liquidity Management",
            "content": """
            SECTION 5.1.1 - CASH RESERVES AND LIQUIDITY REQUIREMENTS
            
            5.1.1.1 Minimum Cash Reserve: The Fund must maintain a minimum cash reserve of two percent (2%) 
            of net asset value to meet redemption requests and operational needs.
            
            5.1.1.2 Liquidity Classification: All portfolio securities must be classified into liquidity 
            buckets (Highly Liquid, Moderately Liquid, Less Liquid, Illiquid) based on market trading volumes.
            
            5.1.1.3 Illiquid Securities Limit: Holdings of illiquid securities shall not exceed fifteen 
            percent (15%) of the Fund's net asset value.
            
            5.1.1.4 Stress Testing: Liquidity positions must be stress tested monthly to ensure the Fund 
            can meet potential redemption scenarios under adverse market conditions.
            """
        }
    ]
    
    session = db_manager.get_session()
    try:
        for i, section in enumerate(policy_sections, 1):
            # Create policy document
            policy_id = f"POL_{i:03d}"
            content_hash = str(hash(section["content"]))[:32]
            
            existing_doc = session.query(PolicyDocument).filter_by(policy_id=policy_id).first()
            if not existing_doc:
                policy_doc = PolicyDocument(
                    policy_id=policy_id,
                    title=section["title"],
                    filename=f"investment_policy_section_{i}.pdf",
                    content_hash=content_hash,
                    status=DocumentStatus.INDEXED,
                    uploaded_by="System Admin"
                )
                session.add(policy_doc)
                
                # Create policy chunk
                chunk_id = f"CHUNK_{i:03d}"
                policy_chunk = PolicyChunk(
                    chunk_id=chunk_id,
                    policy_id=policy_id,
                    chunk_index=0,
                    page_number=i,
                    section_title=section["title"],
                    content=section["content"],
                    word_count=len(section["content"].split()),
                    char_count=len(section["content"]),
                    chunk_metadata=json.dumps({
                        "section": f"4.{i}",
                        "importance": "high",
                        "compliance_rules": [f"RULE_{i:03d}"]
                    })
                )
                session.add(policy_chunk)
                
                print(f"Created policy document: {policy_id} - {section['title']}")
        
        session.commit()
        print(f"Successfully created {len(policy_sections)} policy documents")
        
    except Exception as e:
        session.rollback()
        print(f"Error creating policy documents: {e}")
        raise
    finally:
        session.close()

def create_mock_breaches():
    """Create some mock compliance breaches for testing"""
    print("Creating mock compliance breaches...")
    
    breaches = [
        {
            "breach_id": str(uuid.uuid4()),
            "rule_id": "RULE_001",
            "status": BreachStatus.OPEN,
            "observed_value": 0.078,  # 7.8% vs 5% limit
            "threshold_value": 0.05,
            "breach_magnitude": 0.028,
            "breach_timestamp": datetime.now() - timedelta(hours=2),
            "portfolio_snapshot": json.dumps({
                "portfolio_id": "PORT_001",
                "total_nav": 1000000.0,
                "issuer": "AAPL",
                "issuer_value": 78000.0,
                "issuer_weight": 0.078
            }),
            "impact_assessment": json.dumps({
                "severity": "HIGH",
                "materiality_bps": 280,
                "risk_impact": "Concentration risk elevated"
            })
        },
        {
            "breach_id": str(uuid.uuid4()),
            "rule_id": "RULE_005",
            "status": BreachStatus.OPEN,
            "observed_value": 0.015,  # 1.5% vs 2% minimum
            "threshold_value": 0.02,
            "breach_magnitude": -0.005,
            "breach_timestamp": datetime.now() - timedelta(minutes=30),
            "portfolio_snapshot": json.dumps({
                "portfolio_id": "PORT_001", 
                "total_nav": 1000000.0,
                "cash_value": 15000.0,
                "cash_weight": 0.015
            }),
            "impact_assessment": json.dumps({
                "severity": "LOW",
                "materiality_bps": 50,
                "risk_impact": "Liquidity risk increased"
            })
        }
    ]
    
    session = db_manager.get_session()
    try:
        for breach_data in breaches:
            breach = ComplianceBreach(**breach_data)
            session.add(breach)
            print(f"Created breach: {breach_data['rule_id']} - {breach_data['breach_magnitude']:.3f}")
        
        session.commit()
        print(f"Successfully created {len(breaches)} compliance breaches")
        
    except Exception as e:
        session.rollback()
        print(f"Error creating breaches: {e}")
        raise
    finally:
        session.close()

def setup_vector_store():
    """Add policy documents to ChromaDB vector store"""
    print("Setting up vector store with policy documents...")
    
    try:
        settings = get_settings()
        vector_service = VectorStoreService()
        
        session = db_manager.get_session()
        
        # Get all policy chunks
        chunks = session.query(PolicyChunk).join(PolicyDocument).all()
        
        # Group chunks by policy_id
        policy_chunks = {}
        for chunk in chunks:
            policy_id = chunk.policy_id
            if policy_id not in policy_chunks:
                policy_chunks[policy_id] = []
            
            # Parse JSON metadata if it's a string
            try:
                metadata = json.loads(chunk.chunk_metadata) if isinstance(chunk.chunk_metadata, str) else chunk.chunk_metadata
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            
            chunk_data = {
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "word_count": chunk.word_count,
                "char_count": chunk.char_count,
                "metadata": metadata
            }
            policy_chunks[policy_id].append(chunk_data)
        
        # Add documents to vector store by policy
        total_added = 0
        for policy_id, policy_chunk_data in policy_chunks.items():
            if vector_service.add_policy_chunks(policy_id, policy_chunk_data):
                total_added += len(policy_chunk_data)
                print(f"Added {len(policy_chunk_data)} chunks for policy {policy_id}")
        
        print(f"Total: Added {total_added} policy chunks to vector store")
        
        session.close()
        
    except Exception as e:
        print(f"Error setting up vector store: {e}")
        raise

def main():
    """Set up all mock data for compliance system testing"""
    print("=" * 60)
    print("COMPLIANCE SYSTEM TEST DATA SETUP")
    print("=" * 60)
    
    try:
        # Ensure database tables exist
        print("Ensuring database tables exist...")
        db_manager.create_tables()
        
        # Create all mock data
        create_mock_compliance_rules()
        create_mock_portfolios()
        create_mock_positions() 
        create_mock_policy_documents()
        create_mock_breaches()
        
        # Setup vector store
        setup_vector_store()
        
        print("=" * 60)
        print("✅ COMPLIANCE SYSTEM TEST DATA SETUP COMPLETE")
        print("=" * 60)
        print("\nTest Data Summary:")
        print("- 5 compliance rules (issuer, sector, credit, country, cash)")
        print("- 2 test portfolios with positions")
        print("- 5 policy document sections")
        print("- 2 sample compliance breaches")
        print("- Policy documents indexed in ChromaDB")
        print("\nYou can now test the compliance system APIs!")
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        raise

if __name__ == "__main__":
    main()