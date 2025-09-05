#!/usr/bin/env python3
"""
Database Migration Script for Compliance System
Fixes missing fields and schema mismatches
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, Any

def migrate_database(db_path: str = "./compliance.db"):
    """
    Migrate the database to add missing fields and fix data inconsistencies
    """
    print(f"Starting database migration for {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 1. Add missing columns to portfolios table
        print("1. Updating portfolios table...")
        
        portfolio_columns = [
            ("position_id", "VARCHAR(255)"),
            ("symbol", "VARCHAR(64)"),
            ("name", "VARCHAR(255)"),
            ("weight", "FLOAT"),
            ("market_value", "FLOAT"),
            ("quantity", "FLOAT"),
            ("price", "FLOAT"),
            ("sector", "VARCHAR(255)"),
            ("industry", "VARCHAR(255)"),
            ("country", "VARCHAR(255)"),
            ("currency", "VARCHAR(10)"),
            ("rating", "VARCHAR(50)"),
            ("rating_agency", "VARCHAR(100)"),
            ("instrument_type", "VARCHAR(100)"),
            ("exchange", "VARCHAR(100)"),
            ("maturity_date", "DATETIME"),
            ("acquisition_date", "DATETIME"),
            ("bloomberg_id", "VARCHAR(50)"),
            ("cusip", "VARCHAR(50)"),
            ("isin", "VARCHAR(50)"),
            ("sedol", "VARCHAR(50)"),
            ("rule_metadata", "TEXT"),
            ("last_updated", "DATETIME")
        ]
        
        for col_name, col_type in portfolio_columns:
            try:
                cursor.execute(f"ALTER TABLE portfolios ADD COLUMN {col_name} {col_type}")
                print(f"   Added column: {col_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    print(f"   Column {col_name} already exists")
                else:
                    print(f"   Error adding {col_name}: {e}")
        
        # Set default values for new columns
        cursor.execute("UPDATE portfolios SET weight = 0.0 WHERE weight IS NULL")
        cursor.execute("UPDATE portfolios SET market_value = 0.0 WHERE market_value IS NULL") 
        cursor.execute("UPDATE portfolios SET currency = 'USD' WHERE currency IS NULL")
        cursor.execute("UPDATE portfolios SET last_updated = CURRENT_TIMESTAMP WHERE last_updated IS NULL")
        
        # Update portfolio position_ids if missing
        cursor.execute("""
            UPDATE portfolios 
            SET position_id = COALESCE(position_id, portfolio_id || '_' || RANDOM())
            WHERE position_id IS NULL
        """)
        
        # 2. Add missing columns to compliance_rules table
        print("2. Updating compliance_rules table...")
        
        rule_columns = [
            ("source_policy_id", "VARCHAR(255)"),
            ("created_by", "VARCHAR(255)"),
            ("modified_by", "VARCHAR(255)"),
            ("modified_at", "DATETIME"),
            ("position_metadata", "TEXT")
        ]
        
        for col_name, col_type in rule_columns:
            try:
                cursor.execute(f"ALTER TABLE compliance_rules ADD COLUMN {col_name} {col_type}")
                print(f"   Added column: {col_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    print(f"   Column {col_name} already exists")
                else:
                    print(f"   Error adding {col_name}: {e}")
        
        # Set default values for compliance_rules
        cursor.execute("UPDATE compliance_rules SET created_by = 'system' WHERE created_by IS NULL")
        
        # Update rule expressions to be JSON if they're strings
        cursor.execute("SELECT rule_id, expression FROM compliance_rules")
        rules = cursor.fetchall()
        
        for rule_id, expression in rules:
            try:
                # Try to parse as JSON, if it fails, wrap in a dict
                if isinstance(expression, str):
                    try:
                        json.loads(expression)
                    except json.JSONDecodeError:
                        # Convert string expression to dict
                        new_expression = {"raw_expression": expression, "type": "legacy"}
                        cursor.execute(
                            "UPDATE compliance_rules SET expression = ? WHERE rule_id = ?",
                            (json.dumps(new_expression), rule_id)
                        )
                        print(f"   Updated expression for rule {rule_id}")
            except Exception as e:
                print(f"   Error updating rule {rule_id}: {e}")
        
        # 3. Fix compliance_breaches table
        print("3. Updating compliance_breaches table...")
        
        # Convert ENUM status values to lowercase
        cursor.execute("""
            UPDATE compliance_breaches 
            SET status = LOWER(status)
            WHERE status IN ('OPEN', 'RESOLVED', 'FALSE_POSITIVE')
        """)
        print("   Converted status values to lowercase")
        
        # Fix impact_assessment JSON strings
        cursor.execute("SELECT breach_id, impact_assessment FROM compliance_breaches")
        breaches = cursor.fetchall()
        
        for breach_id, impact_assessment in breaches:
            if impact_assessment and isinstance(impact_assessment, str):
                try:
                    # Try to parse as JSON
                    json.loads(impact_assessment)
                except json.JSONDecodeError:
                    # If it's not valid JSON, create a proper structure
                    new_impact = {
                        "legacy_data": impact_assessment,
                        "severity": "LOW",
                        "materiality_bps": 0,
                        "risk_impact": "Data migration - review required"
                    }
                    cursor.execute(
                        "UPDATE compliance_breaches SET impact_assessment = ? WHERE breach_id = ?",
                        (json.dumps(new_impact), breach_id)
                    )
                    print(f"   Fixed impact_assessment for breach {breach_id}")
        
        # 4. Add missing columns to policy_documents table
        print("4. Updating policy_documents table...")
        
        policy_columns = [
            ("document_type", "VARCHAR(100)"),
            ("jurisdiction", "VARCHAR(100)"),
            ("upload_date", "DATETIME"),
            ("effective_date", "DATETIME"),
            ("expiry_date", "DATETIME"),
            ("version", "INTEGER"),
            ("document_metadata", "TEXT"),
            ("updated_at", "DATETIME")
        ]
        
        for col_name, col_type in policy_columns:
            try:
                cursor.execute(f"ALTER TABLE policy_documents ADD COLUMN {col_name} {col_type}")
                print(f"   Added column: {col_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    print(f"   Column {col_name} already exists")
                else:
                    print(f"   Error adding {col_name}: {e}")
        
        # Set default values for policy_documents
        cursor.execute("UPDATE policy_documents SET upload_date = created_at WHERE upload_date IS NULL")
        cursor.execute("UPDATE policy_documents SET version = 1 WHERE version IS NULL")
        cursor.execute("UPDATE policy_documents SET updated_at = created_at WHERE updated_at IS NULL")
        
        # 5. Add missing columns to position_history table
        print("5. Updating position_history table...")
        
        history_columns = [
            ("history_id", "VARCHAR(255)"),
            ("position_id", "VARCHAR(255)"),
            ("weight", "FLOAT"),
            ("market_value", "FLOAT"),
            ("change_type", "VARCHAR(50)"),
            ("changed_fields", "TEXT"),
            ("previous_values", "TEXT"),
            ("change_reason", "VARCHAR(255)"),
            ("changed_by", "VARCHAR(255)"),
            ("position_metadata", "TEXT")
        ]
        
        for col_name, col_type in history_columns:
            try:
                cursor.execute(f"ALTER TABLE position_history ADD COLUMN {col_name} {col_type}")
                print(f"   Added column: {col_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    print(f"   Column {col_name} already exists")
                else:
                    print(f"   Error adding {col_name}: {e}")
        
        # Set default values for position_history 
        cursor.execute("UPDATE position_history SET weight = 0.0 WHERE weight IS NULL")
        cursor.execute("UPDATE position_history SET market_value = 0.0 WHERE market_value IS NULL")
        
        # Generate history_ids for existing records
        cursor.execute("""
            UPDATE position_history 
            SET history_id = 'hist_' || id || '_' || RANDOM()
            WHERE history_id IS NULL
        """)
        
        # 6. Create indexes for better performance
        print("6. Creating indexes...")
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_portfolios_symbol ON portfolios(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_portfolios_weight ON portfolios(weight)",
            "CREATE INDEX IF NOT EXISTS idx_breaches_status ON compliance_breaches(status)",
            "CREATE INDEX IF NOT EXISTS idx_breaches_timestamp ON compliance_breaches(breach_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_rules_active ON compliance_rules(is_active)",
            "CREATE INDEX IF NOT EXISTS idx_policies_status ON policy_documents(status)"
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
                print(f"   Created index: {index_sql.split('idx_')[1].split(' ')[0]}")
            except Exception as e:
                print(f"   Error creating index: {e}")
        
        # 7. Verify data integrity
        print("7. Verifying data integrity...")
        
        # Check portfolios
        cursor.execute("SELECT COUNT(*) FROM portfolios WHERE symbol IS NOT NULL")
        portfolio_count = cursor.fetchone()[0]
        print(f"   Portfolios with symbols: {portfolio_count}")
        
        # Check rules
        cursor.execute("SELECT COUNT(*) FROM compliance_rules WHERE is_active = 1")
        active_rules = cursor.fetchone()[0]
        print(f"   Active compliance rules: {active_rules}")
        
        # Check breaches
        cursor.execute("SELECT COUNT(*) FROM compliance_breaches")
        breach_count = cursor.fetchone()[0]
        print(f"   Total breaches: {breach_count}")
        
        # Check policies
        cursor.execute("SELECT COUNT(*) FROM policy_documents")
        policy_count = cursor.fetchone()[0]
        print(f"   Policy documents: {policy_count}")
        
        # Commit all changes
        conn.commit()
        print("\n✅ Database migration completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        conn.rollback()
        return False
        
    finally:
        conn.close()

def verify_migration(db_path: str = "./compliance.db"):
    """
    Verify that the migration was successful
    """
    print("\nVerifying migration...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if key columns exist
        tables_to_check = {
            "portfolios": ["symbol", "weight", "market_value", "position_id"],
            "compliance_rules": ["source_policy_id", "created_by", "metadata"],
            "compliance_breaches": ["impact_assessment"],
            "policy_documents": ["document_type", "upload_date", "version"],
            "position_history": ["history_id", "change_type", "weight"]
        }
        
        for table, columns in tables_to_check.items():
            cursor.execute(f"PRAGMA table_info({table})")
            existing_columns = [row[1] for row in cursor.fetchall()]
            
            missing_columns = [col for col in columns if col not in existing_columns]
            
            if missing_columns:
                print(f"❌ Table {table} missing columns: {missing_columns}")
                return False
            else:
                print(f"✅ Table {table} has all required columns")
        
        print("\n✅ Migration verification successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return False
        
    finally:
        conn.close()

if __name__ == "__main__":
    import sys
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else "./compliance.db"
    
    print("Compliance System Database Migration")
    print("=" * 40)
    
    success = migrate_database(db_path)
    
    if success:
        verify_migration(db_path)
    else:
        print("\nMigration failed. Please check the errors above.")
        sys.exit(1)