"""
Compliance Rule Engine - Production Ready Implementation
Handles rule evaluation, breach detection, and compliance monitoring
"""
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from app.models.database import (
    ComplianceRule, ComplianceBreach, ComplianceCase, Portfolio, RuleEvaluation,
    BreachStatus, CaseStatus, CaseType, RuleSeverity, ControlType
)
from app.utils.logger import get_logger, log_execution_time
from app.utils.exceptions import (
    RuleEvaluationException, DatabaseException, ValidationException,
    validate_rule_expression
)
from app.config.settings import get_settings

settings = get_settings()
logger = get_logger(__name__)


class ComplianceEngine:
    """Production-ready compliance engine for rule evaluation and monitoring"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.portfolio_cache = {}
        self.rule_cache = {}
        self.evaluation_context = {
            "evaluation_id": str(uuid.uuid4()),
            "start_time": datetime.now(),
            "settings": {
                "default_materiality_bps": settings.default_materiality_bps,
                "auto_resolve_breaches": settings.auto_resolve_breaches
            }
        }
    
    def _load_portfolio_cache(self) -> Dict[str, Any]:
        """Load portfolio data into memory for fast access"""
        try:
            # Query PositionHistory for current positions
            from app.models.database import PositionHistory
            positions = self.db.query(PositionHistory).all()
            portfolio_cache = {}
            total_market_value = 0
            
            for pos in positions:
                if pos.market_value and pos.market_value > 0:
                    total_market_value += pos.market_value
                    portfolio_cache[pos.symbol] = {
                        "position_id": getattr(pos, 'position_id', pos.history_id),
                        "portfolio_id": pos.portfolio_id,
                        "symbol": pos.symbol,
                        "name": f"{pos.symbol} Position",
                        "weight": float(pos.weight or 0),
                        "market_value": float(pos.market_value or 0),
                        "quantity": float(pos.quantity or 0) if pos.quantity else 0,
                        "price": float(pos.price or 0) if pos.price else 0,
                        "sector": getattr(pos, 'sector', None) or "Unknown",
                        "industry": getattr(pos, 'industry', None) or "Unknown",
                        "country": getattr(pos, 'country', None) or "Unknown",
                        "currency": "USD",
                        "rating": getattr(pos, 'rating', None) or "NR",
                        "rating_agency": getattr(pos, 'rating_agency', None),
                        "instrument_type": pos.instrument_type or "Unknown",
                        "exchange": pos.exchange,
                        "maturity_date": pos.maturity_date.isoformat() if pos.maturity_date else None,
                        "acquisition_date": pos.acquisition_date.isoformat() if pos.acquisition_date else None,
                        "last_updated": pos.last_updated.isoformat() if pos.last_updated else None,
                        "metadata": pos.metadata or {}
                    }
            
            # Recalculate weights based on current market values
            if total_market_value > 0:
                for symbol, position in portfolio_cache.items():
                    position["weight"] = position["market_value"] / total_market_value
            
            logger.info(f"Loaded {len(portfolio_cache)} portfolio positions with total value ${total_market_value:,.2f}")
            return portfolio_cache
            
        except Exception as e:
            logger.error(f"Error loading portfolio cache: {e}")
            raise DatabaseException(f"Failed to load portfolio data: {str(e)}")
    
    def _load_rule_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load active compliance rules into memory"""
        try:
            active_rules = self.db.query(ComplianceRule).filter(
                ComplianceRule.is_active == True,
                or_(
                    ComplianceRule.expiry_date.is_(None),
                    ComplianceRule.expiry_date > datetime.now()
                )
            ).all()
            
            rule_cache = {}
            for rule in active_rules:
                try:
                    # Validate rule expression
                    validate_rule_expression(rule.expression, rule.control_type.value)
                    
                    rule_cache[rule.rule_id] = {
                        "rule": rule,
                        "expression": rule.expression,
                        "control_type": rule.control_type.value,
                        "severity": rule.severity.value,
                        "description": rule.description,
                        "name": rule.name,
                        "materiality_bps": rule.materiality_bps,
                        "source_section": rule.source_section,
                        "effective_date": rule.effective_date,
                        "version": rule.version
                    }
                except ValidationException as e:
                    logger.warning(f"Invalid rule {rule.rule_id}: {e.message}")
                    continue
            
            logger.info(f"Loaded {len(rule_cache)} valid active compliance rules")
            return rule_cache
            
        except Exception as e:
            logger.error(f"Error loading rule cache: {e}")
            raise DatabaseException(f"Failed to load compliance rules: {str(e)}")
    
    @log_execution_time("evaluate_all_rules")
    def evaluate_all_rules(self) -> Dict[str, Any]:
        """Evaluate all active compliance rules with comprehensive reporting"""
        start_time = datetime.now()
        
        results = {
            "evaluation_id": self.evaluation_context["evaluation_id"],
            "total_rules_checked": 0,
            "compliant_rules": 0,
            "breached_rules": 0,
            "case_created_rules": 0,
            "error_rules": 0,
            "skipped_rules": 0,
            "new_breaches": [],
            "new_cases": [],
            "errors": [],
            "evaluation_timestamp": start_time.isoformat(),
            "portfolio_summary": {},
            "results": [],
            "performance": {},
            "system_health": {}
        }
        
        try:
            # Load current data
            self.portfolio_cache = self._load_portfolio_cache()
            self.rule_cache = self._load_rule_cache()
            
            results["total_rules_checked"] = len(self.rule_cache)
            results["portfolio_summary"] = self._get_portfolio_summary()
            
            if not self.rule_cache:
                results["warning"] = "No active rules found for evaluation"
                return results
            
            if not self.portfolio_cache:
                results["warning"] = "No portfolio positions found for evaluation"
                # Still evaluate rules that don't require portfolio data
            
            # Evaluate rules in parallel for better performance
            rule_evaluation_futures = []
            with ThreadPoolExecutor(max_workers=min(len(self.rule_cache), 10)) as executor:
                for rule_id, rule_data in self.rule_cache.items():
                    future = executor.submit(self._evaluate_single_rule, rule_data["rule"])
                    rule_evaluation_futures.append((rule_id, future))
                
                # Collect results as they complete
                for rule_id, future in rule_evaluation_futures:
                    try:
                        rule_result = future.result(timeout=30)  # 30 second timeout per rule
                        results["results"].append(rule_result)
                        
                        # Update counters based on result status
                        status = rule_result.get("status", "error")
                        if status == "compliant":
                            results["compliant_rules"] += 1
                        elif status == "breach":
                            results["breached_rules"] += 1
                            self._handle_breach_result(rule_result, results)
                        elif status == "case_created":
                            results["case_created_rules"] += 1
                            self._handle_case_result(rule_result, results)
                        elif status == "skipped":
                            results["skipped_rules"] += 1
                        else:
                            results["error_rules"] += 1
                            results["errors"].append({
                                "rule_id": rule_id,
                                "error": rule_result.get("message", "Unknown error")
                            })
                    
                    except Exception as e:
                        logger.error(f"Critical error evaluating rule {rule_id}: {e}")
                        results["error_rules"] += 1
                        results["errors"].append({
                            "rule_id": rule_id,
                            "error": str(e)
                        })
            
            # Calculate overall metrics
            total_time = (datetime.now() - start_time).total_seconds()
            total_evaluated = results["compliant_rules"] + results["breached_rules"]
            
            results["performance"] = {
                "total_evaluation_time_seconds": round(total_time, 3),
                "average_rule_time_ms": round(total_time * 1000 / max(results["total_rules_checked"], 1), 2),
                "positions_analyzed": len(self.portfolio_cache),
                "rules_per_second": round(results["total_rules_checked"] / max(total_time, 0.001), 2)
            }
            
            # Calculate compliance rate
            if total_evaluated > 0:
                results["compliance_rate"] = round((results["compliant_rules"] / total_evaluated) * 100, 2)
            else:
                results["compliance_rate"] = 100.0
            
            # System health indicators
            results["system_health"] = {
                "evaluation_success_rate": round(
                    ((results["total_rules_checked"] - results["error_rules"]) / 
                     max(results["total_rules_checked"], 1)) * 100, 2
                ),
                "critical_breaches": len([b for b in results["new_breaches"] if b.get("severity") == "critical"]),
                "high_priority_cases": len([c for c in results["new_cases"] if c.get("priority") == "high"]),
                "portfolio_data_freshness": self._assess_portfolio_freshness()
            }
            
            logger.info(
                f"Compliance evaluation completed: {results['compliance_rate']}% compliance rate, "
                f"{results['breached_rules']} breaches, {results['case_created_rules']} cases created"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Critical error in evaluate_all_rules: {e}")
            results["critical_error"] = str(e)
            return results
    
    def _evaluate_single_rule(self, rule: ComplianceRule) -> Dict[str, Any]:
        """Evaluate a single compliance rule with comprehensive error handling"""
        evaluation_start = datetime.now()
        
        base_result = {
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "rule_description": rule.description,
            "control_type": rule.control_type.value,
            "severity": rule.severity.value,
            "timestamp": evaluation_start.isoformat(),
            "version": rule.version
        }
        
        try:
            expression = rule.expression
            control_type = rule.control_type.value
            
            # Route to appropriate evaluator
            if control_type == "quant_limit":
                result = self._evaluate_quantitative_limit(rule, expression)
            elif control_type == "list_constraint":
                result = self._evaluate_list_constraint(rule, expression)
            elif control_type == "temporal_window":
                result = self._evaluate_temporal_window(rule, expression)
            elif control_type == "process_control":
                result = self._evaluate_process_control(rule, expression)
            elif control_type == "reporting_disclosure":
                result = self._evaluate_reporting_disclosure(rule, expression)
            else:
                result = {"status": "error", "message": f"Unknown control type: {control_type}"}
            
            # Add execution time
            execution_time = (datetime.now() - evaluation_start).total_seconds() * 1000
            result["execution_time_ms"] = round(execution_time, 2)
            
            # Record evaluation in database
            self._record_evaluation(rule.rule_id, {**base_result, **result}, execution_time)
            
            return {**base_result, **result}
        
        except Exception as e:
            execution_time = (datetime.now() - evaluation_start).total_seconds() * 1000
            error_result = {
                "status": "error",
                "message": str(e),
                "execution_time_ms": round(execution_time, 2)
            }
            
            # Record failed evaluation
            self._record_evaluation(rule.rule_id, {**base_result, **error_result}, execution_time)
            
            logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
            return {**base_result, **error_result}
    
    def _evaluate_quantitative_limit(self, rule: ComplianceRule, expression: Dict) -> Dict[str, Any]:
        """Comprehensive quantitative limit evaluation"""
        try:
            metric = expression.get("metric")
            threshold = float(expression.get("threshold", 0))
            operator = expression.get("operator", "<=")
            scope = expression.get("scope", "portfolio")
            group_by = expression.get("group_by")
            filter_value = expression.get("filter")
            
            if not self.portfolio_cache:
                return {"status": "skipped", "message": "No portfolio positions available"}
            
            # Route to specific metric evaluators
            if metric == "issuer_weight":
                return self._evaluate_issuer_concentration(threshold, operator, filter_value, rule)
            elif metric == "sector_weight":
                return self._evaluate_sector_concentration(threshold, operator, filter_value, rule)
            elif metric == "country_weight":
                return self._evaluate_country_concentration(threshold, operator, filter_value, rule)
            elif metric == "industry_weight":
                return self._evaluate_industry_concentration(threshold, operator, filter_value, rule)
            elif metric == "rating_weight":
                return self._evaluate_rating_concentration(threshold, operator, filter_value, rule)
            elif metric == "total_exposure":
                return self._evaluate_total_exposure(threshold, operator, rule)
            elif metric == "leverage_ratio":
                return self._evaluate_leverage_ratio(threshold, operator, rule)
            else:
                return {"status": "error", "message": f"Unknown quantitative metric: {metric}"}
                
        except Exception as e:
            return {"status": "error", "message": f"Quantitative evaluation error: {str(e)}"}
    
    def _evaluate_issuer_concentration(self, threshold: float, operator: str, 
                                     filter_issuer: str = None, rule: ComplianceRule = None) -> Dict[str, Any]:
        """Evaluate issuer concentration limits with detailed analysis"""
        try:
            issuer_weights = {}
            issuer_details = {}
            
            for symbol, position in self.portfolio_cache.items():
                weight = position["weight"]
                issuer_weights[symbol] = weight
                issuer_details[symbol] = {
                    "name": position.get("name", symbol),
                    "market_value": position["market_value"],
                    "sector": position["sector"],
                    "country": position["country"],
                    "rating": position["rating"]
                }
            
            if filter_issuer:
                # Check specific issuer
                observed_value = issuer_weights.get(filter_issuer, 0)
                violation = self._check_threshold(observed_value, threshold, operator)
                violating_issuers = [filter_issuer] if violation else []
            else:
                # Check all issuers against threshold
                violating_issuers = []
                max_weight = 0
                max_issuer = None
                
                for symbol, weight in issuer_weights.items():
                    if self._check_threshold(weight, threshold, operator):
                        violating_issuers.append(symbol)
                    if weight > max_weight:
                        max_weight = weight
                        max_issuer = symbol
                
                observed_value = max_weight
                violation = len(violating_issuers) > 0
            
            # Calculate materiality
            materiality_exceeded = self._check_materiality(
                observed_value, threshold, rule.materiality_bps if rule else 0
            )
            
            result = {
                "status": "breach" if violation and materiality_exceeded else "compliant",
                "observed_value": round(observed_value, 6),
                "threshold": threshold,
                "metric": "issuer_weight",
                "operator": operator,
                "violation_count": len(violating_issuers),
                "materiality_exceeded": materiality_exceeded,
                "details": {
                    "violating_issuers": [
                        {
                            "symbol": symbol,
                            "weight": round(issuer_weights[symbol], 6),
                            "details": issuer_details[symbol]
                        }
                        for symbol in violating_issuers
                    ],
                    "top_5_concentrations": [
                        {
                            "symbol": symbol,
                            "weight": round(weight, 6),
                            "details": issuer_details[symbol]
                        }
                        for symbol, weight in sorted(issuer_weights.items(), key=lambda x: x[1], reverse=True)[:5]
                    ],
                    "portfolio_count": len(issuer_weights),
                    "concentration_hhi": self._calculate_hhi(list(issuer_weights.values()))
                }
            }
            
            # Add filter context if applicable
            if filter_issuer:
                result["filter_issuer"] = filter_issuer
            
            return result
                
        except Exception as e:
            return {"status": "error", "message": f"Issuer concentration evaluation error: {str(e)}"}
    
    def _evaluate_sector_concentration(self, threshold: float, operator: str, 
                                     filter_sector: str = None, rule: ComplianceRule = None) -> Dict[str, Any]:
        """Evaluate sector concentration limits"""
        try:
            sector_weights = defaultdict(float)
            sector_positions = defaultdict(list)
            
            for symbol, position in self.portfolio_cache.items():
                sector = position["sector"]
                weight = position["weight"]
                sector_weights[sector] += weight
                sector_positions[sector].append({
                    "symbol": symbol,
                    "name": position.get("name", symbol),
                    "weight": weight,
                    "market_value": position["market_value"]
                })
            
            sector_weights = dict(sector_weights)
            
            if filter_sector:
                observed_value = sector_weights.get(filter_sector, 0)
                violation = self._check_threshold(observed_value, threshold, operator)
                violating_sectors = [filter_sector] if violation else []
            else:
                violating_sectors = []
                max_weight = 0
                
                for sector, weight in sector_weights.items():
                    if self._check_threshold(weight, threshold, operator):
                        violating_sectors.append(sector)
                    max_weight = max(max_weight, weight)
                
                observed_value = max_weight
                violation = len(violating_sectors) > 0
            
            materiality_exceeded = self._check_materiality(
                observed_value, threshold, rule.materiality_bps if rule else 0
            )
            
            return {
                "status": "breach" if violation and materiality_exceeded else "compliant",
                "observed_value": round(observed_value, 6),
                "threshold": threshold,
                "metric": "sector_weight",
                "operator": operator,
                "violation_count": len(violating_sectors),
                "materiality_exceeded": materiality_exceeded,
                "details": {
                    "violating_sectors": [
                        {
                            "sector": sector,
                            "weight": round(sector_weights[sector], 6),
                            "position_count": len(sector_positions[sector]),
                            "top_positions": sorted(
                                sector_positions[sector], 
                                key=lambda x: x["weight"], 
                                reverse=True
                            )[:3]
                        }
                        for sector in violating_sectors
                    ],
                    "sector_breakdown": {k: round(v, 6) for k, v in sector_weights.items()},
                    "sector_hhi": self._calculate_hhi(list(sector_weights.values()))
                }
            }
                
        except Exception as e:
            return {"status": "error", "message": f"Sector concentration evaluation error: {str(e)}"}
    
    def _evaluate_country_concentration(self, threshold: float, operator: str, 
                                      filter_country: str = None, rule: ComplianceRule = None) -> Dict[str, Any]:
        """Evaluate country concentration limits"""
        try:
            country_weights = defaultdict(float)
            country_positions = defaultdict(list)
            
            for symbol, position in self.portfolio_cache.items():
                country = position["country"]
                weight = position["weight"]
                country_weights[country] += weight
                country_positions[country].append({
                    "symbol": symbol,
                    "name": position.get("name", symbol),
                    "weight": weight,
                    "sector": position["sector"]
                })
            
            country_weights = dict(country_weights)
            
            if filter_country:
                observed_value = country_weights.get(filter_country, 0)
                violation = self._check_threshold(observed_value, threshold, operator)
            else:
                observed_value = max(country_weights.values()) if country_weights else 0
                violation = self._check_threshold(observed_value, threshold, operator)
            
            materiality_exceeded = self._check_materiality(
                observed_value, threshold, rule.materiality_bps if rule else 0
            )
            
            return {
                "status": "breach" if violation and materiality_exceeded else "compliant",
                "observed_value": round(observed_value, 6),
                "threshold": threshold,
                "metric": "country_weight",
                "operator": operator,
                "materiality_exceeded": materiality_exceeded,
                "details": {
                    "country_breakdown": {k: round(v, 6) for k, v in country_weights.items()},
                    "country_hhi": self._calculate_hhi(list(country_weights.values()))
                }
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Country concentration evaluation error: {str(e)}"}
    
    def _evaluate_industry_concentration(self, threshold: float, operator: str, 
                                       filter_industry: str = None, rule: ComplianceRule = None) -> Dict[str, Any]:
        """Evaluate industry concentration limits"""
        try:
            industry_weights = defaultdict(float)
            
            for symbol, position in self.portfolio_cache.items():
                industry = position.get("industry", "Unknown")
                industry_weights[industry] += position["weight"]
            
            industry_weights = dict(industry_weights)
            
            if filter_industry:
                observed_value = industry_weights.get(filter_industry, 0)
            else:
                observed_value = max(industry_weights.values()) if industry_weights else 0
            
            violation = self._check_threshold(observed_value, threshold, operator)
            materiality_exceeded = self._check_materiality(
                observed_value, threshold, rule.materiality_bps if rule else 0
            )
            
            return {
                "status": "breach" if violation and materiality_exceeded else "compliant",
                "observed_value": round(observed_value, 6),
                "threshold": threshold,
                "metric": "industry_weight",
                "operator": operator,
                "materiality_exceeded": materiality_exceeded,
                "details": {
                    "industry_breakdown": {k: round(v, 6) for k, v in industry_weights.items()}
                }
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Industry concentration evaluation error: {str(e)}"}
    
    def _evaluate_rating_concentration(self, threshold: float, operator: str, 
                                     filter_rating: str = None, rule: ComplianceRule = None) -> Dict[str, Any]:
        """Evaluate credit rating concentration limits"""
        try:
            rating_weights = defaultdict(float)
            
            for symbol, position in self.portfolio_cache.items():
                rating = position.get("rating", "NR")
                rating_weights[rating] += position["weight"]
            
            rating_weights = dict(rating_weights)
            
            if filter_rating:
                observed_value = rating_weights.get(filter_rating, 0)
            else:
                observed_value = max(rating_weights.values()) if rating_weights else 0
            
            violation = self._check_threshold(observed_value, threshold, operator)
            materiality_exceeded = self._check_materiality(
                observed_value, threshold, rule.materiality_bps if rule else 0
            )
            
            return {
                "status": "breach" if violation and materiality_exceeded else "compliant",
                "observed_value": round(observed_value, 6),
                "threshold": threshold,
                "metric": "rating_weight",
                "operator": operator,
                "materiality_exceeded": materiality_exceeded,
                "details": {
                    "rating_breakdown": {k: round(v, 6) for k, v in rating_weights.items()},
                    "investment_grade_pct": round(
                        sum(v for k, v in rating_weights.items() 
                            if k in ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"]) * 100, 2
                    )
                }
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Rating concentration evaluation error: {str(e)}"}
    
    def _evaluate_total_exposure(self, threshold: float, operator: str, rule: ComplianceRule = None) -> Dict[str, Any]:
        """Evaluate total portfolio exposure"""
        try:
            total_exposure = sum(position["weight"] for position in self.portfolio_cache.values())
            violation = self._check_threshold(total_exposure, threshold, operator)
            
            materiality_exceeded = self._check_materiality(
                total_exposure, threshold, rule.materiality_bps if rule else 0
            )
            
            return {
                "status": "breach" if violation and materiality_exceeded else "compliant",
                "observed_value": round(total_exposure, 6),
                "threshold": threshold,
                "metric": "total_exposure",
                "operator": operator,
                "materiality_exceeded": materiality_exceeded,
                "details": {
                    "position_count": len(self.portfolio_cache),
                    "total_market_value": sum(position["market_value"] for position in self.portfolio_cache.values()),
                    "exposure_variance": round(abs(1.0 - total_exposure), 6)
                }
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Total exposure evaluation error: {str(e)}"}
    
    def _evaluate_leverage_ratio(self, threshold: float, operator: str, rule: ComplianceRule = None) -> Dict[str, Any]:
        """Evaluate portfolio leverage ratio"""
        try:
            # Simplified leverage calculation - in production this would be more complex
            total_market_value = sum(position["market_value"] for position in self.portfolio_cache.values())
            
            # Assume 1:1 leverage for equity positions (no actual leverage)
            # In reality, you'd calculate based on derivatives, margins, borrowed funds, etc.
            leverage_ratio = 1.0
            
            violation = self._check_threshold(leverage_ratio, threshold, operator)
            materiality_exceeded = self._check_materiality(
                leverage_ratio, threshold, rule.materiality_bps if rule else 0
            )
            
            return {
                "status": "breach" if violation and materiality_exceeded else "compliant",
                "observed_value": leverage_ratio,
                "threshold": threshold,
                "metric": "leverage_ratio",
                "operator": operator,
                "materiality_exceeded": materiality_exceeded,
                "details": {
                    "calculation_method": "simplified_equity_only",
                    "total_market_value": total_market_value,
                    "note": "Production implementation should include derivatives and margin calculations"
                }
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Leverage ratio evaluation error: {str(e)}"}
    
    def _evaluate_list_constraint(self, rule: ComplianceRule, expression: Dict) -> Dict[str, Any]:
        """Comprehensive list constraint evaluation"""
        try:
            field = expression.get("field")
            allowed_values = expression.get("allowed_values", [])
            denied_values = expression.get("denied_values", [])
            scope = expression.get("scope", "position")
            
            if not field:
                return {"status": "error", "message": "Missing field specification in list constraint"}
            
            violations = []
            compliant_positions = []
            
            for symbol, position in self.portfolio_cache.items():
                value = position.get(field)
                position_info = {
                    "symbol": symbol,
                    "name": position.get("name", symbol),
                    "field": field,
                    "value": value,
                    "weight": position["weight"],
                    "market_value": position["market_value"],
                    "sector": position.get("sector"),
                    "country": position.get("country")
                }
                
                is_violation = False
                violation_reason = None
                
                # Check allowed values constraint
                if allowed_values and value not in allowed_values:
                    is_violation = True
                    violation_reason = "not_in_allowed_list"
                
                # Check denied values constraint
                if denied_values and value in denied_values:
                    is_violation = True
                    violation_reason = "in_denied_list"
                
                if is_violation:
                    violations.append({
                        **position_info,
                        "violation_type": violation_reason
                    })
                else:
                    compliant_positions.append(position_info)
            
            # Calculate violation statistics
            total_positions = len(self.portfolio_cache)
            violation_count = len(violations)
            violation_weight = sum(v["weight"] for v in violations)
            violation_market_value = sum(v["market_value"] for v in violations)
            
            # Check materiality
            materiality_exceeded = self._check_materiality(
                violation_weight, 0, rule.materiality_bps if rule else 0
            )
            
            return {
                "status": "breach" if violations and materiality_exceeded else "compliant",
                "violation_count": violation_count,
                "violation_weight": round(violation_weight, 6),
                "violation_market_value": violation_market_value,
                "materiality_exceeded": materiality_exceeded,
                "field": field,
                "allowed_values": allowed_values,
                "denied_values": denied_values,
                "violations": violations,
                "details": {
                    "total_positions": total_positions,
                    "compliant_count": len(compliant_positions),
                    "violation_weight_pct": round(violation_weight * 100, 2),
                    "violation_summary": self._summarize_violations(violations, field),
                    "value_distribution": self._analyze_field_distribution(field)
                }
            }
            
        except Exception as e:
            return {"status": "error", "message": f"List constraint evaluation error: {str(e)}"}
    
    def _evaluate_temporal_window(self, rule: ComplianceRule, expression: Dict) -> Dict[str, Any]:
        """Evaluate temporal constraints (holding periods, lock-ups)"""
        try:
            metric = expression.get("metric")
            minimum_days = expression.get("minimum_days", 0)
            maximum_days = expression.get("maximum_days")
            scope = expression.get("scope", "position")
            
            current_time = datetime.now()
            violations = []
            
            for symbol, position in self.portfolio_cache.items():
                # Calculate holding period from acquisition date
                acquisition_date_str = position.get("acquisition_date")
                if acquisition_date_str:
                    try:
                        acquisition_date = datetime.fromisoformat(acquisition_date_str.replace('Z', '+00:00'))
                        holding_days = (current_time - acquisition_date).days
                    except:
                        holding_days = 0
                else:
                    # Fallback to last_updated if acquisition_date not available
                    last_updated_str = position.get("last_updated")
                    if last_updated_str:
                        try:
                            last_update_time = datetime.fromisoformat(last_updated_str.replace('Z', '+00:00'))
                            holding_days = (current_time - last_update_time).days
                        except:
                            holding_days = 0
                    else:
                        holding_days = 0
                
                violation = False
                violation_reason = None
                
                if metric == "holding_period":
                    if holding_days < minimum_days:
                        violation = True
                        violation_reason = f"Minimum holding period not met: {holding_days} < {minimum_days} days"
                    elif maximum_days and holding_days > maximum_days:
                        violation = True
                        violation_reason = f"Maximum holding period exceeded: {holding_days} > {maximum_days} days"
                
                if violation:
                    violations.append({
                        "symbol": symbol,
                        "name": position.get("name", symbol),
                        "holding_days": holding_days,
                        "minimum_required": minimum_days,
                        "maximum_allowed": maximum_days,
                        "violation_reason": violation_reason,
                        "weight": position["weight"],
                        "acquisition_date": acquisition_date_str
                    })
            
            violation_weight = sum(v["weight"] for v in violations)
            materiality_exceeded = self._check_materiality(
                violation_weight, 0, rule.materiality_bps if rule else 0
            )
            
            return {
                "status": "breach" if violations and materiality_exceeded else "compliant",
                "violation_count": len(violations),
                "violation_weight": round(violation_weight, 6),
                "materiality_exceeded": materiality_exceeded,
                "violations": violations,
                "metric": metric,
                "minimum_days": minimum_days,
                "maximum_days": maximum_days,
                "details": {
                    "positions_checked": len(self.portfolio_cache),
                    "avg_holding_period": self._calculate_average_holding_period(),
                    "implementation_note": "Uses acquisition_date or last_updated timestamps"
                }
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Temporal window evaluation error: {str(e)}"}
    
    def _evaluate_process_control(self, rule: ComplianceRule, expression: Dict) -> Dict[str, Any]:
        """Evaluate process controls and create cases if needed"""
        try:
            approval_required = expression.get("approval_required", True)
            approver_role = expression.get("approver_role", "compliance_officer")
            evidence_required = expression.get("evidence_required", "Approval documentation required")
            sla_days = expression.get("sla_days", 5)
            
            # Check if there's already an open case for this rule
            existing_case = self.db.query(ComplianceCase).filter(
                ComplianceCase.rule_id == rule.rule_id,
                ComplianceCase.status.in_([CaseStatus.OPEN, CaseStatus.IN_REVIEW])
            ).first()
            
            if existing_case:
                return {
                    "status": "case_exists",
                    "case_id": existing_case.case_id,
                    "message": "Process control case already exists",
                    "existing_case_status": existing_case.status.value,
                    "sla_deadline": existing_case.sla_deadline.isoformat() if existing_case.sla_deadline else None
                }
            
            # Create new case
            case = ComplianceCase(
                rule_id=rule.rule_id,
                case_type=CaseType.PROCESS_CONTROL,
                title=f"Process Control: {rule.name or rule.description[:50]}",
                description=f"Process control required: {rule.description}",
                evidence_required=evidence_required,
                sla_deadline=datetime.now() + timedelta(days=sla_days),
                status=CaseStatus.OPEN,
                priority="high" if rule.severity == RuleSeverity.CRITICAL else "medium"
            )
            
            self.db.add(case)
            self.db.commit()
            self.db.refresh(case)
            
            return {
                "status": "case_created",
                "case_id": case.case_id,
                "case_type": "process_control",
                "message": "Process control case created",
                "sla_deadline": case.sla_deadline.isoformat(),
                "priority": case.priority,
                "details": {
                    "approval_required": approval_required,
                    "approver_role": approver_role,
                    "evidence_required": evidence_required,
                    "sla_days": sla_days
                }
            }
            
        except Exception as e:
            self.db.rollback()
            return {"status": "error", "message": f"Process control evaluation error: {str(e)}"}
    
    def _evaluate_reporting_disclosure(self, rule: ComplianceRule, expression: Dict) -> Dict[str, Any]:
        """Evaluate reporting and disclosure requirements"""
        try:
            report_type = expression.get("report_type", "compliance_report")
            frequency = expression.get("frequency", "monthly")
            deadline_days = expression.get("deadline_days", 30)
            recipient = expression.get("recipient", "regulator")
            
            # Check if there's already an open case for this rule
            existing_case = self.db.query(ComplianceCase).filter(
                ComplianceCase.rule_id == rule.rule_id,
                ComplianceCase.status.in_([CaseStatus.OPEN, CaseStatus.IN_REVIEW])
            ).first()
            
            if existing_case:
                return {
                    "status": "case_exists",
                    "case_id": existing_case.case_id,
                    "message": "Reporting disclosure case already exists",
                    "existing_case_status": existing_case.status.value
                }
            
            # Create new reporting case
            case = ComplianceCase(
                rule_id=rule.rule_id,
                case_type=CaseType.REPORTING_DISCLOSURE,
                title=f"Reporting Required: {report_type}",
                description=f"Reporting required: {rule.description}",
                evidence_required=f"{report_type} must be submitted to {recipient}",
                sla_deadline=datetime.now() + timedelta(days=deadline_days),
                status=CaseStatus.OPEN,
                priority="medium"
            )
            
            self.db.add(case)
            self.db.commit()
            self.db.refresh(case)
            
            return {
                "status": "case_created",
                "case_id": case.case_id,
                "case_type": "reporting_disclosure",
                "message": "Reporting disclosure case created",
                "sla_deadline": case.sla_deadline.isoformat(),
                "details": {
                    "report_type": report_type,
                    "frequency": frequency,
                    "deadline_days": deadline_days,
                    "recipient": recipient
                }
            }
            
        except Exception as e:
            self.db.rollback()
            return {"status": "error", "message": f"Reporting disclosure evaluation error: {str(e)}"}
    
    # Helper methods continue...
    def _check_threshold(self, observed: float, threshold: float, operator: str) -> bool:
        """Check if observed value violates threshold based on operator"""
        try:
            if operator == "<=":
                return observed > threshold
            elif operator == ">=":
                return observed < threshold
            elif operator == "<":
                return observed >= threshold
            elif operator == ">":
                return observed <= threshold
            elif operator == "==":
                return abs(observed - threshold) > 1e-9  # Handle floating point precision
            elif operator == "!=":
                return abs(observed - threshold) <= 1e-9
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False
        except (TypeError, ValueError) as e:
            logger.error(f"Threshold check error: {e}")
            return False
    
    def _check_materiality(self, observed: float, threshold: float, materiality_bps: int) -> bool:
        """Check if breach exceeds materiality threshold"""
        if materiality_bps <= 0:
            return True  # No materiality threshold
        
        try:
            materiality_threshold = materiality_bps / 10000.0  # Convert basis points to decimal
            breach_magnitude = abs(observed - threshold)
            return breach_magnitude >= materiality_threshold
        except:
            return True  # Default to material if calculation fails
    
    def _calculate_hhi(self, weights: List[float]) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration"""
        try:
            return round(sum(w * w for w in weights), 4)
        except:
            return 0.0
    
    def _calculate_average_holding_period(self) -> Optional[float]:
        """Calculate average holding period across portfolio"""
        try:
            current_time = datetime.now()
            holding_periods = []
            
            for position in self.portfolio_cache.values():
                acquisition_date_str = position.get("acquisition_date")
                if acquisition_date_str:
                    try:
                        acquisition_date = datetime.fromisoformat(acquisition_date_str.replace('Z', '+00:00'))
                        holding_days = (current_time - acquisition_date).days
                        holding_periods.append(holding_days)
                    except:
                        continue
            
            if holding_periods:
                return round(sum(holding_periods) / len(holding_periods), 1)
            return None
        except:
            return None
    
    def _summarize_violations(self, violations: List[Dict], field: str) -> Dict[str, Any]:
        """Create a summary of list constraint violations"""
        if not violations:
            return {}
        
        violation_by_value = defaultdict(int)
        violation_by_type = defaultdict(int)
        
        for violation in violations:
            violation_by_value[violation["value"]] += 1
            violation_by_type[violation["violation_type"]] += 1
        
        return {
            "by_value": dict(violation_by_value),
            "by_type": dict(violation_by_type),
            "most_common_violation": max(violation_by_value.items(), key=lambda x: x[1]) if violation_by_value else None
        }
    
    def _analyze_field_distribution(self, field: str) -> Dict[str, Any]:
        """Analyze distribution of field values across portfolio"""
        try:
            field_values = defaultdict(float)  # value -> weight
            
            for position in self.portfolio_cache.values():
                value = position.get(field, "Unknown")
                field_values[value] += position["weight"]
            
            return {k: round(v, 4) for k, v in dict(field_values).items()}
        except:
            return {}
    
    def _assess_portfolio_freshness(self) -> str:
        """Assess how fresh the portfolio data is"""
        try:
            current_time = datetime.now()
            most_recent_update = None
            
            for position in self.portfolio_cache.values():
                last_updated_str = position.get("last_updated")
                if last_updated_str:
                    try:
                        last_updated = datetime.fromisoformat(last_updated_str.replace('Z', '+00:00'))
                        if most_recent_update is None or last_updated > most_recent_update:
                            most_recent_update = last_updated
                    except:
                        continue
            
            if most_recent_update:
                hours_old = (current_time - most_recent_update).total_seconds() / 3600
                if hours_old < 1:
                    return "fresh"
                elif hours_old < 24:
                    return "recent"
                elif hours_old < 168:  # 1 week
                    return "aging"
                else:
                    return "stale"
            
            return "unknown"
        except:
            return "unknown"
    
    # Continue with remaining helper methods in next part...
    
    def _handle_breach_result(self, rule_result: Dict[str, Any], results: Dict[str, Any]):
        """Handle breach result and create breach record"""
        try:
            rule_id = rule_result.get("rule_id")
            rule = self.rule_cache.get(rule_id, {}).get("rule")
            
            if not rule:
                return
            
            # Create breach record
            breach = self._create_breach_record(rule, rule_result)
            if breach:
                results["new_breaches"].append({
                    "breach_id": breach.breach_id,
                    "rule_id": rule_id,
                    "rule_name": rule_result.get("rule_name"),
                    "rule_description": rule_result.get("rule_description"),
                    "severity": rule_result.get("severity"),
                    "observed_value": rule_result.get("observed_value"),
                    "threshold": rule_result.get("threshold"),
                    "timestamp": breach.breach_timestamp.isoformat(),
                    "materiality_exceeded": rule_result.get("materiality_exceeded", True),
                    "violation_count": rule_result.get("violation_count", 1),
                    "details": rule_result.get("details", {})
                })
        except Exception as e:
            logger.error(f"Error handling breach result: {e}")
    
    def _handle_case_result(self, rule_result: Dict[str, Any], results: Dict[str, Any]):
        """Handle case creation result"""
        try:
            if "case_id" in rule_result:
                results["new_cases"].append({
                    "case_id": rule_result["case_id"],
                    "rule_id": rule_result.get("rule_id"),
                    "case_type": rule_result.get("case_type"),
                    "priority": rule_result.get("priority", "medium"),
                    "sla_deadline": rule_result.get("sla_deadline"),
                    "description": rule_result.get("rule_description")
                })
        except Exception as e:
            logger.error(f"Error handling case result: {e}")
    
    def _create_breach_record(self, rule: ComplianceRule, evaluation_result: Dict) -> Optional[ComplianceBreach]:
        """Create a breach record if one doesn't already exist"""
        try:
            # Check for existing open breach
            existing_breach = self.db.query(ComplianceBreach).filter(
                ComplianceBreach.rule_id == rule.rule_id,
                ComplianceBreach.status == BreachStatus.OPEN
            ).first()
            
            if existing_breach:
                # Update existing breach with new data
                existing_breach.observed_value = evaluation_result.get("observed_value")
                existing_breach.threshold_value = evaluation_result.get("threshold")
                existing_breach.portfolio_snapshot = self.portfolio_cache
                existing_breach.breach_magnitude = self._calculate_breach_magnitude_value(
                    evaluation_result.get("observed_value", 0),
                    evaluation_result.get("threshold", 0)
                )
                existing_breach.updated_at = datetime.now()
                self.db.commit()
                return existing_breach
            
            # Create new breach record
            breach = ComplianceBreach(
                rule_id=rule.rule_id,
                portfolio_snapshot=self.portfolio_cache,
                observed_value=evaluation_result.get("observed_value"),
                threshold_value=evaluation_result.get("threshold"),
                breach_magnitude=self._calculate_breach_magnitude_value(
                    evaluation_result.get("observed_value", 0),
                    evaluation_result.get("threshold", 0)
                ),
                status=BreachStatus.OPEN,
                impact_assessment={
                    "materiality_exceeded": evaluation_result.get("materiality_exceeded", True),
                    "violation_count": evaluation_result.get("violation_count", 1),
                    "evaluation_details": evaluation_result.get("details", {})
                }
            )
            
            self.db.add(breach)
            self.db.commit()
            self.db.refresh(breach)
            
            logger.info(f"Created new breach record: {breach.breach_id} for rule: {rule.rule_id}")
            return breach
            
        except Exception as e:
            logger.error(f"Error creating breach record for rule {rule.rule_id}: {e}")
            self.db.rollback()
            return None
    
    def _calculate_breach_magnitude_value(self, observed: float, threshold: float) -> Optional[float]:
        """Calculate breach magnitude as percentage over threshold"""
        try:
            if threshold == 0:
                return observed if observed > 0 else None
            return round(((observed - threshold) / threshold) * 100, 2)
        except:
            return None
    
    def _record_evaluation(self, rule_id: str, result: Dict, execution_time_ms: float):
        """Record rule evaluation for audit purposes"""
        try:
            evaluation = RuleEvaluation(
                rule_id=rule_id,
                portfolio_snapshot=self.portfolio_cache,
                evaluation_result=result,
                triggered_breach=result.get("status") == "breach",
                triggered_case=result.get("status") == "case_created",
                execution_time_ms=execution_time_ms,
                evaluation_context=self.evaluation_context
            )
            
            self.db.add(evaluation)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error recording evaluation for rule {rule_id}: {e}")
            self.db.rollback()
    
    def _get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary statistics"""
        try:
            if not self.portfolio_cache:
                return {"total_positions": 0, "total_weight": 0, "total_market_value": 0}
            
            total_positions = len(self.portfolio_cache)
            total_weight = sum(pos["weight"] for pos in self.portfolio_cache.values())
            total_market_value = sum(pos["market_value"] for pos in self.portfolio_cache.values())
            
            # Sector breakdown
            sector_weights = defaultdict(float)
            for pos in self.portfolio_cache.values():
                sector_weights[pos["sector"]] += pos["weight"]
            
            # Country breakdown
            country_weights = defaultdict(float)
            for pos in self.portfolio_cache.values():
                country_weights[pos["country"]] += pos["weight"]
            
            # Rating breakdown
            rating_weights = defaultdict(float)
            for pos in self.portfolio_cache.values():
                rating_weights[pos["rating"]] += pos["weight"]
            
            return {
                "total_positions": total_positions,
                "total_weight": round(total_weight, 6),
                "total_market_value": total_market_value,
                "top_5_positions": dict(sorted(
                    {symbol: pos["weight"] for symbol, pos in self.portfolio_cache.items()}.items(),
                    key=lambda x: x[1], reverse=True
                )[:5]),
                "sector_breakdown": {k: round(v, 4) for k, v in dict(sector_weights).items()},
                "country_breakdown": {k: round(v, 4) for k, v in dict(country_weights).items()},
                "rating_breakdown": {k: round(v, 4) for k, v in dict(rating_weights).items()},
                "concentration_metrics": {
                    "sector_hhi": self._calculate_hhi(list(sector_weights.values())),
                    "country_hhi": self._calculate_hhi(list(country_weights.values())),
                    "issuer_hhi": self._calculate_hhi([pos["weight"] for pos in self.portfolio_cache.values()])
                },
                "cache_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating portfolio summary: {e}")
            return {"error": str(e)}
    
    # Public methods for breach and case management
    
    def resolve_breach(self, breach_id: str, resolved_by: str, notes: str = "", 
                      resolution_type: str = "resolved") -> bool:
        """Resolve an open breach with comprehensive validation"""
        try:
            valid_statuses = {
                "resolved": BreachStatus.RESOLVED,
                "false_positive": BreachStatus.FALSE_POSITIVE,
                "under_review": BreachStatus.UNDER_REVIEW
            }
            
            if resolution_type not in valid_statuses:
                resolution_type = "resolved"
            
            breach = self.db.query(ComplianceBreach).filter(
                ComplianceBreach.breach_id == breach_id,
                ComplianceBreach.status == BreachStatus.OPEN
            ).first()
            
            if not breach:
                logger.warning(f"Breach {breach_id} not found or already resolved")
                return False
            
            breach.status = valid_statuses[resolution_type]
            breach.resolved_at = datetime.now()
            breach.resolved_by = resolved_by
            breach.resolution_notes = notes
            breach.updated_at = datetime.now()
            
            self.db.commit()
            
            logger.info(f"Breach {breach_id} resolved as {resolution_type} by {resolved_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving breach {breach_id}: {e}")
            self.db.rollback()
            return False
    
    def get_active_breaches(self, severity_filter: str = None) -> List[Dict[str, Any]]:
        """Get all active breaches with comprehensive details"""
        try:
            query = self.db.query(ComplianceBreach).join(
                ComplianceRule, ComplianceBreach.rule_id == ComplianceRule.rule_id
            ).filter(ComplianceBreach.status == "OPEN")
            
            if severity_filter:
                query = query.filter(ComplianceRule.severity == severity_filter)
            
            breaches = query.all()
            
            breach_list = []
            for breach in breaches:
                breach_data = {
                    "breach_id": breach.breach_id,
                    "rule_id": breach.rule_id,
                    "rule_name": breach.rule.name,
                    "rule_description": breach.rule.description,
                    "control_type": str(breach.rule.control_type),
                    "severity": str(breach.rule.severity),
                    "observed_value": breach.observed_value,
                    "threshold": breach.threshold_value,
                    "breach_magnitude": breach.breach_magnitude,
                    "breach_timestamp": breach.breach_timestamp.isoformat(),
                    "status": str(breach.status),
                    "impact_assessment": breach.impact_assessment,
                    "portfolio_snapshot_size": len(breach.portfolio_snapshot) if breach.portfolio_snapshot else 0,
                    "external_reference": breach.external_reference
                }
                
                # Calculate breach age and categorize
                if breach.breach_timestamp:
                    age_hours = (datetime.now() - breach.breach_timestamp).total_seconds() / 3600
                    breach_data["age_hours"] = round(age_hours, 1)
                    breach_data["age_category"] = self._categorize_breach_age(age_hours)
                    breach_data["sla_status"] = self._assess_breach_sla(age_hours, breach.rule.severity.value)
                
                breach_list.append(breach_data)
            
            # Sort by severity and age
            severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            breach_list.sort(key=lambda x: (
                severity_order.get(x["severity"], 0),
                -x.get("age_hours", 0)
            ), reverse=True)
            
            return breach_list
            
        except Exception as e:
            logger.error(f"Error getting active breaches: {e}")
            return []
    
    def _categorize_breach_age(self, age_hours: float) -> str:
        """Categorize breach by age for prioritization"""
        if age_hours < 1:
            return "new"
        elif age_hours < 24:
            return "recent"
        elif age_hours < 168:  # 1 week
            return "aging"
        else:
            return "stale"
    
    def _assess_breach_sla(self, age_hours: float, severity: str) -> str:
        """Assess breach against SLA thresholds"""
        sla_thresholds = {
            "critical": 4,  # 4 hours
            "high": 24,     # 24 hours
            "medium": 72,   # 3 days
            "low": 168      # 1 week
        }
        
        threshold = sla_thresholds.get(severity, 24)
        
        if age_hours < threshold * 0.5:
            return "within_sla"
        elif age_hours < threshold:
            return "approaching_sla"
        else:
            return "exceeds_sla"
    
    def get_compliance_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive compliance analytics"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Basic statistics
            total_rules = self.db.query(ComplianceRule).filter(ComplianceRule.is_active == True).count()
            
            # Breach statistics
            total_breaches = self.db.query(ComplianceBreach).count()
            open_breaches = self.db.query(ComplianceBreach).filter(
                ComplianceBreach.status == "OPEN"
            ).count()
            
            period_breaches = self.db.query(ComplianceBreach).filter(
                ComplianceBreach.breach_timestamp >= start_date
            ).all()
            
            # Evaluation statistics
            period_evaluations = self.db.query(RuleEvaluation).filter(
                RuleEvaluation.evaluation_timestamp >= start_date
            ).count()
            
            # Performance metrics
            avg_execution_times = self.db.query(RuleEvaluation).filter(
                RuleEvaluation.evaluation_timestamp >= start_date,
                RuleEvaluation.execution_time_ms.isnot(None)
            ).all()
            
            if avg_execution_times:
                avg_execution_time = sum(e.execution_time_ms for e in avg_execution_times) / len(avg_execution_times)
            else:
                avg_execution_time = 0
            
            # Breach analysis
            breach_by_severity = defaultdict(int)
            breach_by_control_type = defaultdict(int)
            breach_by_status = defaultdict(int)
            
            for breach in period_breaches:
                if breach.rule:
                    breach_by_severity[breach.rule.severity.value] += 1
                    breach_by_control_type[breach.rule.control_type.value] += 1
                breach_by_status[breach.status.value] += 1
            
            # Resolution statistics
            resolved_breaches = [b for b in period_breaches if b.status == BreachStatus.RESOLVED and b.resolved_at]
            if resolved_breaches:
                resolution_times = [
                    (b.resolved_at - b.breach_timestamp).total_seconds() / 3600 
                    for b in resolved_breaches
                ]
                avg_resolution_time = sum(resolution_times) / len(resolution_times)
                median_resolution_time = sorted(resolution_times)[len(resolution_times) // 2]
            else:
                avg_resolution_time = 0
                median_resolution_time = 0
            
            # Calculate compliance rate
            if total_rules > 0:
                compliance_rate = ((total_rules - open_breaches) / total_rules) * 100
            else:
                compliance_rate = 100.0
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days
                },
                "summary": {
                    "total_active_rules": total_rules,
                    "total_breaches_all_time": total_breaches,
                    "open_breaches": open_breaches,
                    "period_breaches": len(period_breaches),
                    "compliance_rate": round(compliance_rate, 2)
                },
                "breach_analysis": {
                    "by_severity": dict(breach_by_severity),
                    "by_control_type": dict(breach_by_control_type),
                    "by_status": dict(breach_by_status),
                    "resolution_rate": round(len(resolved_breaches) / max(len(period_breaches), 1) * 100, 2),
                    "avg_resolution_time_hours": round(avg_resolution_time, 1),
                    "median_resolution_time_hours": round(median_resolution_time, 1)
                },
                "performance": {
                    "total_evaluations": period_evaluations,
                    "avg_execution_time_ms": round(avg_execution_time, 2),
                    "evaluations_per_day": round(period_evaluations / max(days, 1), 1),
                    "system_reliability": round(
                        (period_evaluations - len([e for e in avg_execution_times if e.execution_time_ms is None])) /
                        max(period_evaluations, 1) * 100, 2
                    )
                },
                "trends": self._calculate_compliance_trends(days),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating compliance analytics: {e}")
            return {"error": str(e)}
    
    def _calculate_compliance_trends(self, days: int) -> Dict[str, Any]:
        """Calculate compliance trends over time"""
        try:
            end_date = datetime.now()
            mid_date = end_date - timedelta(days=days//2)
            start_date = end_date - timedelta(days=days)
            
            # Get breaches for each half of the period
            first_half_breaches = self.db.query(ComplianceBreach).filter(
                ComplianceBreach.breach_timestamp >= start_date,
                ComplianceBreach.breach_timestamp < mid_date
            ).count()
            
            second_half_breaches = self.db.query(ComplianceBreach).filter(
                ComplianceBreach.breach_timestamp >= mid_date,
                ComplianceBreach.breach_timestamp <= end_date
            ).count()
            
            # Calculate trend
            if first_half_breaches > 0:
                trend_percentage = ((second_half_breaches - first_half_breaches) / first_half_breaches) * 100
            else:
                trend_percentage = 0 if second_half_breaches == 0 else 100
            
            if trend_percentage > 10:
                trend = "worsening"
            elif trend_percentage < -10:
                trend = "improving"
            else:
                trend = "stable"
            
            return {
                "trend": trend,
                "trend_percentage": round(trend_percentage, 1),
                "first_half_breaches": first_half_breaches,
                "second_half_breaches": second_half_breaches,
                "analysis_period_days": days
            }
            
        except Exception as e:
            return {"trend": "unknown", "error": str(e)}