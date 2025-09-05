"""
Analytics and Reporting API Routes
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

from app.models.database import (
    get_db, ComplianceRule, ComplianceBreach, ComplianceCase, Portfolio,
    RuleEvaluation, PositionHistory, PolicyDocument
)
from app.schemas import ComplianceAnalytics, BaseResponse
from app.api.dependencies import get_compliance_engine, get_current_user
from app.services.compliance_engine import ComplianceEngine
from app.utils.logger import get_logger, compliance_logger
from app.utils.exceptions import ValidationException

logger = get_logger(__name__)
router = APIRouter()


@router.get("/compliance-summary")
async def get_compliance_summary(
    days: int = Query(30, ge=1, le=365, description="Analysis period in days"),
    include_trends: bool = Query(True, description="Include trend analysis"),
    engine: ComplianceEngine = Depends(get_compliance_engine),
    current_user: str = Depends(get_current_user)
):
    """
    Get comprehensive compliance analytics summary
    """
    try:
        # Get comprehensive analytics
        analytics = engine.get_compliance_analytics(days=days)
        
        return {
            "summary": analytics,
            "generated_for_user": current_user,
            "analysis_period_days": days,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating compliance summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate compliance summary: {str(e)}"
        )


@router.get("/breach-analysis")
async def get_breach_analysis(
    days: int = Query(30, ge=1, le=365, description="Analysis period in days"),
    group_by: str = Query("severity", description="Group by: severity, control_type, rule"),
    include_resolution_metrics: bool = Query(True, description="Include resolution metrics"),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Detailed breach analysis with grouping and metrics
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get breaches in period with rule information
        breaches_query = db.query(ComplianceBreach).join(
            ComplianceRule, ComplianceBreach.rule_id == ComplianceRule.rule_id
        ).filter(
            ComplianceBreach.breach_timestamp >= start_date
        )
        
        breaches = breaches_query.all()
        
        # Group and analyze breaches
        grouped_data = {}
        resolution_times = []
        
        for breach in breaches:
            # Determine grouping key
            if group_by == "severity":
                group_key = breach.rule.severity.value
            elif group_by == "control_type":
                group_key = breach.rule.control_type.value
            elif group_by == "rule":
                group_key = f"{breach.rule.name or breach.rule.description[:50]}..."
            else:
                group_key = "all"
            
            if group_key not in grouped_data:
                grouped_data[group_key] = {
                    "total_breaches": 0,
                    "open_breaches": 0,
                    "resolved_breaches": 0,
                    "false_positives": 0,
                    "avg_resolution_time_hours": 0,
                    "breach_magnitude_avg": 0,
                    "recent_breaches": []
                }
            
            group_data = grouped_data[group_key]
            group_data["total_breaches"] += 1
            
            if breach.status.value == "open":
                group_data["open_breaches"] += 1
            elif breach.status.value == "resolved":
                group_data["resolved_breaches"] += 1
                
                # Calculate resolution time if available
                if breach.resolved_at and include_resolution_metrics:
                    resolution_time = (breach.resolved_at - breach.breach_timestamp).total_seconds() / 3600
                    resolution_times.append(resolution_time)
            elif breach.status.value == "false_positive":
                group_data["false_positives"] += 1
            
            # Add to recent breaches (last 5)
            if len(group_data["recent_breaches"]) < 5:
                group_data["recent_breaches"].append({
                    "breach_id": breach.breach_id,
                    "rule_description": breach.rule.description[:100] + "...",
                    "observed_value": breach.observed_value,
                    "threshold": breach.threshold_value,
                    "magnitude": breach.breach_magnitude,
                    "timestamp": breach.breach_timestamp.isoformat(),
                    "status": breach.status.value
                })
            
            # Track breach magnitude
            if breach.breach_magnitude:
                group_data["breach_magnitude_avg"] = (
                    group_data["breach_magnitude_avg"] + breach.breach_magnitude
                ) / 2  # Running average
        
        # Calculate average resolution time per group
        if resolution_times and include_resolution_metrics:
            for group_key in grouped_data:
                group_breaches = [b for b in breaches if (
                    (group_by == "severity" and b.rule.severity.value == group_key) or
                    (group_by == "control_type" and b.rule.control_type.value == group_key) or
                    (group_by == "rule" and f"{b.rule.name or b.rule.description[:50]}..." == group_key)
                )]
                
                group_resolution_times = []
                for breach in group_breaches:
                    if breach.resolved_at and breach.status.value == "resolved":
                        rt = (breach.resolved_at - breach.breach_timestamp).total_seconds() / 3600
                        group_resolution_times.append(rt)
                
                if group_resolution_times:
                    grouped_data[group_key]["avg_resolution_time_hours"] = round(
                        sum(group_resolution_times) / len(group_resolution_times), 1
                    )
        
        # Overall statistics
        total_breaches = len(breaches)
        open_breaches = len([b for b in breaches if b.status.value == "open"])
        
        # Trend analysis
        if days >= 14:
            mid_date = start_date + timedelta(days=days//2)
            first_half_breaches = len([b for b in breaches if b.breach_timestamp < mid_date])
            second_half_breaches = len([b for b in breaches if b.breach_timestamp >= mid_date])
            
            trend = "stable"
            if first_half_breaches > 0:
                change_pct = ((second_half_breaches - first_half_breaches) / first_half_breaches) * 100
                if change_pct > 20:
                    trend = "worsening"
                elif change_pct < -20:
                    trend = "improving"
        else:
            trend = "insufficient_data"
        
        return {
            "analysis_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            },
            "grouping": group_by,
            "overall_metrics": {
                "total_breaches": total_breaches,
                "open_breaches": open_breaches,
                "resolution_rate": round((total_breaches - open_breaches) / max(total_breaches, 1) * 100, 2),
                "avg_resolution_time_hours": round(sum(resolution_times) / len(resolution_times), 1) if resolution_times else 0,
                "trend": trend
            },
            "grouped_analysis": grouped_data,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating breach analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Breach analysis failed: {str(e)}"
        )


@router.get("/performance-metrics")
async def get_performance_metrics(
    days: int = Query(7, ge=1, le=90, description="Analysis period in days"),
    include_rule_performance: bool = Query(True, description="Include individual rule performance"),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Get system performance metrics and rule evaluation statistics
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get rule evaluations in period
        evaluations = db.query(RuleEvaluation).filter(
            RuleEvaluation.evaluation_timestamp >= start_date
        ).all()
        
        if not evaluations:
            return {
                "message": "No evaluation data available for the specified period",
                "period": {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
                "metrics": {}
            }
        
        # Overall performance metrics
        total_evaluations = len(evaluations)
        successful_evaluations = len([e for e in evaluations if e.execution_time_ms is not None])
        failed_evaluations = total_evaluations - successful_evaluations
        
        # Execution time statistics
        execution_times = [e.execution_time_ms for e in evaluations if e.execution_time_ms is not None]
        
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        min_execution_time = min(execution_times) if execution_times else 0
        max_execution_time = max(execution_times) if execution_times else 0
        
        # Calculate percentiles
        sorted_times = sorted(execution_times) if execution_times else []
        p50 = sorted_times[len(sorted_times)//2] if sorted_times else 0
        p95 = sorted_times[int(len(sorted_times)*0.95)] if sorted_times else 0
        p99 = sorted_times[int(len(sorted_times)*0.99)] if sorted_times else 0
        
        # Breach detection statistics
        breaches_triggered = len([e for e in evaluations if e.triggered_breach])
        cases_triggered = len([e for e in evaluations if e.triggered_case])
        
        # Daily evaluation counts
        daily_counts = {}
        for eval in evaluations:
            eval_date = eval.evaluation_timestamp.date().isoformat()
            daily_counts[eval_date] = daily_counts.get(eval_date, 0) + 1
        
        # Rule-specific performance if requested
        rule_performance = {}
        if include_rule_performance:
            rule_stats = {}
            
            for eval in evaluations:
                rule_id = eval.rule_id
                if rule_id not in rule_stats:
                    rule_stats[rule_id] = {
                        "evaluations": 0,
                        "execution_times": [],
                        "breaches_triggered": 0,
                        "cases_triggered": 0,
                        "errors": 0
                    }
                
                rule_stat = rule_stats[rule_id]
                rule_stat["evaluations"] += 1
                
                if eval.execution_time_ms is not None:
                    rule_stat["execution_times"].append(eval.execution_time_ms)
                else:
                    rule_stat["errors"] += 1
                
                if eval.triggered_breach:
                    rule_stat["breaches_triggered"] += 1
                
                if eval.triggered_case:
                    rule_stat["cases_triggered"] += 1
            
            # Calculate averages and identify problematic rules
            for rule_id, stats in rule_stats.items():
                if stats["execution_times"]:
                    stats["avg_execution_time_ms"] = round(
                        sum(stats["execution_times"]) / len(stats["execution_times"]), 2
                    )
                    stats["max_execution_time_ms"] = max(stats["execution_times"])
                else:
                    stats["avg_execution_time_ms"] = 0
                    stats["max_execution_time_ms"] = 0
                
                stats["error_rate"] = round(stats["errors"] / stats["evaluations"] * 100, 2)
                stats["breach_rate"] = round(stats["breaches_triggered"] / stats["evaluations"] * 100, 2)
                
                # Remove raw execution times from output
                del stats["execution_times"]
            
            rule_performance = rule_stats
        
        return {
            "analysis_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            },
            "overall_performance": {
                "total_evaluations": total_evaluations,
                "successful_evaluations": successful_evaluations,
                "failed_evaluations": failed_evaluations,
                "success_rate": round(successful_evaluations / total_evaluations * 100, 2),
                "evaluations_per_day": round(total_evaluations / days, 1)
            },
            "execution_time_metrics": {
                "average_ms": round(avg_execution_time, 2),
                "minimum_ms": round(min_execution_time, 2),
                "maximum_ms": round(max_execution_time, 2),
                "median_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2)
            },
            "detection_metrics": {
                "breaches_triggered": breaches_triggered,
                "cases_triggered": cases_triggered,
                "breach_detection_rate": round(breaches_triggered / total_evaluations * 100, 4),
                "case_creation_rate": round(cases_triggered / total_evaluations * 100, 4)
            },
            "daily_evaluation_counts": daily_counts,
            "rule_performance": rule_performance if include_rule_performance else {},
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Performance metrics generation failed: {str(e)}"
        )


@router.get("/portfolio-analytics")
async def get_portfolio_analytics(
    include_concentration_analysis: bool = Query(True, description="Include concentration analysis"),
    include_historical_changes: bool = Query(False, description="Include historical position changes"),
    days_back: int = Query(30, ge=1, le=365, description="Days back for historical analysis"),
    engine: ComplianceEngine = Depends(get_compliance_engine),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Comprehensive portfolio analytics and risk assessment
    """
    try:
        # Load current portfolio
        portfolio_cache = engine._load_portfolio_cache()
        
        if not portfolio_cache:
            return {
                "message": "No portfolio data available",
                "analytics": {},
                "generated_at": datetime.now().isoformat()
            }
        
        # Basic portfolio summary
        portfolio_summary = engine._get_portfolio_summary()
        
        # Advanced concentration analysis
        concentration_analysis = {}
        if include_concentration_analysis:
            from collections import defaultdict
            
            # Calculate various concentration metrics
            sector_weights = defaultdict(float)
            country_weights = defaultdict(float)
            rating_weights = defaultdict(float)
            issuer_weights = {}
            
            for symbol, position in portfolio_cache.items():
                weight = position["weight"]
                sector_weights[position["sector"]] += weight
                country_weights[position["country"]] += weight
                rating_weights[position["rating"]] += weight
                issuer_weights[symbol] = weight
            
            # HHI calculations
            def calculate_hhi(weights_dict):
                return sum(weight * weight for weight in weights_dict.values())
            
            concentration_analysis = {
                "herfindahl_indices": {
                    "sector_hhi": round(calculate_hhi(sector_weights), 4),
                    "country_hhi": round(calculate_hhi(country_weights), 4),
                    "issuer_hhi": round(calculate_hhi(issuer_weights), 4),
                    "rating_hhi": round(calculate_hhi(rating_weights), 4)
                },
                "top_concentrations": {
                    "top_5_issuers": dict(sorted(issuer_weights.items(), key=lambda x: x[1], reverse=True)[:5]),
                    "top_3_sectors": dict(sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)[:3]),
                    "top_3_countries": dict(sorted(country_weights.items(), key=lambda x: x[1], reverse=True)[:3])
                },
                "risk_assessment": {
                    "concentration_risk": "high" if calculate_hhi(issuer_weights) > 0.25 else "medium" if calculate_hhi(issuer_weights) > 0.15 else "low",
                    "sector_diversification": "poor" if max(sector_weights.values()) > 0.4 else "good",
                    "geographic_diversification": "poor" if max(country_weights.values()) > 0.7 else "good"
                }
            }
        
        # Historical changes analysis
        historical_analysis = {}
        if include_historical_changes:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Get position history
            history_records = db.query(PositionHistory).filter(
                PositionHistory.change_timestamp >= start_date
            ).order_by(PositionHistory.change_timestamp.desc()).limit(100).all()
            
            # Analyze change patterns
            change_types = defaultdict(int)
            symbols_changed = set()
            daily_changes = defaultdict(int)
            
            for record in history_records:
                change_types[record.change_type] += 1
                symbols_changed.add(record.symbol)
                change_date = record.change_timestamp.date().isoformat()
                daily_changes[change_date] += 1
            
            historical_analysis = {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days_back
                },
                "change_summary": {
                    "total_changes": len(history_records),
                    "symbols_affected": len(symbols_changed),
                    "change_types": dict(change_types),
                    "avg_changes_per_day": round(len(history_records) / days_back, 1)
                },
                "daily_change_counts": dict(list(daily_changes.items())[-14:]),  # Last 14 days
                "most_active_symbols": list(symbols_changed)[:10]
            }
        
        # Portfolio quality metrics
        total_positions = len(portfolio_cache)
        investment_grade_count = sum(
            1 for pos in portfolio_cache.values()
            if pos["rating"] in ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"]
        )
        
        quality_metrics = {
            "total_positions": total_positions,
            "investment_grade_positions": investment_grade_count,
            "investment_grade_percentage": round(investment_grade_count / total_positions * 100, 1) if total_positions > 0 else 0,
            "unique_sectors": len(set(pos["sector"] for pos in portfolio_cache.values())),
            "unique_countries": len(set(pos["country"] for pos in portfolio_cache.values())),
            "currencies": list(set(pos["currency"] for pos in portfolio_cache.values())),
            "avg_position_size": round(portfolio_summary["total_weight"] / total_positions, 4) if total_positions > 0 else 0
        }
        
        return {
            "portfolio_summary": portfolio_summary,
            "concentration_analysis": concentration_analysis,
            "historical_analysis": historical_analysis,
            "quality_metrics": quality_metrics,
            "compliance_context": {
                "requires_review": concentration_analysis.get("risk_assessment", {}).get("concentration_risk") == "high",
                "diversification_score": _calculate_diversification_score(concentration_analysis),
                "overall_risk_level": _assess_overall_portfolio_risk(concentration_analysis, quality_metrics)
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating portfolio analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Portfolio analytics generation failed: {str(e)}"
        )


@router.get("/regulatory-report")
async def generate_regulatory_report(
    report_type: str = Query("comprehensive", description="Report type: comprehensive, breaches_only, summary"),
    period_days: int = Query(30, ge=1, le=365, description="Reporting period in days"),
    format: str = Query("json", description="Output format: json, text"),
    engine: ComplianceEngine = Depends(get_compliance_engine),
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate regulatory compliance report
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        # Get comprehensive analytics
        analytics = engine.get_compliance_analytics(days=period_days)
        
        # Get detailed breach information
        breaches = db.query(ComplianceBreach).join(
            ComplianceRule, ComplianceBreach.rule_id == ComplianceRule.rule_id
        ).filter(
            ComplianceBreach.breach_timestamp >= start_date
        ).all()
        
        breach_details = []
        for breach in breaches:
            breach_details.append({
                "breach_id": breach.breach_id,
                "rule_description": breach.rule.description,
                "control_type": breach.rule.control_type.value,
                "severity": breach.rule.severity.value,
                "breach_timestamp": breach.breach_timestamp.isoformat(),
                "observed_value": breach.observed_value,
                "threshold_value": breach.threshold_value,
                "breach_magnitude": breach.breach_magnitude,
                "status": breach.status.value,
                "resolution_time_hours": (
                    (breach.resolved_at - breach.breach_timestamp).total_seconds() / 3600
                    if breach.resolved_at else None
                ),
                "external_reference": breach.external_reference
            })
        
        # Get rule effectiveness metrics
        rule_effectiveness = {}
        active_rules = db.query(ComplianceRule).filter(ComplianceRule.is_active == True).all()
        
        for rule in active_rules:
            rule_breaches = [b for b in breaches if b.rule_id == rule.rule_id]
            rule_effectiveness[rule.rule_id] = {
                "rule_name": rule.name or rule.description[:50],
                "control_type": rule.control_type.value,
                "severity": rule.severity.value,
                "breach_count": len(rule_breaches),
                "last_breach": max([b.breach_timestamp for b in rule_breaches]).isoformat() if rule_breaches else None,
                "effectiveness_score": max(0, 100 - len(rule_breaches) * 10)  # Simple scoring
            }
        
        # Generate report content
        report_data = {
            "report_metadata": {
                "report_type": report_type,
                "reporting_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": period_days
                },
                "generated_at": datetime.now().isoformat(),
                "generated_by": current_user,
                "report_id": f"REG_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            },
            "executive_summary": {
                "overall_compliance_rate": analytics["summary"]["compliance_rate"],
                "total_active_rules": analytics["summary"]["total_active_rules"],
                "total_breaches": analytics["summary"]["period_breaches"],
                "open_breaches": analytics["summary"]["open_breaches"],
                "critical_findings": len([b for b in breach_details if b["severity"] == "critical"]),
                "recommendation": (
                    "Immediate attention required" if analytics["summary"]["open_breaches"] > 0
                    else "Compliance posture satisfactory"
                )
            },
            "detailed_findings": {
                "breach_analysis": analytics["breach_analysis"],
                "breach_details": breach_details if report_type in ["comprehensive", "breaches_only"] else [],
                "rule_effectiveness": rule_effectiveness if report_type == "comprehensive" else {},
                "performance_metrics": analytics["performance"] if report_type == "comprehensive" else {},
                "trend_analysis": analytics["trends"]
            },
            "regulatory_implications": {
                "notification_required": len([b for b in breach_details if b["severity"] in ["critical", "high"]]) > 0,
                "follow_up_actions": _generate_follow_up_actions(breach_details, analytics),
                "next_report_due": (end_date + timedelta(days=30)).isoformat()
            }
        }
        
        if format == "text":
            # Generate text format report
            text_report = _generate_text_report(report_data)
            return {"report_content": text_report, "format": "text"}
        else:
            return report_data
        
    except Exception as e:
        logger.error(f"Error generating regulatory report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Regulatory report generation failed: {str(e)}"
        )


# Helper functions
def _calculate_diversification_score(concentration_analysis: Dict) -> int:
    """Calculate portfolio diversification score (0-100)"""
    if not concentration_analysis:
        return 50
    
    hhi_indices = concentration_analysis.get("herfindahl_indices", {})
    
    # Lower HHI = better diversification
    sector_score = max(0, 100 - hhi_indices.get("sector_hhi", 0) * 400)
    country_score = max(0, 100 - hhi_indices.get("country_hhi", 0) * 200)
    issuer_score = max(0, 100 - hhi_indices.get("issuer_hhi", 0) * 500)
    
    return int((sector_score + country_score + issuer_score) / 3)


def _assess_overall_portfolio_risk(concentration_analysis: Dict, quality_metrics: Dict) -> str:
    """Assess overall portfolio risk level"""
    risk_factors = 0
    
    # Concentration risk
    if concentration_analysis.get("risk_assessment", {}).get("concentration_risk") == "high":
        risk_factors += 3
    elif concentration_analysis.get("risk_assessment", {}).get("concentration_risk") == "medium":
        risk_factors += 1
    
    # Credit quality risk
    ig_percentage = quality_metrics.get("investment_grade_percentage", 100)
    if ig_percentage < 70:
        risk_factors += 2
    elif ig_percentage < 85:
        risk_factors += 1
    
    # Diversification risk
    if quality_metrics.get("unique_sectors", 10) < 5:
        risk_factors += 1
    
    if risk_factors >= 5:
        return "high"
    elif risk_factors >= 2:
        return "medium"
    else:
        return "low"


def _generate_follow_up_actions(breach_details: List[Dict], analytics: Dict) -> List[str]:
    """Generate follow-up actions based on breach analysis"""
    actions = []
    
    critical_breaches = [b for b in breach_details if b["severity"] == "critical"]
    open_breaches = [b for b in breach_details if b["status"] == "open"]
    
    if critical_breaches:
        actions.append("Immediate review of critical compliance breaches required")
    
    if open_breaches:
        actions.append(f"Resolve {len(open_breaches)} open compliance breaches")
    
    if analytics["trends"]["trend"] == "worsening":
        actions.append("Investigate underlying causes of increasing breach trend")
    
    if analytics["breach_analysis"]["resolution_rate"] < 80:
        actions.append("Improve breach resolution processes and timelines")
    
    if not actions:
        actions.append("Continue monitoring current compliance posture")
    
    return actions


def _generate_text_report(report_data: Dict) -> str:
    """Generate text format regulatory report"""
    metadata = report_data["report_metadata"]
    summary = report_data["executive_summary"]
    
    text_report = f"""
REGULATORY COMPLIANCE REPORT
{metadata['report_id']}

Report Period: {metadata['reporting_period']['start_date']} to {metadata['reporting_period']['end_date']}
Generated: {metadata['generated_at']}
Generated By: {metadata['generated_by']}

EXECUTIVE SUMMARY
================
Overall Compliance Rate: {summary['overall_compliance_rate']}%
Total Active Rules: {summary['total_active_rules']}
Total Breaches (Period): {summary['total_breaches']}
Open Breaches: {summary['open_breaches']}
Critical Findings: {summary['critical_findings']}

Recommendation: {summary['recommendation']}

REGULATORY IMPLICATIONS
======================
Notification Required: {report_data['regulatory_implications']['notification_required']}
Next Report Due: {report_data['regulatory_implications']['next_report_due']}

Follow-up Actions:
"""
    
    for action in report_data['regulatory_implications']['follow_up_actions']:
        text_report += f"- {action}\n"
    
    text_report += f"\n\nGenerated by Compliance System v2.0\nReport ID: {metadata['report_id']}\n"
    
    return text_report


@router.post("/custom-report")
async def generate_custom_report(
    report_config: Dict[str, Any] = Body(..., description="Custom report configuration"),
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate custom compliance report based on user configuration
    """
    try:
        # This is a placeholder for custom report generation
        # In production, this would parse the report_config and generate
        # reports based on user-specified parameters
        
        return {
            "message": "Custom report generation not fully implemented",
            "config_received": report_config,
            "suggested_implementation": {
                "report_types": ["breach_summary", "rule_effectiveness", "portfolio_risk"],
                "date_ranges": ["7d", "30d", "90d", "1y"],
                "export_formats": ["json", "csv", "pdf"],
                "filters": ["severity", "control_type", "business_unit"]
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating custom report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Custom report generation failed: {str(e)}"
        )