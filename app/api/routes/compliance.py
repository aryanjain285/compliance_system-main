
"""
Compliance Monitoring and Evaluation API Routes
"""
import uuid
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from app.models.database import get_db, ComplianceBreach, ComplianceCase, ComplianceRule
from app.schemas import (
    ComplianceCheckResponse, ComplianceBreachResponse, BreachResolutionRequest,
    BreachExplanationResponse, ComplianceCaseResponse, BaseResponse
)
from app.api.dependencies import (
    get_compliance_engine, get_current_user, rate_limit_compliance_check,
    get_optional_llm_service, get_optional_vector_service
)
from app.services.compliance_engine import ComplianceEngine
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStoreService
from app.utils.logger import get_logger, compliance_logger
from app.utils.exceptions import (
    BreachNotFound, RuleNotFound, ValidationException
)

logger = get_logger(__name__)
router = APIRouter()


@router.post("/check", response_model=ComplianceCheckResponse)
async def run_compliance_check(
    rule_ids: Optional[List[str]] = Query(None, description="Specific rule IDs to evaluate"),
    skip_cache: bool = Query(False, description="Skip cache and reload data"),
    engine: ComplianceEngine = Depends(get_compliance_engine),
    current_user: str = Depends(rate_limit_compliance_check)
):
    """
    Run comprehensive compliance check against all active rules
    """
    try:
        logger.info(f"Running compliance check requested by {current_user}")
        
        if rule_ids:
            # Evaluate specific rules only
            results = {
                "evaluation_id": str(uuid.uuid4()),
                "total_rules_checked": 0,
                "compliant_rules": 0,
                "breached_rules": 0,
                "case_created_rules": 0,
                "error_rules": 0,
                "skipped_rules": 0,
                "compliance_rate": 100.0,
                "new_breaches": [],
                "new_cases": [],
                "errors": [],
                "portfolio_summary": {},
                "performance": {},
                "system_health": {},
                "results": []
            }
            
            # Load caches
            engine.portfolio_cache = engine._load_portfolio_cache()
            engine.rule_cache = engine._load_rule_cache()
            
            for rule_id in rule_ids:
                if rule_id in engine.rule_cache:
                    rule_data = engine.rule_cache[rule_id]
                    result = engine._evaluate_single_rule(rule_data["rule"])
                    results["results"].append(result)
                    
                    # Update counters
                    results["total_rules_checked"] += 1
                    if result["status"] == "compliant":
                        results["compliant_rules"] += 1
                    elif result["status"] == "breach":
                        results["breached_rules"] += 1
                    elif result["status"] == "case_created":
                        results["case_created_rules"] += 1
                    else:
                        results["error_rules"] += 1
            
            # Calculate compliance rate
            total_evaluated = results["compliant_rules"] + results["breached_rules"]
            if total_evaluated > 0:
                results["compliance_rate"] = round(
                    (results["compliant_rules"] / total_evaluated) * 100, 2
                )
        else:
            # Run full compliance evaluation
            results = engine.evaluate_all_rules()
        
        # Log the evaluation
        compliance_logger.log_system_event(
            "compliance_check",
            f"Compliance check completed with {results['compliance_rate']}% compliance rate",
            user_id=current_user,
            evaluation_id=results["evaluation_id"],
            total_rules=results["total_rules_checked"],
            breached_rules=results["breached_rules"]
        )
        
        return ComplianceCheckResponse(**results)
        
    except Exception as e:
        logger.error(f"Error in compliance check: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Compliance check failed: {str(e)}"
        )


@router.get("/rules/{rule_id}/evaluate")
async def evaluate_single_rule(
    rule_id: str = Path(..., description="Rule ID to evaluate"),
    engine: ComplianceEngine = Depends(get_compliance_engine),
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Evaluate a single compliance rule
    """
    try:
        rule = db.query(ComplianceRule).filter(ComplianceRule.rule_id == rule_id).first()
        if not rule:
            raise RuleNotFound(rule_id)
        
        if not rule.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Rule {rule_id} is not active"
            )
        
        # Load portfolio cache
        engine.portfolio_cache = engine._load_portfolio_cache()
        
        # Evaluate the rule
        result = engine._evaluate_single_rule(rule)
        
        return {
            "rule_id": rule_id,
            "evaluation_timestamp": datetime.now().isoformat(),
            "result": result,
            "portfolio_summary": {
                "positions_count": len(engine.portfolio_cache),
                "total_market_value": sum(
                    pos["market_value"] for pos in engine.portfolio_cache.values()
                )
            }
        }
        
    except (RuleNotFound, HTTPException):
        raise
    except Exception as e:
        logger.error(f"Error evaluating rule {rule_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rule evaluation failed: {str(e)}"
        )


@router.get("/breaches", response_model=List[ComplianceBreachResponse])
async def get_breaches(
    status_filter: Optional[str] = Query("open", description="Filter by breach status"),
    severity_filter: Optional[str] = Query(None, description="Filter by rule severity"),
    rule_id_filter: Optional[str] = Query(None, description="Filter by specific rule ID"),
    days_back: Optional[int] = Query(None, ge=1, le=365, description="Days back to search"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    engine: ComplianceEngine = Depends(get_compliance_engine),
    current_user: str = Depends(get_current_user)
):
    """
    Get compliance breaches with comprehensive filtering
    """
    try:
        if status_filter == "open":
            breaches = engine.get_active_breaches(severity_filter)
        else:
            # Get all breaches with custom filtering
            db = engine.db
            query = db.query(ComplianceBreach).join(
                ComplianceRule, ComplianceBreach.rule_id == ComplianceRule.rule_id
            )
            
            # Apply filters with safe status conversion
            if status_filter and status_filter != "all":
                # Convert API status to database status
                db_status = status_filter.upper() if status_filter.lower() == "open" else status_filter
                query = query.filter(ComplianceBreach.status == db_status)
            
            if severity_filter:
                query = query.filter(ComplianceRule.severity == severity_filter)
            
            if rule_id_filter:
                query = query.filter(ComplianceBreach.rule_id == rule_id_filter)
            
            if days_back:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                query = query.filter(ComplianceBreach.breach_timestamp >= cutoff_date)
            
            breaches = query.order_by(
                ComplianceBreach.breach_timestamp.desc()
            ).limit(limit).all()
            
            # Convert to response format with safe field access
            breach_responses = []
            for breach in breaches:
                try:
                    age_hours = (datetime.now() - breach.breach_timestamp).total_seconds() / 3600
                    
                    # Safely parse impact_assessment
                    impact_assessment = breach.impact_assessment
                    if isinstance(impact_assessment, str):
                        try:
                            import json
                            impact_assessment = json.loads(impact_assessment)
                        except:
                            impact_assessment = {}
                    
                    breach_responses.append(ComplianceBreachResponse(
                        breach_id=breach.breach_id,
                        rule_id=breach.rule_id,
                        rule_name=getattr(breach.rule, 'name', None),
                        rule_description=breach.rule.description,
                        control_type=breach.rule.control_type.value,
                        severity=breach.rule.severity.value.lower(),  # Convert to lowercase
                        observed_value=breach.observed_value,
                        threshold=breach.threshold_value,
                        breach_magnitude=breach.breach_magnitude,
                        breach_timestamp=breach.breach_timestamp,
                        status=breach.status.lower() if hasattr(breach.status, 'lower') else str(breach.status).lower(),
                        age_hours=round(age_hours, 1),
                        age_category="recent" if age_hours < 24 else "old",
                        sla_status="within_sla" if age_hours < 48 else "breached_sla",
                        impact_assessment=impact_assessment,
                        external_reference=breach.external_reference,
                        portfolio_snapshot_size=len(breach.portfolio_snapshot) if breach.portfolio_snapshot else 0
                    ))
                except Exception as breach_e:
                    logger.error(f"Error processing breach {breach.breach_id}: {breach_e}")
                    continue
            
            breaches = breach_responses
        
        return breaches
        
    except Exception as e:
        logger.error(f"Error getting breaches: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve breaches: {str(e)}"
        )


@router.get("/breaches/{breach_id}", response_model=Dict[str, Any])
async def get_breach_detail(
    breach_id: str = Path(..., description="Breach ID"),
    include_portfolio_snapshot: bool = Query(False, description="Include full portfolio snapshot"),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Get detailed information about a specific breach
    """
    try:
        breach = db.query(ComplianceBreach).join(
            ComplianceRule, ComplianceBreach.rule_id == ComplianceRule.rule_id
        ).filter(ComplianceBreach.breach_id == breach_id).first()
        
        if not breach:
            raise BreachNotFound(breach_id)
        
        # Calculate breach age
        age_hours = (datetime.now() - breach.breach_timestamp).total_seconds() / 3600
        
        breach_detail = {
            "breach_id": breach.breach_id,
            "rule_id": breach.rule_id,
            "rule_name": breach.rule.name,
            "rule_description": breach.rule.description,
            "rule_severity": breach.rule.severity.value,
            "control_type": breach.rule.control_type.value,
            "observed_value": breach.observed_value,
            "threshold_value": breach.threshold_value,
            "breach_magnitude": breach.breach_magnitude,
            "breach_timestamp": breach.breach_timestamp.isoformat(),
            "status": breach.status.value,
            "age_hours": round(age_hours, 1),
            "age_category": ComplianceEngine._categorize_breach_age(None, age_hours),
            "sla_status": ComplianceEngine._assess_breach_sla(None, age_hours, breach.rule.severity.value),
            "resolution_info": {
                "resolved_at": breach.resolved_at.isoformat() if breach.resolved_at else None,
                "resolved_by": breach.resolved_by,
                "resolution_notes": breach.resolution_notes,
                "severity_override": breach.severity_override.value if breach.severity_override else None
            },
            "impact_assessment": breach.impact_assessment or {},
            "remediation_plan": breach.remediation_plan,
            "external_reference": breach.external_reference,
            "metadata": breach.metadata or {}
        }
        
        # Include portfolio snapshot if requested
        if include_portfolio_snapshot and breach.portfolio_snapshot:
            breach_detail["portfolio_snapshot"] = breach.portfolio_snapshot
        else:
            breach_detail["portfolio_snapshot_summary"] = {
                "total_positions": len(breach.portfolio_snapshot) if breach.portfolio_snapshot else 0,
                "snapshot_hash": breach.portfolio_snapshot_hash
            }
        
        return breach_detail
        
    except BreachNotFound:
        raise
    except Exception as e:
        logger.error(f"Error getting breach detail {breach_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve breach details: {str(e)}"
        )


@router.post("/breaches/{breach_id}/resolve", response_model=BaseResponse)
async def resolve_breach(
    breach_id: str = Path(..., description="Breach ID"),
    resolution: BreachResolutionRequest = Body(...),
    engine: ComplianceEngine = Depends(get_compliance_engine),
    current_user: str = Depends(get_current_user)
):
    """
    Resolve a compliance breach
    """
    try:
        success = engine.resolve_breach(
            breach_id=breach_id,
            resolved_by=resolution.resolved_by or current_user,
            notes=resolution.notes,
            resolution_type=resolution.resolution_type
        )
        
        if not success:
            raise BreachNotFound(breach_id)
        
        # Log the resolution
        compliance_logger.log_breach_resolved(breach_id, resolution.resolved_by or current_user)
        
        return BaseResponse(
            success=True,
            message=f"Breach {breach_id} resolved successfully as {resolution.resolution_type}"
        )
        
    except BreachNotFound:
        raise
    except Exception as e:
        logger.error(f"Error resolving breach {breach_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resolve breach: {str(e)}"
        )


@router.post("/breaches/{breach_id}/explain", response_model=BreachExplanationResponse)
async def generate_breach_explanation(
    breach_id: str = Path(..., description="Breach ID"),
    llm_service: LLMService = Depends(get_optional_llm_service),
    vector_service: VectorStoreService = Depends(get_optional_vector_service),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Generate detailed breach explanation using LLM
    """
    try:
        # Get breach details
        breach = db.query(ComplianceBreach).join(
            ComplianceRule, ComplianceBreach.rule_id == ComplianceRule.rule_id
        ).filter(ComplianceBreach.breach_id == breach_id).first()
        
        if not breach:
            raise BreachNotFound(breach_id)
        
        # Prepare breach data
        breach_data = {
            "observed_value": breach.observed_value,
            "threshold_value": breach.threshold_value,
            "breach_magnitude": breach.breach_magnitude,
            "breach_timestamp": breach.breach_timestamp.isoformat(),
            "portfolio_snapshot": breach.portfolio_snapshot
        }
        
        rule_data = {
            "description": breach.rule.description,
            "control_type": breach.rule.control_type.value,
            "severity": breach.rule.severity.value,
            "expression": breach.rule.expression
        }
        
        # Get policy context if available
        policy_context = []
        if vector_service and vector_service.is_available() and breach.rule.source_policy_id:
            try:
                policy_context = vector_service.get_policy_context(
                    rule_description=breach.rule.description,
                    policy_id=breach.rule.source_policy_id,
                    n_results=3
                )
            except Exception as ctx_e:
                logger.warning(f"Failed to get policy context: {ctx_e}")
        
        # Generate explanation
        if llm_service:
            explanation = await llm_service.generate_breach_explanation(
                breach_data, rule_data, policy_context
            )
        else:
            # Fallback explanation
            explanation = f"""
COMPLIANCE BREACH EXPLANATION

Breach ID: {breach_id}
Rule: {rule_data['description']}
Severity: {rule_data['severity'].upper()}

Details:
- Observed Value: {breach_data.get('observed_value', 'N/A')}
- Threshold: {breach_data.get('threshold_value', 'N/A')}
- Breach Magnitude: {breach_data.get('breach_magnitude', 'N/A')}%
- Detection Time: {breach_data['breach_timestamp']}

This breach indicates that the portfolio position or exposure has exceeded 
the permitted limits as defined in the compliance framework. Immediate review 
and corrective action may be required to restore compliance.

Generated using fallback system due to LLM service unavailability.
""".strip()
        
        return BreachExplanationResponse(
            breach_id=breach_id,
            explanation=explanation,
            generated_at=datetime.now()
        )
        
    except BreachNotFound:
        raise
    except Exception as e:
        logger.error(f"Error generating breach explanation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate explanation: {str(e)}"
        )


@router.get("/cases")
async def get_compliance_cases(
    status_filter: Optional[str] = Query("open", description="Filter by case status"),
    case_type_filter: Optional[str] = Query(None, description="Filter by case type"),
    assigned_to_filter: Optional[str] = Query(None, description="Filter by assignee"),
    overdue_only: bool = Query(False, description="Show only overdue cases"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Get compliance cases with filtering
    """
    try:
        query = db.query(ComplianceCase).join(
            ComplianceRule, ComplianceCase.rule_id == ComplianceRule.rule_id
        )
        
        # Apply filters
        if status_filter and status_filter != "all":
            query = query.filter(ComplianceCase.status == status_filter)
        
        if case_type_filter:
            query = query.filter(ComplianceCase.case_type == case_type_filter)
        
        if assigned_to_filter:
            query = query.filter(ComplianceCase.assigned_to == assigned_to_filter)
        
        if overdue_only:
            query = query.filter(
                ComplianceCase.sla_deadline < datetime.now(),
                ComplianceCase.status.in_(["open", "in_review"])
            )
        
        cases = query.order_by(
            ComplianceCase.created_at.desc()
        ).limit(limit).all()
        
        case_responses = []
        for case in cases:
            case_responses.append(ComplianceCaseResponse(
                case_id=case.case_id,
                rule_id=case.rule_id,
                case_type=case.case_type.value,
                status=case.status,
                priority=case.priority,
                title=case.title,
                description=case.description,
                evidence_required=case.evidence_required,
                sla_deadline=case.sla_deadline,
                assigned_to=case.assigned_to,
                created_at=case.created_at,
                updated_at=case.updated_at
            ))
        
        return case_responses
        
    except Exception as e:
        logger.error(f"Error getting compliance cases: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve cases: {str(e)}"
        )


@router.get("/status")
async def get_compliance_status(
    engine: ComplianceEngine = Depends(get_compliance_engine),
    current_user: str = Depends(get_current_user)
):
    """
    Get overall compliance status and key metrics
    """
    try:
        # Get basic analytics safely
        try:
            analytics = engine.get_compliance_analytics(days=30)
        except Exception as analytics_e:
            logger.warning(f"Analytics failed: {analytics_e}")
            analytics = {
                "summary": {
                    "open_breaches": 0,
                    "compliance_rate": 100.0,
                    "period_breaches": 0
                },
                "breach_analysis": {"resolution_rate": 0},
                "trends": {"trend": "stable"},
                "performance": {},
                "generated_at": datetime.now()
            }
        
        # Get current breach counts by severity safely
        try:
            active_breaches = engine.get_active_breaches()
            breach_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            
            for breach in active_breaches:
                severity = breach.get("severity", "low").lower()
                if severity in breach_counts:
                    breach_counts[severity] += 1
        except Exception as breach_e:
            logger.warning(f"Breach counting failed: {breach_e}")
            breach_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        # Calculate risk score (simplified)
        risk_score = (
            breach_counts["critical"] * 10 +
            breach_counts["high"] * 5 +
            breach_counts["medium"] * 2 +
            breach_counts["low"] * 1
        )
        
        if risk_score == 0:
            risk_level = "low"
        elif risk_score <= 10:
            risk_level = "medium"
        elif risk_score <= 25:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        return {
            "overall_status": "compliant" if analytics["summary"]["open_breaches"] == 0 else "non_compliant",
            "compliance_rate": analytics["summary"]["compliance_rate"],
            "risk_level": risk_level,
            "risk_score": risk_score,
            "active_breaches": {
                "total": analytics["summary"]["open_breaches"],
                "by_severity": breach_counts
            },
            "recent_activity": {
                "period_breaches": analytics["summary"]["period_breaches"],
                "resolution_rate": analytics["breach_analysis"]["resolution_rate"],
                "trend": analytics["trends"]["trend"]
            },
            "performance": analytics["performance"],
            "last_evaluation": analytics["generated_at"],
            "recommendation": _get_status_recommendation(risk_level, analytics)
        }
        
    except Exception as e:
        logger.error(f"Error getting compliance status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get compliance status: {str(e)}"
        )


def _get_status_recommendation(risk_level: str, analytics: Dict[str, Any]) -> str:
    """Get status-based recommendation"""
    if risk_level == "critical":
        return "Immediate action required: Multiple critical breaches detected"
    elif risk_level == "high":
        return "High priority: Review and resolve outstanding breaches"
    elif risk_level == "medium":
        return "Monitor closely: Some compliance issues require attention"
    else:
        if analytics["summary"]["compliance_rate"] >= 99:
            return "Excellent compliance posture maintained"
        else:
            return "Good compliance status: Continue monitoring"


@router.post("/simulate")
async def simulate_compliance_scenario(
    scenario: Dict[str, Any] = Body(..., description="Scenario to simulate"),
    engine: ComplianceEngine = Depends(get_compliance_engine),
    current_user: str = Depends(get_current_user)
):
    """
    Simulate compliance scenarios (what-if analysis)
    """
    try:
        # This would implement scenario simulation
        # For now, return a placeholder response
        
        return {
            "simulation_id": f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "scenario": scenario,
            "results": {
                "message": "Compliance scenario simulation not fully implemented",
                "note": "This would simulate portfolio changes and predict compliance impacts"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in compliance simulation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {str(e)}"
        )