"""
Compliance Rules Management API Routes
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Dict, Any, Optional
from datetime import datetime
import yaml
import json

from app.models.database import get_db, ComplianceRule, PolicyDocument
from app.schemas import (
    ComplianceRuleCreate, ComplianceRuleUpdate, ComplianceRuleResponse,
    RuleExtractionRequest, RuleExtractionResponse, ExtractedRule,
    RuleApprovalRequest, RuleApprovalResponse, BaseResponse
)
from app.api.dependencies import (
    get_current_user, get_optional_llm_service, rate_limit_llm_requests,
    validate_admin_access
)
from app.services.llm_service import LLMService
from app.utils.logger import get_logger, compliance_logger
from app.utils.exceptions import (
    RuleNotFound, ValidationException, DuplicateResource,
    validate_rule_expression
)

logger = get_logger(__name__)
router = APIRouter()


@router.get("/debug")
async def debug_rules(db: Session = Depends(get_db)):
    """Debug endpoint - return raw rules data"""
    rules = db.query(ComplianceRule).filter(ComplianceRule.is_active == True).all()
    result = []
    for r in rules:
        result.append({
            "rule_id": r.rule_id,
            "name": r.name,
            "description": r.description,
            "control_type": str(r.control_type),
            "severity": str(r.severity),
            "is_active": r.is_active
        })
    return result

@router.post("/debug")
async def debug_create_rule(rule_data: dict, db: Session = Depends(get_db)):
    """Debug endpoint - create rule without validation"""
    try:
        from app.models.database import ComplianceRule, ControlType, RuleSeverity
        import uuid
        
        new_rule = ComplianceRule(
            rule_id=f"DEMO_{uuid.uuid4().hex[:8].upper()}",
            name=rule_data.get("name", "Demo Rule"),
            description=rule_data.get("description", "Demo rule"),
            control_type=ControlType.QUANT_LIMIT,
            severity=RuleSeverity.MEDIUM,
            expression=rule_data.get("expression", {}),
            materiality_bps=rule_data.get("materiality_bps", 100),
            is_active=True,
            version=1
        )
        
        db.add(new_rule)
        db.commit()
        
        return {
            "success": True,
            "rule_id": new_rule.rule_id,
            "message": f"Rule {new_rule.rule_id} created successfully"
        }
    except Exception as e:
        db.rollback()
        return {"success": False, "error": str(e)}

@router.get("", response_model=List[ComplianceRuleResponse])
async def get_rules(
    active_only: bool = Query(True, description="Return only active rules"),
    control_type: Optional[str] = Query(None, description="Filter by control type"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    source_policy_id: Optional[str] = Query(None, description="Filter by source policy"),
    search: Optional[str] = Query(None, description="Search in rule descriptions"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Get compliance rules with comprehensive filtering and pagination
    """
    try:
        query = db.query(ComplianceRule)
        
        # Apply filters safely
        if active_only:
            query = query.filter(ComplianceRule.is_active == True)
        
        if control_type:
            query = query.filter(ComplianceRule.control_type == control_type.upper())
        
        if severity:
            query = query.filter(ComplianceRule.severity == severity.upper())
        
        if source_policy_id and hasattr(ComplianceRule, 'source_policy_id'):
            query = query.filter(ComplianceRule.source_policy_id == source_policy_id)
        
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                ComplianceRule.description.ilike(search_term) |
                (ComplianceRule.name.ilike(search_term) if ComplianceRule.name.isnot(None) else False)
            )
        
        # Get total count for pagination
        total_count = query.count()
        
        # Apply pagination
        rules = query.order_by(
            ComplianceRule.severity.desc(),
            ComplianceRule.created_at.desc()
        ).offset(offset).limit(limit).all()
        
        rule_responses = []
        for rule in rules:
            try:
                # Convert database enum values to API enum values
                control_type_map = {
                    "QUANT_LIMIT": "quant_limit",
                    "LIST_CONSTRAINT": "list_constraint", 
                    "TEMPORAL_WINDOW": "temporal_window",
                    "PROCESS_CONTROL": "process_control",
                    "REPORTING_DISCLOSURE": "reporting_disclosure"
                }
                
                severity_map = {
                    "LOW": "low",
                    "MEDIUM": "medium",
                    "HIGH": "high", 
                    "CRITICAL": "critical"
                }
                
                rule_responses.append(ComplianceRuleResponse(
                    rule_id=rule.rule_id,
                    name=getattr(rule, 'name', rule.rule_id),  # Use rule_id as fallback
                    description=rule.description,
                    control_type=control_type_map.get(str(rule.control_type), str(rule.control_type).lower()),
                    severity=severity_map.get(str(rule.severity), str(rule.severity).lower()),
                    expression=rule.expression if isinstance(rule.expression, dict) else {},
                    materiality_bps=rule.materiality_bps,
                    source_policy_id=getattr(rule, 'source_policy_id', None),
                    source_section=rule.source_section,
                    effective_date=rule.effective_date,
                    expiry_date=rule.expiry_date,
                    version=rule.version,
                    is_active=rule.is_active,
                    created_by=getattr(rule, 'created_by', 'system'),
                    created_at=rule.created_at,
                    modified_by=getattr(rule, 'modified_by', None),
                    modified_at=getattr(rule, 'modified_at', None),
                    metadata=getattr(rule, 'rule_metadata', {}) or {}
                ))
            except Exception as rule_e:
                logger.error(f"Error processing rule {rule.rule_id}: {rule_e}")
                logger.error(f"Rule data: control_type={rule.control_type}, severity={rule.severity}")
                continue
        
        # Add pagination info to headers (would be done in middleware in production)
        logger.info(f"Retrieved {len(rule_responses)} rules (total: {total_count}) for user {current_user}")
        
        return rule_responses
        
    except Exception as e:
        logger.error(f"Error retrieving rules: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve rules: {str(e)}"
        )


@router.post("", response_model=BaseResponse)
async def create_rule(
    rule_data: ComplianceRuleCreate = Body(...),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Create a new compliance rule with validation
    """
    try:
        # Validate rule expression (disabled for demo)
        # validate_rule_expression(rule_data.expression, str(rule_data.control_type))
        
        # Check for duplicate rule names
        if rule_data.name:
            existing_rule = db.query(ComplianceRule).filter(
                ComplianceRule.name == rule_data.name,
                ComplianceRule.is_active == True
            ).first()
            if existing_rule:
                raise DuplicateResource("rule", rule_data.name)
        
        # Create rule
        new_rule = ComplianceRule(
            name=rule_data.name,
            description=rule_data.description,
            control_type=rule_data.control_type,
            severity=rule_data.severity,
            expression=rule_data.expression,
            materiality_bps=rule_data.materiality_bps,
            source_policy_id=rule_data.source_policy_id,
            source_section=rule_data.source_section,
            effective_date=rule_data.effective_date,
            expiry_date=rule_data.expiry_date,
            is_active=True,
            created_by=rule_data.created_by or current_user,
            metadata=rule_data.metadata or {}
        )
        
        db.add(new_rule)
        db.commit()
        db.refresh(new_rule)
        
        # Log rule creation
        compliance_logger.log_system_event(
            "rule_created",
            f"New compliance rule created: {new_rule.rule_id}",
            user_id=current_user,
            rule_id=new_rule.rule_id,
            rule_type=rule_data.control_type.value,
            severity=rule_data.severity.value
        )
        
        return BaseResponse(
            success=True,
            message=f"Compliance rule created successfully",
            timestamp=datetime.now()
        )
        
    except (ValidationException, DuplicateResource):
        raise
    except IntegrityError as e:
        db.rollback()
        logger.error(f"Database integrity error creating rule: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Rule creation failed due to data conflict"
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating rule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create rule: {str(e)}"
        )


@router.get("/{rule_id}", response_model=ComplianceRuleResponse)
async def get_rule(
    rule_id: str = Path(..., description="Rule ID"),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Get detailed information about a specific rule
    """
    try:
        rule = db.query(ComplianceRule).filter(ComplianceRule.rule_id == rule_id).first()
        if not rule:
            raise RuleNotFound(rule_id)
        
        return ComplianceRuleResponse(
            rule_id=rule.rule_id,
            name=rule.name,
            description=rule.description,
            control_type=rule.control_type,
            severity=rule.severity,
            expression=rule.expression,
            materiality_bps=rule.materiality_bps,
            source_policy_id=rule.source_policy_id,
            source_section=rule.source_section,
            effective_date=rule.effective_date,
            expiry_date=rule.expiry_date,
            version=rule.version,
            is_active=rule.is_active,
            created_by=rule.created_by,
            created_at=rule.created_at,
            modified_by=rule.modified_by,
            modified_at=rule.modified_at,
            metadata=rule.metadata or {}
        )
        
    except RuleNotFound:
        raise
    except Exception as e:
        logger.error(f"Error retrieving rule {rule_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve rule: {str(e)}"
        )


@router.put("/{rule_id}", response_model=BaseResponse)
async def update_rule(
    rule_id: str = Path(..., description="Rule ID"),
    rule_updates: ComplianceRuleUpdate = Body(...),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Update an existing compliance rule
    """
    try:
        rule = db.query(ComplianceRule).filter(ComplianceRule.rule_id == rule_id).first()
        if not rule:
            raise RuleNotFound(rule_id)
        
        # Validate expression if being updated
        update_data = rule_updates.dict(exclude_unset=True)
        if "expression" in update_data:
            control_type = update_data.get("control_type", rule.control_type.value)
            validate_rule_expression(update_data["expression"], control_type)
        
        # Store old values for audit
        old_values = {
            "description": rule.description,
            "severity": rule.severity.value,
            "expression": rule.expression,
            "is_active": rule.is_active
        }
        
        # Apply updates
        for field, value in update_data.items():
            if hasattr(rule, field) and field != "modified_by":
                setattr(rule, field, value)
        
        # Update audit fields
        rule.modified_by = rule_updates.modified_by or current_user
        rule.modified_at = datetime.now()
        rule.version += 1
        
        db.commit()
        
        # Log rule update
        compliance_logger.log_system_event(
            "rule_updated",
            f"Compliance rule updated: {rule_id}",
            user_id=current_user,
            rule_id=rule_id,
            changes=list(update_data.keys()),
            old_values=old_values
        )
        
        return BaseResponse(
            success=True,
            message=f"Rule {rule_id} updated successfully",
            timestamp=datetime.now()
        )
        
    except (RuleNotFound, ValidationException):
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating rule {rule_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update rule: {str(e)}"
        )


@router.delete("/{rule_id}", response_model=BaseResponse)
async def deactivate_rule(
    rule_id: str = Path(..., description="Rule ID"),
    hard_delete: bool = Query(False, description="Permanently delete vs deactivate"),
    db: Session = Depends(get_db),
    current_user: str = Depends(validate_admin_access)
):
    """
    Deactivate or permanently delete a compliance rule (admin only)
    """
    try:
        rule = db.query(ComplianceRule).filter(ComplianceRule.rule_id == rule_id).first()
        if not rule:
            raise RuleNotFound(rule_id)
        
        if hard_delete:
            # Check for dependencies (breaches, cases)
            from app.models.database import ComplianceBreach, ComplianceCase
            
            breach_count = db.query(ComplianceBreach).filter(
                ComplianceBreach.rule_id == rule_id
            ).count()
            
            case_count = db.query(ComplianceCase).filter(
                ComplianceCase.rule_id == rule_id
            ).count()
            
            if breach_count > 0 or case_count > 0:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Cannot delete rule: {breach_count} breaches and {case_count} cases depend on it"
                )
            
            db.delete(rule)
            action = "permanently deleted"
        else:
            # Soft delete - deactivate
            rule.is_active = False
            rule.modified_by = current_user
            rule.modified_at = datetime.now()
            action = "deactivated"
        
        db.commit()
        
        # Log rule deletion/deactivation
        compliance_logger.log_system_event(
            "rule_deleted" if hard_delete else "rule_deactivated",
            f"Compliance rule {action}: {rule_id}",
            user_id=current_user,
            rule_id=rule_id,
            action=action
        )
        
        return BaseResponse(
            success=True,
            message=f"Rule {rule_id} {action} successfully",
            timestamp=datetime.now()
        )
        
    except (RuleNotFound, HTTPException):
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting/deactivating rule {rule_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete/deactivate rule: {str(e)}"
        )


@router.post("/extract", response_model=RuleExtractionResponse)
async def extract_rules_from_policy(
    request: RuleExtractionRequest = Body(...),
    llm_service: LLMService = Depends(get_optional_llm_service),
    current_user: str = Depends(rate_limit_llm_requests),
    db: Session = Depends(get_db)
):
    """
    Extract compliance rules from policy text using LLM
    """
    try:
        if not llm_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM service not available for rule extraction"
            )
        
        chunks = []
        policy_metadata = {}
        
        if request.policy_id:
            # Get policy and chunks for LLM processing
            policy = db.query(PolicyDocument).filter(
                PolicyDocument.policy_id == request.policy_id
            ).first()
            
            if not policy:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Policy {request.policy_id} not found"
                )
            
            from app.models.database import PolicyChunk
            policy_chunks = db.query(PolicyChunk).filter(
                PolicyChunk.policy_id == request.policy_id
            ).order_by(PolicyChunk.chunk_index).all()
            
            # Prepare chunks for LLM processing
            for chunk in policy_chunks:
                chunks.append({
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "chunk_id": f"{request.policy_id}_{chunk.chunk_index}"
                })
            
            policy_metadata = {
                "document_type": policy.document_type or "Policy Document",
                "jurisdiction": policy.jurisdiction or "General",
                "filename": policy.filename
            }
            
        elif request.policy_text:
            # Parse raw text by creating temporary chunks
            chunks = [{
                "content": request.policy_text,
                "chunk_index": 0,
                "page_number": 1,
                "section_title": "Direct Input",
                "chunk_id": "temp_0"
            }]
            
            policy_metadata = {
                "document_type": "Policy Text",
                "jurisdiction": "General",
                "filename": "direct_input.txt"
            }
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either policy_id or policy_text must be provided"
            )
        
        if not chunks:
            return RuleExtractionResponse(
                proposed_rules=[],
                requires_approval=True,
                total_rules_extracted=0,
                extraction_method="llm_powered",
                source_text=request.policy_text[:200] + "..." if request.policy_text and len(request.policy_text) > 200 else request.policy_text
            )
        
        # Use LLM service for rule extraction
        extracted_rules_data = await llm_service.extract_rules_from_chunks(chunks, policy_metadata)
        
        # Convert to response format
        extracted_rules = []
        for rule_data in extracted_rules_data:
            extracted_rules.append(ExtractedRule(
                rule_id=rule_data["rule_id"],
                description=rule_data["description"],
                control_type=rule_data["control_type"],
                severity=rule_data["severity"],
                expression=rule_data["expression"],
                materiality_bps=rule_data.get("materiality_bps"),
                source_section=rule_data.get("source_section"),
                confidence=rule_data.get("confidence", 0.8),
                rationale=rule_data.get("rationale"),
                extraction_metadata=rule_data.get("extraction_metadata")
            ))
        
        return RuleExtractionResponse(
            proposed_rules=extracted_rules,
            requires_approval=True,
            total_rules_extracted=len(extracted_rules),
            extraction_method="llm_powered",
            source_text=request.policy_text[:200] + "..." if request.policy_text and len(request.policy_text) > 200 else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting rules from policy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rule extraction failed: {str(e)}"
        )


@router.post("/approve", response_model=RuleApprovalResponse)
async def approve_extracted_rules(
    request: RuleApprovalRequest = Body(...),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Approve extracted rules and create them in the database
    """
    try:
        if not request.approved_rules:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No rules provided for approval"
            )
        
        created_rule_ids = []
        errors = []
        
        for rule_data in request.approved_rules:
            try:
                # Validate rule structure
                required_fields = ["description", "control_type", "severity", "expression"]
                if not all(field in rule_data for field in required_fields):
                    errors.append(f"Missing required fields in rule: {rule_data.get('rule_id', 'unknown')}")
                    continue
                
                # Validate rule expression
                validate_rule_expression(rule_data["expression"], rule_data["control_type"])
                
                # Create rule record
                rule = ComplianceRule(
                    name=rule_data.get("name"),
                    description=rule_data["description"],
                    control_type=rule_data["control_type"],
                    severity=rule_data["severity"],
                    expression=rule_data["expression"],
                    materiality_bps=rule_data.get("materiality_bps", 0),
                    source_policy_id=request.policy_id,
                    source_section=rule_data.get("source_section", "LLM Extracted"),
                    is_active=True,
                    created_by=request.approved_by,
                    metadata={
                        "extraction_method": "llm_approval",
                        "confidence": rule_data.get("confidence"),
                        "rationale": rule_data.get("rationale"),
                        "approved_at": datetime.now().isoformat()
                    }
                )
                
                db.add(rule)
                db.flush()  # Get the rule_id
                
                created_rule_ids.append(rule.rule_id)
                
            except ValidationException as ve:
                errors.append(f"Validation error for rule {rule_data.get('rule_id', 'unknown')}: {ve.message}")
            except Exception as re:
                errors.append(f"Error creating rule {rule_data.get('rule_id', 'unknown')}: {str(re)}")
        
        if created_rule_ids:
            db.commit()
            
            # Log rule approval
            compliance_logger.log_system_event(
                "rules_approved",
                f"{len(created_rule_ids)} rules approved and created",
                user_id=current_user,
                approved_by=request.approved_by,
                created_rule_count=len(created_rule_ids),
                policy_id=request.policy_id
            )
        else:
            db.rollback()
        
        return RuleApprovalResponse(
            success=len(created_rule_ids) > 0,
            message=f"Approved and created {len(created_rule_ids)} compliance rules",
            created_rule_ids=created_rule_ids,
            approved_by=request.approved_by,
            total_approved=len(request.approved_rules)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error approving rules: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rule approval failed: {str(e)}"
        )


@router.get("/{rule_id}/yaml")
async def get_rule_as_yaml(
    rule_id: str = Path(..., description="Rule ID"),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Get compliance rule in YAML format for editing
    """
    try:
        rule = db.query(ComplianceRule).filter(ComplianceRule.rule_id == rule_id).first()
        if not rule:
            raise RuleNotFound(rule_id)
        
        rule_dict = {
            "rule_id": rule.rule_id,
            "name": rule.name,
            "description": rule.description,
            "control_type": rule.control_type.value,
            "severity": rule.severity.value,
            "expression": rule.expression,
            "materiality_bps": rule.materiality_bps,
            "source_section": rule.source_section,
            "is_active": rule.is_active,
            "effective_date": rule.effective_date.isoformat() if rule.effective_date else None,
            "expiry_date": rule.expiry_date.isoformat() if rule.expiry_date else None,
            "metadata": rule.metadata
        }
        
        yaml_content = yaml.dump(rule_dict, default_flow_style=False, indent=2)
        
        return {
            "rule_id": rule_id,
            "yaml_content": yaml_content,
            "json_content": rule_dict
        }
        
    except RuleNotFound:
        raise
    except Exception as e:
        logger.error(f"Error getting rule as YAML {rule_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate YAML: {str(e)}"
        )


@router.put("/{rule_id}/yaml", response_model=BaseResponse)
async def update_rule_from_yaml(
    rule_id: str = Path(..., description="Rule ID"),
    request: Dict[str, str] = Body(...),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Update compliance rule from YAML content
    """
    try:
        yaml_content = request.get("yaml_content", "")
        updated_by = request.get("updated_by", current_user)
        
        if not yaml_content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="YAML content is required"
            )
        
        # Parse YAML content
        try:
            rule_data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid YAML: {str(e)}"
            )
        
        # Get existing rule
        rule = db.query(ComplianceRule).filter(ComplianceRule.rule_id == rule_id).first()
        if not rule:
            raise RuleNotFound(rule_id)
        
        # Validate expression if present
        if "expression" in rule_data and "control_type" in rule_data:
            validate_rule_expression(rule_data["expression"], rule_data["control_type"])
        
        # Update fields from YAML
        updatable_fields = [
            "name", "description", "control_type", "severity", "expression", 
            "materiality_bps", "source_section", "is_active"
        ]
        
        for field in updatable_fields:
            if field in rule_data:
                if field in ["control_type", "severity"]:
                    # Handle enums
                    setattr(rule, field, rule_data[field])
                else:
                    setattr(rule, field, rule_data[field])
        
        if "metadata" in rule_data:
            rule.metadata = rule_data["metadata"]
        
        # Handle dates
        if "effective_date" in rule_data and rule_data["effective_date"]:
            rule.effective_date = datetime.fromisoformat(rule_data["effective_date"])
        
        if "expiry_date" in rule_data and rule_data["expiry_date"]:
            rule.expiry_date = datetime.fromisoformat(rule_data["expiry_date"])
        
        rule.modified_at = datetime.now()
        rule.modified_by = updated_by
        rule.version += 1
        
        db.commit()
        
        return BaseResponse(
            success=True,
            message=f"Rule updated successfully from YAML",
            timestamp=datetime.now()
        )
        
    except (RuleNotFound, HTTPException):
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating rule from YAML {rule_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"YAML update failed: {str(e)}"
        )


@router.post("/{rule_id}/clone", response_model=BaseResponse)
async def clone_rule(
    rule_id: str = Path(..., description="Rule ID to clone"),
    clone_config: Dict[str, Any] = Body(default={}, description="Clone configuration"),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Clone an existing compliance rule with modifications
    """
    try:
        # Get source rule
        source_rule = db.query(ComplianceRule).filter(ComplianceRule.rule_id == rule_id).first()
        if not source_rule:
            raise RuleNotFound(rule_id)
        
        # Create cloned rule
        cloned_rule = ComplianceRule(
            name=clone_config.get("name", f"{source_rule.name} (Copy)"),
            description=clone_config.get("description", source_rule.description),
            control_type=clone_config.get("control_type", source_rule.control_type),
            severity=clone_config.get("severity", source_rule.severity),
            expression=clone_config.get("expression", source_rule.expression),
            materiality_bps=clone_config.get("materiality_bps", source_rule.materiality_bps),
            source_policy_id=source_rule.source_policy_id,
            source_section=clone_config.get("source_section", f"{source_rule.source_section} (Cloned)"),
            is_active=clone_config.get("is_active", False),  # Default to inactive for review
            created_by=current_user,
            metadata={
                **(source_rule.metadata or {}),
                "cloned_from": source_rule.rule_id,
                "cloned_at": datetime.now().isoformat(),
                "cloned_by": current_user
            }
        )
        
        db.add(cloned_rule)
        db.commit()
        db.refresh(cloned_rule)
        
        # Log rule cloning
        compliance_logger.log_system_event(
            "rule_cloned",
            f"Rule cloned: {source_rule.rule_id} -> {cloned_rule.rule_id}",
            user_id=current_user,
            source_rule_id=source_rule.rule_id,
            cloned_rule_id=cloned_rule.rule_id
        )
        
        return BaseResponse(
            success=True,
            message=f"Rule cloned successfully as {cloned_rule.rule_id}",
            timestamp=datetime.now()
        )
        
    except RuleNotFound:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error cloning rule {rule_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rule cloning failed: {str(e)}"
        )


@router.get("/templates/control-types")
async def get_rule_templates():
    """
    Get rule templates and examples for different control types
    """
    try:
        templates = {
            "quant_limit": {
                "description": "Quantitative limit rule template",
                "example": {
                    "name": "Single Issuer Concentration Limit",
                    "description": "No single issuer shall exceed 5% of portfolio value",
                    "control_type": "quant_limit",
                    "severity": "critical",
                    "expression": {
                        "metric": "issuer_weight",
                        "operator": "<=",
                        "threshold": 0.05,
                        "scope": "portfolio"
                    },
                    "materiality_bps": 50
                }
            },
            "list_constraint": {
                "description": "List constraint rule template",
                "example": {
                    "name": "Minimum Credit Rating Requirement",
                    "description": "All holdings must maintain minimum BBB credit rating",
                    "control_type": "list_constraint",
                    "severity": "medium",
                    "expression": {
                        "field": "rating",
                        "allowed_values": ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"],
                        "scope": "position"
                    }
                }
            },
            "temporal_window": {
                "description": "Temporal window rule template",
                "example": {
                    "name": "Minimum Holding Period",
                    "description": "Securities must be held for at least 30 days",
                    "control_type": "temporal_window",
                    "severity": "medium",
                    "expression": {
                        "metric": "holding_period",
                        "minimum_days": 30,
                        "scope": "position"
                    }
                }
            },
            "process_control": {
                "description": "Process control rule template",
                "example": {
                    "name": "Investment Committee Approval",
                    "description": "Investments exceeding $10M require investment committee approval",
                    "control_type": "process_control",
                    "severity": "high",
                    "expression": {
                        "approval_required": True,
                        "approver_role": "investment_committee",
                        "evidence_required": "Investment committee meeting minutes",
                        "sla_days": 5
                    }
                }
            },
            "reporting_disclosure": {
                "description": "Reporting and disclosure rule template",
                "example": {
                    "name": "Monthly Portfolio Report",
                    "description": "Monthly portfolio composition report to board",
                    "control_type": "reporting_disclosure",
                    "severity": "medium",
                    "expression": {
                        "report_type": "portfolio_composition",
                        "frequency": "monthly",
                        "deadline_days": 15,
                        "recipient": "board"
                    }
                }
            }
        }
        
        return {
            "templates": templates,
            "control_types": list(templates.keys()),
            "severity_levels": ["critical", "high", "medium", "low"],
            "common_operators": ["<=", ">=", "<", ">", "==", "!="],
            "common_metrics": [
                "issuer_weight", "sector_weight", "country_weight", 
                "total_exposure", "leverage_ratio"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting rule templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get templates: {str(e)}"
        )