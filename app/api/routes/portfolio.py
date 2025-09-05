"""
Portfolio Management API Routes
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.models.database import get_db, Portfolio, PositionHistory
from app.schemas import (
    PositionCreate, PositionUpdate, PositionResponse, 
    PortfolioSummary, BaseResponse
)
from app.api.dependencies import get_current_user, get_compliance_engine
from app.services.compliance_engine import ComplianceEngine
from app.utils.logger import get_logger, compliance_logger
from app.utils.exceptions import (
    PositionNotFound, ValidationException, DuplicateResource
)

logger = get_logger(__name__)
router = APIRouter()


@router.get("", response_model=Dict[str, PositionResponse])
async def get_portfolio(
    include_inactive: bool = Query(False, description="Include inactive positions"),
    sector_filter: Optional[str] = Query(None, description="Filter by sector"),
    country_filter: Optional[str] = Query(None, description="Filter by country"),
    min_weight: Optional[float] = Query(None, ge=0, le=1, description="Minimum weight filter"),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Get current portfolio positions with optional filtering
    """
    try:
        query = db.query(Portfolio)
        
        # Apply filters - but check if fields exist first
        if not include_inactive and hasattr(Portfolio, 'weight'):
            query = query.filter(Portfolio.weight > 0)
        
        if sector_filter and hasattr(Portfolio, 'sector'):
            query = query.filter(Portfolio.sector == sector_filter)
        
        if country_filter and hasattr(Portfolio, 'country'):
            query = query.filter(Portfolio.country == country_filter)
        
        if min_weight is not None and hasattr(Portfolio, 'weight'):
            query = query.filter(Portfolio.weight >= min_weight)
        
        positions = query.all()
        
        # Convert to response format with safe field access
        portfolio = {}
        for position in positions:
            try:
                portfolio[position.symbol] = PositionResponse(
                    position_id=getattr(position, 'position_id', position.portfolio_id),
                    symbol=position.symbol,
                    name=getattr(position, 'name', None),
                    weight=getattr(position, 'weight', 0.0),
                    market_value=getattr(position, 'market_value', 0.0),
                    quantity=getattr(position, 'quantity', None),
                    price=getattr(position, 'price', None),
                    sector=getattr(position, 'sector', None),
                    industry=getattr(position, 'industry', None),
                    country=getattr(position, 'country', None),
                    currency=getattr(position, 'currency', "USD"),
                    rating=getattr(position, 'rating', None),
                    rating_agency=getattr(position, 'rating_agency', None),
                    instrument_type=getattr(position, 'instrument_type', None),
                    exchange=getattr(position, 'exchange', None),
                    maturity_date=getattr(position, 'maturity_date', None),
                    acquisition_date=getattr(position, 'acquisition_date', None),
                    bloomberg_id=getattr(position, 'bloomberg_id', None),
                    cusip=getattr(position, 'cusip', None),
                    isin=getattr(position, 'isin', None),
                    sedol=getattr(position, 'sedol', None),
                    metadata=getattr(position, 'metadata', {}) or {},
                    last_updated=getattr(position, 'last_updated', datetime.now()),
                    created_at=getattr(position, 'created_at', datetime.now())
                )
            except Exception as pos_e:
                logger.error(f"Error processing position {position.symbol}: {pos_e}")
                continue
        
        logger.info(f"Retrieved {len(portfolio)} portfolio positions for user {current_user}")
        return portfolio
        
    except Exception as e:
        logger.error(f"Error retrieving portfolio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve portfolio: {str(e)}"
        )


@router.post("/{symbol}", response_model=BaseResponse)
async def create_or_update_position(
    symbol: str = Path(..., description="Security symbol"),
    position_data: PositionCreate = Body(...),
    run_compliance_check: bool = Query(True, description="Run compliance check after update"),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Create new position or update existing position
    """
    try:
        # Check if position already exists
        existing_position = db.query(Portfolio).filter(Portfolio.symbol == symbol).first()
        
        if existing_position:
            # Update existing position
            old_values = {
                "weight": existing_position.weight,
                "market_value": existing_position.market_value,
                "quantity": existing_position.quantity,
                "price": existing_position.price,
                "sector": existing_position.sector,
                "country": existing_position.country,
                "rating": existing_position.rating
            }
            
            # Update fields
            for field, value in position_data.dict(exclude_unset=True).items():
                if hasattr(existing_position, field):
                    setattr(existing_position, field, value)
            
            existing_position.last_updated = datetime.now()
            
            # Record change in history
            history = PositionHistory(
                position_id=existing_position.position_id,
                symbol=symbol,
                weight=existing_position.weight,
                market_value=existing_position.market_value,
                change_type="update",
                changed_fields=list(position_data.dict(exclude_unset=True).keys()),
                previous_values=old_values,
                change_reason="API update",
                changed_by=current_user
            )
            db.add(history)
            
            action = "updated"
            
        else:
            # Create new position
            new_position = Portfolio(
                symbol=symbol,
                **position_data.dict(exclude_unset=True)
            )
            db.add(new_position)
            db.flush()  # Get the ID
            
            # Record creation in history
            history = PositionHistory(
                position_id=new_position.position_id,
                symbol=symbol,
                weight=new_position.weight,
                market_value=new_position.market_value,
                change_type="insert",
                changed_fields=list(position_data.dict(exclude_unset=True).keys()),
                change_reason="API creation",
                changed_by=current_user
            )
            db.add(history)
            
            action = "created"
        
        db.commit()
        
        # Log the change
        compliance_logger.log_portfolio_update(
            symbol=symbol,
            changes=position_data.dict(exclude_unset=True)
        )
        
        response_data = {
            "success": True,
            "message": f"Position {symbol} {action} successfully",
            "symbol": symbol,
            "action": action
        }
        
        # Run compliance check if requested
        if run_compliance_check:
            try:
                engine = ComplianceEngine(db)
                compliance_results = engine.evaluate_all_rules()
                response_data["compliance_check"] = {
                    "executed": True,
                    "compliance_rate": compliance_results.get("compliance_rate", 0),
                    "new_breaches": len(compliance_results.get("new_breaches", [])),
                    "evaluation_id": compliance_results.get("evaluation_id")
                }
            except Exception as comp_e:
                logger.warning(f"Compliance check failed after position update: {comp_e}")
                response_data["compliance_check"] = {
                    "executed": False,
                    "error": str(comp_e)
                }
        
        return response_data
        
    except IntegrityError as e:
        db.rollback()
        logger.error(f"Database integrity error creating/updating position {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Position {symbol} conflicts with existing data"
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating/updating position {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create/update position: {str(e)}"
        )


@router.patch("/{symbol}", response_model=BaseResponse)
async def update_position(
    symbol: str = Path(..., description="Security symbol"),
    position_updates: PositionUpdate = Body(...),
    run_compliance_check: bool = Query(True, description="Run compliance check after update"),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Partially update an existing position
    """
    try:
        position = db.query(Portfolio).filter(Portfolio.symbol == symbol).first()
        if not position:
            raise PositionNotFound(symbol)
        
        # Store old values for history
        old_values = {
            field: getattr(position, field) 
            for field in position_updates.dict(exclude_unset=True).keys()
            if hasattr(position, field)
        }
        
        # Apply updates
        update_data = position_updates.dict(exclude_unset=True)
        changed_fields = []
        
        for field, value in update_data.items():
            if hasattr(position, field):
                setattr(position, field, value)
                changed_fields.append(field)
        
        position.last_updated = datetime.now()
        
        # Record change in history
        history = PositionHistory(
            position_id=position.position_id,
            symbol=symbol,
            weight=position.weight,
            market_value=position.market_value,
            change_type="update",
            changed_fields=changed_fields,
            previous_values=old_values,
            change_reason="API patch update",
            changed_by=current_user
        )
        db.add(history)
        
        db.commit()
        
        # Log the change
        compliance_logger.log_portfolio_update(symbol=symbol, changes=update_data)
        
        response_data = {
            "success": True,
            "message": f"Position {symbol} updated successfully",
            "symbol": symbol,
            "updated_fields": changed_fields
        }
        
        # Run compliance check if requested
        if run_compliance_check:
            try:
                engine = ComplianceEngine(db)
                compliance_results = engine.evaluate_all_rules()
                response_data["compliance_check"] = {
                    "executed": True,
                    "compliance_rate": compliance_results.get("compliance_rate", 0),
                    "new_breaches": len(compliance_results.get("new_breaches", [])),
                    "evaluation_id": compliance_results.get("evaluation_id")
                }
            except Exception as comp_e:
                logger.warning(f"Compliance check failed after position update: {comp_e}")
                response_data["compliance_check"] = {
                    "executed": False,
                    "error": str(comp_e)
                }
        
        return response_data
        
    except PositionNotFound:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating position {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update position: {str(e)}"
        )


@router.delete("/{symbol}", response_model=BaseResponse)
async def delete_position(
    symbol: str = Path(..., description="Security symbol"),
    soft_delete: bool = Query(True, description="Soft delete (set weight to 0) vs hard delete"),
    run_compliance_check: bool = Query(True, description="Run compliance check after deletion"),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Delete a portfolio position
    """
    try:
        position = db.query(Portfolio).filter(Portfolio.symbol == symbol).first()
        if not position:
            raise PositionNotFound(symbol)
        
        if soft_delete:
            # Soft delete - set weight and market value to 0
            old_weight = position.weight
            old_market_value = position.market_value
            
            position.weight = 0
            position.market_value = 0
            position.last_updated = datetime.now()
            
            # Record soft delete in history
            history = PositionHistory(
                position_id=position.position_id,
                symbol=symbol,
                weight=0,
                market_value=0,
                change_type="soft_delete",
                previous_values={"weight": old_weight, "market_value": old_market_value},
                change_reason="API soft delete",
                changed_by=current_user
            )
            db.add(history)
            
            action = "soft deleted (weight set to 0)"
        else:
            # Hard delete
            # Record deletion in history first
            history = PositionHistory(
                position_id=position.position_id,
                symbol=symbol,
                weight=position.weight,
                market_value=position.market_value,
                change_type="delete",
                previous_values={
                    "weight": position.weight,
                    "market_value": position.market_value,
                    "sector": position.sector,
                    "country": position.country
                },
                change_reason="API hard delete",
                changed_by=current_user
            )
            db.add(history)
            
            db.delete(position)
            action = "permanently deleted"
        
        db.commit()
        
        # Log the deletion
        compliance_logger.log_portfolio_update(
            symbol=symbol, 
            changes={"action": "delete", "type": "soft" if soft_delete else "hard"}
        )
        
        response_data = {
            "success": True,
            "message": f"Position {symbol} {action} successfully",
            "symbol": symbol,
            "action": "soft_delete" if soft_delete else "hard_delete"
        }
        
        # Run compliance check if requested
        if run_compliance_check:
            try:
                engine = ComplianceEngine(db)
                compliance_results = engine.evaluate_all_rules()
                response_data["compliance_check"] = {
                    "executed": True,
                    "compliance_rate": compliance_results.get("compliance_rate", 0),
                    "new_breaches": len(compliance_results.get("new_breaches", [])),
                    "evaluation_id": compliance_results.get("evaluation_id")
                }
            except Exception as comp_e:
                logger.warning(f"Compliance check failed after position deletion: {comp_e}")
                response_data["compliance_check"] = {
                    "executed": False,
                    "error": str(comp_e)
                }
        
        return response_data
        
    except PositionNotFound:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting position {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete position: {str(e)}"
        )


@router.get("/{symbol}", response_model=PositionResponse)
async def get_position(
    symbol: str = Path(..., description="Security symbol"),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Get details of a specific position
    """
    try:
        position = db.query(Portfolio).filter(Portfolio.symbol == symbol).first()
        if not position:
            raise PositionNotFound(symbol)
        
        return PositionResponse(
            position_id=getattr(position, 'position_id', position.portfolio_id),
            symbol=position.symbol,
            name=getattr(position, 'name', None),
            weight=getattr(position, 'weight', 0.0),
            market_value=getattr(position, 'market_value', 0.0),
            quantity=getattr(position, 'quantity', None),
            price=getattr(position, 'price', None),
            sector=getattr(position, 'sector', None),
            industry=getattr(position, 'industry', None),
            country=getattr(position, 'country', None),
            currency=getattr(position, 'currency', "USD"),
            rating=getattr(position, 'rating', None),
            rating_agency=getattr(position, 'rating_agency', None),
            instrument_type=getattr(position, 'instrument_type', None),
            exchange=getattr(position, 'exchange', None),
            maturity_date=getattr(position, 'maturity_date', None),
            acquisition_date=getattr(position, 'acquisition_date', None),
            bloomberg_id=getattr(position, 'bloomberg_id', None),
            cusip=getattr(position, 'cusip', None),
            isin=getattr(position, 'isin', None),
            sedol=getattr(position, 'sedol', None),
            metadata=getattr(position, 'metadata', {}) or {},
            last_updated=getattr(position, 'last_updated', datetime.now()),
            created_at=getattr(position, 'created_at', datetime.now())
        )
        
    except PositionNotFound:
        raise
    except Exception as e:
        logger.error(f"Error retrieving position {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve position: {str(e)}"
        )


@router.get("/{symbol}/history")
async def get_position_history(
    symbol: str = Path(..., description="Security symbol"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of history records"),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Get change history for a specific position
    """
    try:
        position = db.query(Portfolio).filter(Portfolio.symbol == symbol).first()
        if not position:
            raise PositionNotFound(symbol)
        
        history = db.query(PositionHistory).filter(
            PositionHistory.symbol == symbol
        ).order_by(
            PositionHistory.change_timestamp.desc()
        ).limit(limit).all()
        
        history_data = []
        for record in history:
            history_data.append({
                "history_id": record.history_id,
                "change_timestamp": record.change_timestamp.isoformat(),
                "change_type": record.change_type,
                "changed_fields": record.changed_fields or [],
                "previous_values": record.previous_values or {},
                "new_weight": record.weight,
                "new_market_value": record.market_value,
                "change_reason": record.change_reason,
                "changed_by": record.changed_by,
                "metadata": record.metadata or {}
            })
        
        return {
            "symbol": symbol,
            "history_records": len(history_data),
            "history": history_data
        }
        
    except PositionNotFound:
        raise
    except Exception as e:
        logger.error(f"Error retrieving position history for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve position history: {str(e)}"
        )


@router.get("/summary/overview", response_model=PortfolioSummary)
async def get_portfolio_summary(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Get comprehensive portfolio summary and analytics
    """
    try:
        engine = ComplianceEngine(db)
        portfolio_cache = engine._load_portfolio_cache()
        
        if not portfolio_cache:
            return PortfolioSummary(
                total_positions=0,
                total_market_value=0,
                total_weight=0,
                top_positions={},
                sector_breakdown={},
                country_breakdown={},
                rating_breakdown={},
                concentration_metrics={},
                last_updated=datetime.now()
            )
        
        summary = engine._get_portfolio_summary()
        
        return PortfolioSummary(
            total_positions=summary["total_positions"],
            total_market_value=summary["total_market_value"],
            total_weight=summary["total_weight"],
            top_positions=summary["top_5_positions"],
            sector_breakdown=summary["sector_breakdown"],
            country_breakdown=summary["country_breakdown"],
            rating_breakdown=summary["rating_breakdown"],
            concentration_metrics=summary["concentration_metrics"],
            last_updated=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error generating portfolio summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate portfolio summary: {str(e)}"
        )


@router.post("/bulk-update", response_model=BaseResponse)
async def bulk_update_positions(
    positions: List[PositionCreate] = Body(...),
    run_compliance_check: bool = Query(True, description="Run compliance check after updates"),
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Bulk update multiple portfolio positions
    """
    try:
        if len(positions) > 1000:
            raise ValidationException("Maximum 1000 positions allowed in bulk update")
        
        updated_count = 0
        created_count = 0
        errors = []
        
        for position_data in positions:
            try:
                symbol = position_data.symbol
                existing_position = db.query(Portfolio).filter(Portfolio.symbol == symbol).first()
                
                if existing_position:
                    # Update existing
                    for field, value in position_data.dict(exclude_unset=True).items():
                        if hasattr(existing_position, field):
                            setattr(existing_position, field, value)
                    existing_position.last_updated = datetime.now()
                    updated_count += 1
                else:
                    # Create new
                    new_position = Portfolio(**position_data.dict())
                    db.add(new_position)
                    created_count += 1
                
            except Exception as pos_e:
                errors.append({
                    "symbol": getattr(position_data, 'symbol', 'unknown'),
                    "error": str(pos_e)
                })
        
        db.commit()
        
        response_data = {
            "success": True,
            "message": f"Bulk update completed: {created_count} created, {updated_count} updated",
            "created_count": created_count,
            "updated_count": updated_count,
            "error_count": len(errors),
            "errors": errors[:10]  # Limit error details
        }
        
        # Run compliance check if requested
        if run_compliance_check and (created_count > 0 or updated_count > 0):
            try:
                engine = ComplianceEngine(db)
                compliance_results = engine.evaluate_all_rules()
                response_data["compliance_check"] = {
                    "executed": True,
                    "compliance_rate": compliance_results.get("compliance_rate", 0),
                    "new_breaches": len(compliance_results.get("new_breaches", [])),
                    "evaluation_id": compliance_results.get("evaluation_id")
                }
            except Exception as comp_e:
                logger.warning(f"Compliance check failed after bulk update: {comp_e}")
                response_data["compliance_check"] = {
                    "executed": False,
                    "error": str(comp_e)
                }
        
        # Log bulk update
        compliance_logger.log_system_event(
            "bulk_portfolio_update",
            f"Bulk update completed by {current_user}",
            created_count=created_count,
            updated_count=updated_count,
            error_count=len(errors)
        )
        
        return response_data
        
    except ValidationException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error in bulk position update: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bulk update failed: {str(e)}"
        )