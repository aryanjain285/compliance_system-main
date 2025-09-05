"""
Policy Document Management API Routes
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
from pathlib import Path as FilePath

from app.models.database import get_db, PolicyDocument, PolicyChunk
from app.schemas import (
    PolicyDocumentResponse, PolicyUploadResponse, PolicySearchRequest,
    PolicySearchResult, SemanticSearchRequest, SemanticSearchResponse,
    KnowledgeQueryRequest, KnowledgeQueryResponse, BaseResponse
)
from app.api.dependencies import (
    get_policy_parser, get_current_user, get_optional_llm_service,
    get_optional_vector_service, rate_limit_file_upload, rate_limit_llm_requests,
    validate_file_upload_permissions
)
from app.services.policy_parser import PolicyParser
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStoreService
from app.utils.logger import get_logger, compliance_logger
from app.utils.exceptions import (
    PolicyNotFound, FileUploadException, UnsupportedFileType, FileTooLarge
)
from app.config.settings import get_settings

settings = get_settings()
logger = get_logger(__name__)
router = APIRouter()


@router.post("/upload", response_model=PolicyUploadResponse)
async def upload_policy_document(
    file: UploadFile = File(...),
    add_to_knowledge_base: bool = Query(True, description="Add to vector knowledge base"),
    extract_rules: bool = Query(True, description="Extract rules using LLM"),
    document_type: Optional[str] = Query(None, description="Document type override"),
    jurisdiction: Optional[str] = Query(None, description="Jurisdiction override"),
    parser: PolicyParser = Depends(get_policy_parser),
    llm_service: LLMService = Depends(get_optional_llm_service),
    vector_service: VectorStoreService = Depends(get_optional_vector_service),
    current_user: str = Depends(rate_limit_file_upload),
    db: Session = Depends(get_db)
):
    """
    Upload and process a policy document with comprehensive analysis
    """
    try:
        # Validate file
        if not file.filename:
            raise FileUploadException("Filename is required")
        
        file_ext = FilePath(file.filename).suffix.lower()
        if file_ext.lstrip('.') not in settings.allowed_file_types:
            raise UnsupportedFileType(file_ext, settings.allowed_file_types)
        
        # Read file content
        content = await file.read()
        
        if len(content) > settings.max_file_size_bytes:
            raise FileTooLarge(len(content), settings.max_file_size_bytes)
        
        # Process with policy parser
        metadata = {}
        if document_type:
            metadata['document_type'] = document_type
        if jurisdiction:
            metadata['jurisdiction'] = jurisdiction
        
        policy_id = parser.upload_policy_document(
            filename=file.filename,
            content=content,
            uploaded_by=current_user,
            metadata=metadata
        )
        
        # Get policy and chunks for further processing
        policy = db.query(PolicyDocument).filter(
            PolicyDocument.policy_id == policy_id
        ).first()
        
        chunks = db.query(PolicyChunk).filter(
            PolicyChunk.policy_id == policy_id
        ).all()
        
        chunk_data = [
            {
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "chunk_id": f"{policy_id}_{chunk.chunk_index}",
                "word_count": chunk.word_count,
                "char_count": chunk.char_count,
                "metadata": chunk.metadata or {}
            }
            for chunk in chunks
        ]
        
        # Extract rules using LLM if available and requested
        extracted_rules = []
        if extract_rules and llm_service:
            try:
                policy_metadata = {
                    "document_type": policy.document_type or "Policy Document",
                    "jurisdiction": policy.jurisdiction or "General",
                    "filename": policy.filename
                }
                
                extracted_rules = await llm_service.extract_rules_from_chunks(chunk_data, policy_metadata)
                logger.info(f"Extracted {len(extracted_rules)} rules from policy {policy_id}")
                
            except Exception as llm_e:
                logger.warning(f"LLM rule extraction failed for policy {policy_id}: {llm_e}")
        
        # Add to vector knowledge base if requested and available
        vector_indexing = {
            "requested": add_to_knowledge_base,
            "successful": False,
            "error": None
        }
        
        if add_to_knowledge_base and vector_service and vector_service.is_available():
            try:
                success = vector_service.add_policy_chunks(policy_id, chunk_data)
                vector_indexing["successful"] = success
                
                if success:
                    logger.info(f"Added {len(chunk_data)} chunks to vector store for policy {policy_id}")
                else:
                    vector_indexing["error"] = "Vector indexing failed"
                
            except Exception as vector_e:
                logger.warning(f"Vector indexing failed for policy {policy_id}: {vector_e}")
                vector_indexing["error"] = str(vector_e)
        elif add_to_knowledge_base:
            vector_indexing["error"] = "Vector service not available"
        
        # Log policy upload
        compliance_logger.log_policy_uploaded(
            policy_id=policy_id,
            filename=file.filename,
            chunks_count=len(chunk_data),
            rules_extracted=len(extracted_rules)
        )
        
        return PolicyUploadResponse(
            success=True,
            message="Policy uploaded and processed successfully",
            policy_id=policy_id,
            filename=file.filename,
            chunks_count=len(chunk_data),
            extracted_rules_count=len(extracted_rules),
            extracted_rules=extracted_rules,
            vector_indexing=vector_indexing,
            requires_approval=len(extracted_rules) > 0
        )
        
    except (FileUploadException, UnsupportedFileType, FileTooLarge):
        raise
    except Exception as e:
        logger.error(f"Error uploading policy {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Policy upload failed: {str(e)}"
        )


@router.get("", response_model=List[PolicyDocumentResponse])
async def get_policies(
    status_filter: Optional[str] = Query(None, description="Filter by document status"),
    document_type_filter: Optional[str] = Query(None, description="Filter by document type"),
    jurisdiction_filter: Optional[str] = Query(None, description="Filter by jurisdiction"),
    search: Optional[str] = Query(None, description="Search in filename and metadata"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    parser: PolicyParser = Depends(get_policy_parser),
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get policy documents with filtering and pagination
    """
    try:
        query = db.query(PolicyDocument)
        
        # Apply filters safely
        if status_filter:
            query = query.filter(PolicyDocument.status == status_filter)
        
        if document_type_filter and hasattr(PolicyDocument, 'document_type'):
            query = query.filter(PolicyDocument.document_type == document_type_filter)
        
        if jurisdiction_filter and hasattr(PolicyDocument, 'jurisdiction'):
            query = query.filter(PolicyDocument.jurisdiction == jurisdiction_filter)
        
        if search:
            search_term = f"%{search}%"
            query = query.filter(PolicyDocument.filename.ilike(search_term))
        
        # Get total count for pagination
        total_count = query.count()
        
        # Apply pagination and ordering
        policies = query.order_by(
            PolicyDocument.created_at.desc()
        ).offset(offset).limit(limit).all()
        
        # Build response safely
        policy_responses = []
        for policy in policies:
            try:
                # Get summary info safely
                try:
                    summary = parser.get_policy_summary(policy.policy_id)
                except:
                    summary = {"content_stats": {}, "chunk_stats": {}, "rules_generated": 0}
                
                policy_responses.append(PolicyDocumentResponse(
                    policy_id=policy.policy_id,
                    filename=policy.filename,
                    document_type=getattr(policy, 'document_type', None),
                    jurisdiction=getattr(policy, 'jurisdiction', None),
                    status=policy.status,
                    upload_date=getattr(policy, 'upload_date', policy.created_at),
                    effective_date=getattr(policy, 'effective_date', None),
                    expiry_date=getattr(policy, 'expiry_date', None),
                    uploaded_by=policy.uploaded_by,
                    version=getattr(policy, 'version', 1),
                    content_stats=summary.get("content_stats", {}),
                    chunk_stats=summary.get("chunk_stats", {}),
                    rules_generated=summary.get("rules_generated", 0),
                    metadata=getattr(policy, 'metadata', {}) or {},
                    created_at=policy.created_at,
                    updated_at=getattr(policy, 'updated_at', policy.created_at)
                ))
            except Exception as policy_e:
                logger.error(f"Error processing policy {policy.policy_id}: {policy_e}")
                continue
        
        logger.info(f"Retrieved {len(policy_responses)} policies (total: {total_count}) for user {current_user}")
        return policy_responses
        
    except Exception as e:
        logger.error(f"Error retrieving policies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve policies: {str(e)}"
        )


@router.get("/{policy_id}", response_model=PolicyDocumentResponse)
async def get_policy_detail(
    policy_id: str = Path(..., description="Policy ID"),
    parser: PolicyParser = Depends(get_policy_parser),
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific policy document
    """
    try:
        policy = db.query(PolicyDocument).filter(
            PolicyDocument.policy_id == policy_id
        ).first()
        
        if not policy:
            raise PolicyNotFound(policy_id)
        
        # Get comprehensive summary
        summary = parser.get_policy_summary(policy_id)
        
        if not summary:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate policy summary"
            )
        
        return PolicyDocumentResponse(
            policy_id=policy.policy_id,
            filename=policy.filename,
            document_type=policy.document_type,
            jurisdiction=policy.jurisdiction,
            status=policy.status,
            upload_date=policy.upload_date,
            effective_date=policy.effective_date,
            expiry_date=policy.expiry_date,
            uploaded_by=policy.uploaded_by,
            version=policy.version,
            content_stats=summary.get("content_stats", {}),
            chunk_stats=summary.get("chunk_stats", {}),
            rules_generated=summary.get("rules_generated", 0),
            metadata=policy.metadata or {},
            created_at=policy.created_at,
            updated_at=policy.updated_at
        )
        
    except PolicyNotFound:
        raise
    except Exception as e:
        logger.error(f"Error retrieving policy {policy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve policy details: {str(e)}"
        )


@router.post("/{policy_id}/search", response_model=List[PolicySearchResult])
async def search_policy_content(
    policy_id: str = Path(..., description="Policy ID"),
    search_request: PolicySearchRequest = Body(...),
    parser: PolicyParser = Depends(get_policy_parser),
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Search for specific terms within a policy document
    """
    try:
        # Verify policy exists
        policy = db.query(PolicyDocument).filter(
            PolicyDocument.policy_id == policy_id
        ).first()
        
        if not policy:
            raise PolicyNotFound(policy_id)
        
        # Perform search
        results = parser.search_policy_content(
            policy_id=policy_id,
            search_terms=search_request.search_terms,
            case_sensitive=search_request.case_sensitive
        )
        
        # Convert to response format
        search_results = []
        for result in results:
            search_results.append(PolicySearchResult(
                chunk_id=result["chunk_id"],
                chunk_index=result["chunk_index"],
                section_title=result["section_title"],
                page_number=result["page_number"],
                matches=result["matches"],
                match_count=result["match_count"]
            ))
        
        return search_results
        
    except PolicyNotFound:
        raise
    except Exception as e:
        logger.error(f"Error searching policy {policy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Policy search failed: {str(e)}"
        )


@router.post("/semantic-search", response_model=SemanticSearchResponse)
async def semantic_search_policies(
    search_request: SemanticSearchRequest = Body(...),
    vector_service: VectorStoreService = Depends(get_optional_vector_service),
    current_user: str = Depends(get_current_user)
):
    """
    Perform semantic search across all policy documents
    """
    try:
        if not vector_service or not vector_service.is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector search service not available"
            )
        
        # Perform semantic search
        if search_request.keywords:
            # Use hybrid search with keyword boosting
            results = vector_service.hybrid_search(
                query=search_request.query,
                keywords=search_request.keywords,
                n_results=search_request.n_results
            )
            search_type = "hybrid"
        else:
            # Use pure semantic search
            results = vector_service.semantic_search(
                query=search_request.query,
                n_results=search_request.n_results,
                policy_id=search_request.policy_id,
                min_relevance_score=search_request.min_relevance_score
            )
            search_type = "semantic"
        
        return SemanticSearchResponse(
            query=search_request.query,
            results=results,
            total_found=len(results),
            search_type=search_type,
            filters_applied={
                "policy_id": search_request.policy_id,
                "keywords": search_request.keywords,
                "min_relevance_score": search_request.min_relevance_score
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}"
        )


@router.post("/ask", response_model=KnowledgeQueryResponse)
async def ask_knowledge_base(
    query_request: KnowledgeQueryRequest = Body(...),
    vector_service: VectorStoreService = Depends(get_optional_vector_service),
    llm_service: LLMService = Depends(get_optional_llm_service),
    current_user: str = Depends(rate_limit_llm_requests)
):
    """
    Ask natural language questions about policies with LLM-powered answers
    """
    try:
        if not vector_service or not vector_service.is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector search service not available for knowledge queries"
            )
        
        # Search for relevant policy context
        context_results = vector_service.semantic_search(
            query=query_request.question,
            n_results=query_request.n_context,
            policy_id=query_request.policy_id,
            min_relevance_score=0.3
        )
        
        if not context_results:
            return KnowledgeQueryResponse(
                question=query_request.question,
                answer="I couldn't find relevant information in the policy knowledge base to answer your question.",
                confidence=0.0,
                sources=[],
                context_used=0
            )
        
        # Generate answer using LLM if available
        if llm_service:
            # Use LLM to generate a proper answer based on context
            context_text = "\n\n".join([
                f"[Source: {ctx['section_title'] or 'Unknown'}]\n{ctx['content'][:500]}..."
                for ctx in context_results[:3]
            ])
            
            prompt = f"""
Based on the following policy information, answer this question: {query_request.question}

Policy Context:
{context_text}

Provide a clear, accurate answer based only on the information provided. If the information is insufficient, say so.
"""
            
            try:
                response = await llm_service.generate(
                    prompt,
                    temperature=0.1,
                    max_tokens=500,
                    system_message="You are a compliance assistant answering questions based on policy documents."
                )
                answer = response.content.strip()
                confidence = min(max([ctx["relevance_score"] for ctx in context_results]) * 0.9, 0.95)
                
            except Exception as llm_e:
                logger.warning(f"LLM answer generation failed: {llm_e}")
                # Fallback to context summary
                answer = f"Based on the policy documents, here are the relevant findings:\n\n{context_text[:300]}..."
                confidence = max([ctx["relevance_score"] for ctx in context_results])
        else:
            # No LLM available - provide context summary
            answer = f"Based on the policy documents, here are the relevant sections:\n\n"
            for i, ctx in enumerate(context_results[:3], 1):
                answer += f"{i}. {ctx['section_title'] or 'Policy Section'}: {ctx['content'][:200]}...\n\n"
            confidence = max([ctx["relevance_score"] for ctx in context_results])
        
        # Prepare sources
        sources = [
            {
                "policy_id": ctx["policy_id"],
                "section": ctx.get("section_title", "Unknown"),
                "page_number": ctx.get("page_number"),
                "relevance_score": ctx["relevance_score"],
                "content_preview": ctx["content"][:100] + "..."
            }
            for ctx in context_results
        ]
        
        return KnowledgeQueryResponse(
            question=query_request.question,
            answer=answer,
            confidence=round(confidence, 3),
            sources=sources,
            context_used=len(context_results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing knowledge base query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Knowledge query failed: {str(e)}"
        )


@router.put("/{policy_id}/metadata", response_model=BaseResponse)
async def update_policy_metadata(
    policy_id: str = Path(..., description="Policy ID"),
    metadata_updates: Dict[str, Any] = Body(...),
    parser: PolicyParser = Depends(get_policy_parser),
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update policy document metadata
    """
    try:
        # Verify policy exists
        policy = db.query(PolicyDocument).filter(
            PolicyDocument.policy_id == policy_id
        ).first()
        
        if not policy:
            raise PolicyNotFound(policy_id)
        
        # Update metadata
        success = parser.update_policy_metadata(policy_id, metadata_updates)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update policy metadata"
            )
        
        return BaseResponse(
            success=True,
            message=f"Policy metadata updated successfully",
            timestamp=datetime.now()
        )
        
    except PolicyNotFound:
        raise
    except Exception as e:
        logger.error(f"Error updating policy metadata {policy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metadata update failed: {str(e)}"
        )


@router.delete("/{policy_id}", response_model=BaseResponse)
async def delete_policy(
    policy_id: str = Path(..., description="Policy ID"),
    remove_from_vector_store: bool = Query(True, description="Remove from vector knowledge base"),
    parser: PolicyParser = Depends(get_policy_parser),
    vector_service: VectorStoreService = Depends(get_optional_vector_service),
    current_user: str = Depends(validate_file_upload_permissions),
    db: Session = Depends(get_db)
):
    """
    Delete a policy document and associated data
    """
    try:
        # Verify policy exists
        policy = db.query(PolicyDocument).filter(
            PolicyDocument.policy_id == policy_id
        ).first()
        
        if not policy:
            raise PolicyNotFound(policy_id)
        
        # Check for dependencies (rules created from this policy)
        from app.models.database import ComplianceRule
        dependent_rules = db.query(ComplianceRule).filter(
            ComplianceRule.source_policy_id == policy_id,
            ComplianceRule.is_active == True
        ).count()
        
        if dependent_rules > 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot delete policy: {dependent_rules} active rules depend on it"
            )
        
        # Remove from vector store if requested and available
        vector_removal_success = True
        if remove_from_vector_store and vector_service and vector_service.is_available():
            try:
                vector_removal_success = vector_service.delete_policy_chunks(policy_id)
            except Exception as vector_e:
                logger.warning(f"Failed to remove policy from vector store: {vector_e}")
                vector_removal_success = False
        
        # Delete policy and chunks
        delete_success = parser.delete_policy(policy_id)
        
        if not delete_success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete policy from database"
            )
        
        # Log policy deletion
        compliance_logger.log_system_event(
            "policy_deleted",
            f"Policy document deleted: {policy.filename}",
            user_id=current_user,
            policy_id=policy_id,
            filename=policy.filename
        )
        
        return BaseResponse(
            success=True,
            message=f"Policy {policy.filename} deleted successfully",
            timestamp=datetime.now()
        )
        
    except (PolicyNotFound, HTTPException):
        raise
    except Exception as e:
        logger.error(f"Error deleting policy {policy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Policy deletion failed: {str(e)}"
        )


@router.get("/{policy_id}/chunks")
async def get_policy_chunks(
    policy_id: str = Path(..., description="Policy ID"),
    limit: int = Query(50, ge=1, le=500, description="Maximum chunks to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get chunks for a specific policy document
    """
    try:
        # Verify policy exists
        policy = db.query(PolicyDocument).filter(
            PolicyDocument.policy_id == policy_id
        ).first()
        
        if not policy:
            raise PolicyNotFound(policy_id)
        
        # Get chunks with pagination
        chunks = db.query(PolicyChunk).filter(
            PolicyChunk.policy_id == policy_id
        ).order_by(
            PolicyChunk.chunk_index
        ).offset(offset).limit(limit).all()
        
        chunk_data = []
        for chunk in chunks:
            chunk_data.append({
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "word_count": chunk.word_count,
                "char_count": chunk.char_count,
                "metadata": chunk.metadata or {},
                "created_at": chunk.created_at.isoformat() if chunk.created_at else None
            })
        
        total_chunks = db.query(PolicyChunk).filter(
            PolicyChunk.policy_id == policy_id
        ).count()
        
        return {
            "policy_id": policy_id,
            "filename": policy.filename,
            "total_chunks": total_chunks,
            "returned_chunks": len(chunk_data),
            "offset": offset,
            "limit": limit,
            "chunks": chunk_data
        }
        
    except PolicyNotFound:
        raise
    except Exception as e:
        logger.error(f"Error retrieving chunks for policy {policy_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve policy chunks: {str(e)}"
        )


@router.get("/stats/processing")
async def get_processing_statistics(
    parser: PolicyParser = Depends(get_policy_parser),
    vector_service: VectorStoreService = Depends(get_optional_vector_service),
    current_user: str = Depends(get_current_user)
):
    """
    Get statistics about policy document processing
    """
    try:
        # Get processing statistics from parser
        processing_stats = parser.get_processing_statistics()
        
        # Get vector store statistics if available
        vector_stats = {}
        if vector_service and vector_service.is_available():
            try:
                vector_stats = vector_service.get_collection_stats()
            except Exception as vector_e:
                logger.warning(f"Failed to get vector store stats: {vector_e}")
                vector_stats = {"error": str(vector_e)}
        
        return {
            "processing_statistics": processing_stats,
            "vector_store_statistics": vector_stats,
            "system_health": {
                "parser_available": True,
                "vector_store_available": bool(vector_service and vector_service.is_available())
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting processing statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get processing statistics: {str(e)}"
        )