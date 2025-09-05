"""
Vector Store Service using ChromaDB for Semantic Search
Handles policy document embeddings, semantic search, and knowledge retrieval
"""

import os
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    VECTOR_DEPS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    chromadb = None
    Settings = None
    embedding_functions = None
    VECTOR_DEPS_AVAILABLE = False

from app.config.settings import get_settings
from app.utils.logger import get_logger, log_execution_time
from app.utils.exceptions import VectorStoreException, ServiceUnavailable

settings = get_settings()
logger = get_logger(__name__)


class VectorStoreService:
    """Advanced vector store service for semantic search and retrieval"""

    def __init__(self):
        self.persist_directory = settings.chroma_persist_dir
        self.collection_name = settings.vector_collection_name
        self.embedding_model_name = settings.embedding_model

        # Initialize ChromaDB client & model
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.embedding_function = None

        if not VECTOR_DEPS_AVAILABLE:
            logger.warning(
                "Vector store dependencies not available. Vector store features will be disabled."
            )
            return

        if settings.skip_vector_store:
            logger.info("Vector store initialization skipped due to configuration")
            return

        try:
            self._initialize_client()
            self._initialize_embedding_model()
            self._initialize_collection()
        except Exception as e:
            logger.error(
                f"Failed to initialize vector store: {e}. Vector store features will be disabled."
            )

    # ---------- Initialization ----------
    def _initialize_client(self):
        """Initialize ChromaDB persistent client"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            # Persistent client (Chroma >= 0.4)
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            logger.info(
                f"ChromaDB client initialized with persist directory: {self.persist_directory}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise VectorStoreException(f"ChromaDB initialization failed: {str(e)}")

    def _initialize_embedding_model(self):
        """Initialize sentence transformer model for embeddings"""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
            logger.info(f"Embedding model loaded: {self.embedding_model_name}")
        except Exception as e:
            logger.warning(
                f"Failed to load embedding model {self.embedding_model_name}: {e}"
            )
            # Fallback to a standard model
            try:
                fallback_model = "all-MiniLM-L6-v2"
                self.embedding_model = SentenceTransformer(fallback_model)
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=fallback_model
                )
                self.embedding_model_name = fallback_model
                logger.info(f"Using fallback embedding model: {fallback_model}")
            except Exception as fallback_e:
                logger.error(f"Failed to load fallback embedding model: {fallback_e}")
                raise VectorStoreException("No embedding model available")

    def _initialize_collection(self):
        """Get or create the vector collection (auto-creates if missing)."""
        if self.client is None or self.embedding_function is None:
            raise VectorStoreException("Client or embedding function not initialized")

        try:
            # Try retrieving an existing collection
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )
            logger.info(f"Retrieved existing collection: {self.collection_name}")
        except Exception:
            # If missing, create it with reasonable HNSW defaults
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16,
                    "created_at": datetime.now().isoformat(),
                    "embedding_model": self.embedding_model_name,
                },
            )
            logger.info(f"Created new collection: {self.collection_name}")

    # ---------- Status ----------
    def is_available(self) -> bool:
        return VECTOR_DEPS_AVAILABLE and all(
            [self.client is not None, self.collection is not None, self.embedding_model is not None]
        )

    def get_service_status(self) -> Dict[str, Any]:
        if not self.is_available():
            return {
                "service": "vector_store",
                "status": "unavailable",
                "error": "Service not properly initialized",
                "skip_vector_store": settings.skip_vector_store,
            }
        stats = self.get_collection_stats()
        return {
            "service": "vector_store",
            "status": "ready",
            "client_type": "chromadb_persistent",
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name,
            "persist_directory": self.persist_directory,
            "collection_stats": stats,
            "capabilities": {
                "semantic_search": True,
                "hybrid_search": True,
                "similarity_search": True,
                "clustering_analysis": True,
                "backup_restore": True,
                "batch_processing": True,
            },
            "health_check_timestamp": datetime.now().isoformat(),
        }

    # ---------- Write ----------
    @log_execution_time("add_policy_chunks")
    def add_policy_chunks(
        self, policy_id: str, chunks: List[Dict[str, Any]], batch_size: int = 100
    ) -> bool:
        """Add policy chunks to vector store with batch processing."""
        try:
            if not self.is_available():
                raise ServiceUnavailable("vector_store", "Service not initialized")

            if not chunks:
                logger.warning("No chunks provided for indexing")
                return True

            # Remove existing chunks for this policy to avoid duplicates
            self._remove_policy_chunks(policy_id)

            total_chunks = len(chunks)
            successfully_added = 0

            for i in range(0, total_chunks, batch_size):
                batch = chunks[i : i + batch_size]

                batch_ids: List[str] = []
                batch_documents: List[str] = []
                batch_metadatas: List[Dict[str, Any]] = []

                for chunk in batch:
                    # accept both 'metadata' and 'chunk_metadata' as source meta
                    meta_src = chunk.get("metadata") or chunk.get("chunk_metadata") or {}
                    content = chunk.get("content", "")
                    idx = chunk.get("chunk_index")

                    if not content or len(content.strip()) < 10 or idx is None:
                        continue

                    chunk_id = f"{policy_id}_{idx}"

                    metadata = {
                        "policy_id": policy_id,
                        "chunk_index": idx,
                        "page_number": chunk.get("page_number", 0),
                        "section_title": chunk.get("section_title", ""),
                        "word_count": chunk.get("word_count", len(content.split())),
                        "char_count": chunk.get("char_count", len(content)),
                        "content_hash": chunk.get(
                            "content_hash", hashlib.md5(content.encode()).hexdigest()
                        ),
                        "chunk_type": meta_src.get("chunk_type", "general"),
                        "created_at": datetime.now().isoformat(),
                        "embedding_model": self.embedding_model_name,
                    }

                    batch_ids.append(chunk_id)
                    batch_documents.append(content)
                    batch_metadatas.append(metadata)

                if batch_ids:
                    self.collection.add(
                        ids=batch_ids, documents=batch_documents, metadatas=batch_metadatas
                    )
                    successfully_added += len(batch_ids)
                    logger.debug(
                        f"Indexed batch {i // batch_size + 1}: {len(batch_ids)} chunks"
                    )

            logger.info(
                f"Successfully added {successfully_added}/{total_chunks} chunks "
                f"for policy {policy_id} to vector store"
            )
            return successfully_added > 0

        except Exception as e:
            logger.error(f"Error adding policy chunks to vector store: {e}")
            raise VectorStoreException(f"Failed to add policy chunks: {str(e)}")

    def delete_policy_chunks(self, policy_id: str) -> bool:
        try:
            if not self.is_available():
                return False
            self._remove_policy_chunks(policy_id)
            return True
        except Exception as e:
            logger.error(f"Error deleting policy chunks: {e}")
            return False

    def _remove_policy_chunks(self, policy_id: str):
        """Remove all chunks for a specific policy."""
        try:
            existing = self.collection.get(where={"policy_id": policy_id}, include=["metadatas"])
            if existing and existing.get("ids"):
                self.collection.delete(ids=existing["ids"])
                logger.info(
                    f"Removed {len(existing['ids'])} existing chunks for policy {policy_id}"
                )
        except Exception as e:
            logger.warning(f"Error removing existing chunks for policy {policy_id}: {e}")

    # ---------- Read / Search ----------
    @log_execution_time("semantic_search")
    def semantic_search(
        self,
        query: str,
        n_results: int = 10,
        policy_id: Optional[str] = None,
        min_relevance_score: float = 0.5,
        chunk_types: Optional[List[str]] = None,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform advanced semantic search with filtering."""
        try:
            if not self.is_available():
                raise ServiceUnavailable("vector_store", "Service not initialized")

            if not query or len(query.strip()) < 3:
                return []

            where_clause: Dict[str, Any] = {}
            if policy_id:
                where_clause["policy_id"] = policy_id
            if chunk_types:
                where_clause["chunk_type"] = {"$in": chunk_types}

            results = self.collection.query(
                query_texts=[query],
                n_results=min(max(n_results, 1), 100),
                where=where_clause or None,
                include=["documents", "metadatas", "distances"],
            )

            formatted: List[Dict[str, Any]] = []
            docs = results.get("documents") or []
            metas = results.get("metadatas") or []
            dists = results.get("distances") or []
            if docs and docs[0]:
                for i, (doc, meta, dist) in enumerate(
                    zip(docs[0], metas[0] if metas else [], dists[0] if dists else [])
                ):
                    similarity = 1 - float(dist)
                    if similarity < min_relevance_score:
                        continue

                    # Date filter if provided
                    if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                        created_at = (meta or {}).get("created_at")
                        if created_at:
                            try:
                                # Try robust parse
                                dt = datetime.fromisoformat(created_at.replace("Z", ""))
                                start = datetime.fromisoformat(date_range[0])
                                end = datetime.fromisoformat(date_range[1])
                                if not (start <= dt <= end):
                                    continue
                            except Exception:
                                # If parse fails, skip date filtering for this item
                                pass

                    formatted.append(
                        {
                            "content": doc,
                            "metadata": meta,
                            "relevance_score": round(similarity, 4),
                            "distance": round(float(dist), 4),
                            "policy_id": (meta or {}).get("policy_id"),
                            "chunk_index": (meta or {}).get("chunk_index"),
                            "section_title": (meta or {}).get("section_title"),
                            "page_number": (meta or {}).get("page_number"),
                            "chunk_type": (meta or {}).get("chunk_type"),
                            "word_count": (meta or {}).get("word_count"),
                            "rank": i + 1,
                        }
                    )

            formatted.sort(key=lambda x: x["relevance_score"], reverse=True)
            logger.info(f"Semantic search returned {len(formatted)} results for query.")
            return formatted

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise VectorStoreException(f"Semantic search failed: {str(e)}")

    def hybrid_search(
        self,
        query: str,
        keywords: Optional[List[str]] = None,
        n_results: int = 10,
        keyword_boost: float = 0.2,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic and simple keyword boosting."""
        try:
            semantic_results = self.semantic_search(query, n_results=n_results * 2, **kwargs)
            if not keywords:
                return semantic_results[:n_results]

            boosted = []
            for result in semantic_results:
                content = (result["content"] or "").lower()
                base = result["relevance_score"]
                kscore = 0.0
                matched = []
                for kw in keywords:
                    k = kw.lower()
                    if k in content:
                        count = content.count(k)
                        kscore += min(count * 0.05, 0.2)
                        matched.append(kw)
                total_boost = min(kscore * keyword_boost, 0.3)
                result["relevance_score"] = round(min(base + total_boost, 1.0), 4)
                result["keyword_matches"] = matched
                result["keyword_boost"] = round(total_boost, 4)
                boosted.append(result)

            boosted.sort(key=lambda x: x["relevance_score"], reverse=True)
            return boosted[:n_results]
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return self.semantic_search(query, n_results, **kwargs)

    def find_similar_chunks(
        self, reference_chunk_id: str, n_results: int = 5, exclude_same_policy: bool = False
    ) -> List[Dict[str, Any]]:
        """Find chunks similar to a given chunk id."""
        try:
            if not self.is_available():
                raise ServiceUnavailable("vector_store", "Service not initialized")

            ref = self.collection.get(ids=[reference_chunk_id], include=["documents", "metadatas"])
            if not ref.get("documents"):
                return []

            ref_doc = ref["documents"][0]
            ref_meta = ref["metadatas"][0] if ref.get("metadatas") else {}

            where_clause: Dict[str, Any] = {}
            if exclude_same_policy and "policy_id" in (ref_meta or {}):
                where_clause["policy_id"] = {"$ne": ref_meta["policy_id"]}

            results = self.collection.query(
                query_texts=[ref_doc],
                n_results=max(n_results + 1, 2),
                where=where_clause or None,
                include=["documents", "metadatas", "distances"],
            )

            sims: List[Dict[str, Any]] = []
            docs = results.get("documents") or []
            metas = results.get("metadatas") or []
            dists = results.get("distances") or []
            if docs and docs[0]:
                for doc, meta, dist in zip(
                    docs[0], metas[0] if metas else [], dists[0] if dists else []
                ):
                    # Skip same chunk (by reconstructed id)
                    chunk_id = f"{(meta or {}).get('policy_id')}_{(meta or {}).get('chunk_index')}"
                    if chunk_id == reference_chunk_id:
                        continue
                    sims.append(
                        {
                            "content": doc,
                            "metadata": meta,
                            "similarity_score": round(1 - float(dist), 4),
                            "policy_id": (meta or {}).get("policy_id"),
                            "section_title": (meta or {}).get("section_title"),
                            "chunk_type": (meta or {}).get("chunk_type"),
                        }
                    )
            return sims[:n_results]
        except Exception as e:
            logger.error(f"Error finding similar chunks: {e}")
            return []

    def get_policy_context(
        self, rule_description: str, policy_id: Optional[str] = None, n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve top chunks that ground a rule description."""
        try:
            return self.semantic_search(
                query=rule_description,
                n_results=n_results,
                policy_id=policy_id,
                min_relevance_score=0.3,
                chunk_types=["rule_content", "definition", "procedure"],
            )
        except Exception as e:
            logger.error(f"Error getting policy context: {e}")
            return []

    # ---------- Analytics / Maintenance ----------
    def analyze_content_clusters(
        self, policy_id: Optional[str] = None, max_chunks: int = 1000
    ) -> Dict[str, Any]:
        """Basic centroid/outlier analysis on embeddings (if numpy available)."""
        try:
            if not self.is_available():
                raise ServiceUnavailable("vector_store", "Service not initialized")
            if not NUMPY_AVAILABLE:
                return {"error": "NumPy not available"}

            where_clause = {"policy_id": policy_id} if policy_id else None
            res = self.collection.get(
                where=where_clause, limit=max_chunks, include=["documents", "metadatas", "embeddings"]
            )
            if not res.get("embeddings") or len(res["embeddings"]) < 5:
                return {"error": "Insufficient data for clustering analysis"}

            embeddings = np.array(res["embeddings"])
            metas = res["metadatas"]
            centroid = np.mean(embeddings, axis=0)
            dist_center = [np.linalg.norm(emb - centroid) for emb in embeddings]
            mean_d, std_d = float(np.mean(dist_center)), float(np.std(dist_center))
            threshold = mean_d + 2 * std_d

            outliers = []
            for i, d in enumerate(dist_center):
                if float(d) > threshold:
                    outliers.append(
                        {
                            "chunk_index": metas[i].get("chunk_index"),
                            "section_title": metas[i].get("section_title"),
                            "distance_from_center": round(float(d), 4),
                            "content_preview": (res["documents"][i] or "")[:200] + "...",
                        }
                    )

            by_type: Dict[str, int] = {}
            for m in metas:
                t = m.get("chunk_type", "unknown")
                by_type[t] = by_type.get(t, 0) + 1

            return {
                "total_chunks_analyzed": len(embeddings),
                "embedding_dimensions": len(centroid),
                "content_diversity": {
                    "mean_distance_from_center": round(mean_d, 4),
                    "std_distance": round(std_d, 4),
                    "outlier_count": len(outliers),
                    "outliers": outliers[:10],
                },
                "content_distribution": by_type,
                "analysis_metadata": {
                    "embedding_model": self.embedding_model_name,
                    "analysis_timestamp": datetime.now().isoformat(),
                },
            }
        except Exception as e:
            logger.error(f"Error in content cluster analysis: {e}")
            return {"error": str(e)}

    def get_collection_stats(self) -> Dict[str, Any]:
        """Summarize collection contents (sampled)."""
        try:
            if not self.is_available():
                return {"error": "Vector store not available"}

            count = self.collection.count()
            if count == 0:
                return {
                    "total_chunks": 0,
                    "unique_policies": 0,
                    "collection_name": self.collection_name,
                    "embedding_model": self.embedding_model_name,
                    "status": "empty",
                }

            sample_size = min(count, 1000)
            sample = self.collection.get(limit=sample_size, include=["metadatas"])
            if not sample.get("metadatas"):
                return {"error": "No metadata available"}

            policy_ids = set()
            chunk_types: Dict[str, int] = {}
            embedding_models = set()
            earliest: Optional[datetime] = None
            latest: Optional[datetime] = None

            for m in sample["metadatas"]:
                pid = m.get("policy_id")
                if pid:
                    policy_ids.add(pid)
                t = m.get("chunk_type", "unknown")
                chunk_types[t] = chunk_types.get(t, 0) + 1
                em = m.get("embedding_model")
                if em:
                    embedding_models.add(em)
                ca = m.get("created_at")
                if ca:
                    try:
                        dt = datetime.fromisoformat(ca.replace("Z", ""))
                        earliest = dt if earliest is None or dt < earliest else earliest
                        latest = dt if latest is None or dt > latest else latest
                    except Exception:
                        pass

            return {
                "total_chunks": count,
                "unique_policies": len(policy_ids),
                "sample_size": sample_size,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model_name,
                "persist_directory": self.persist_directory,
                "chunk_type_distribution": chunk_types,
                "embedding_models_used": list(embedding_models),
                "date_range": {
                    "earliest": earliest.isoformat() if earliest else None,
                    "latest": latest.isoformat() if latest else None,
                },
                "status": "healthy",
                "last_updated": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

    def optimize_collection(self) -> Dict[str, Any]:
        """Remove duplicate chunks by content hash."""
        try:
            if not self.is_available():
                return {"error": "Vector store not available"}

            all_res = self.collection.get(include=["documents", "metadatas"])
            docs = all_res.get("documents") or []
            metas = all_res.get("metadatas") or []
            ids = all_res.get("ids") or []
            if not docs:
                return {"message": "No documents to optimize"}

            seen: Dict[str, Dict[str, Any]] = {}
            dups: List[str] = []

            for i, (doc, meta) in enumerate(zip(docs, metas)):
                h = (meta or {}).get("content_hash") or hashlib.md5((doc or "").encode()).hexdigest()
                if h in seen:
                    dups.append(ids[i])
                else:
                    seen[h] = {"id": ids[i], "idx": i}

            removed = 0
            if dups:
                self.collection.delete(ids=dups)
                removed = len(dups)
                logger.info(f"Removed {removed} duplicate chunks")

            return {
                "optimization_completed": True,
                "duplicates_removed": removed,
                "total_chunks_after": self.collection.count(),
                "optimization_timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error optimizing collection: {e}")
            return {"error": str(e)}

    def backup_collection(self, backup_path: Optional[str] = None) -> Dict[str, Any]:
        """Export full collection to a JSON file for backup."""
        try:
            if not self.is_available():
                return {"error": "Vector store not available"}

            if not backup_path:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"./backups/vector_store_backup_{ts}"
            os.makedirs(backup_path, exist_ok=True)

            all_data = self.collection.get(include=["documents", "metadatas", "embeddings"])
            backup = {
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model_name,
                "backup_timestamp": datetime.now().isoformat(),
                "total_chunks": len(all_data.get("ids") or []),
                "data": {
                    "ids": all_data.get("ids"),
                    "documents": all_data.get("documents"),
                    "metadatas": all_data.get("metadatas"),
                    "embeddings": all_data.get("embeddings"),
                },
            }
            outfile = os.path.join(backup_path, "collection_backup.json")
            with open(outfile, "w", encoding="utf-8") as f:
                json.dump(backup, f, indent=2, default=str)

            logger.info(f"Vector store backup completed: {outfile}")
            return {
                "backup_completed": True,
                "backup_path": outfile,
                "chunks_backed_up": backup["total_chunks"],
                "backup_timestamp": backup["backup_timestamp"],
            }
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return {"error": str(e)}
