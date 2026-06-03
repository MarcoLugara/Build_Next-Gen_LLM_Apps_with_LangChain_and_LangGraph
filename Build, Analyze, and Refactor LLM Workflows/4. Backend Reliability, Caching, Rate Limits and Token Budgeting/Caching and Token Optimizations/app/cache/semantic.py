# app/cache/semantic.py

import uuid   #to generate unique IDs for ech sematic cache entry
import time   #to implement the LastRecentlyUsed (LRU) logic
from typing import Optional, Tuple
import chromadb   #our vector DB client (replaceable for scalability)
from sentence_transformers import SentenceTransformer  #local embedding model for the sematic cache
from loguru import logger  #structured logging
import redis.asyncio as redis   #used for the LRU-coordination

from app.config import settings


class SemanticCache:
    """Chroma-based semantic cache with LRU eviction via Redis."""

    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name="semantic_cache",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"SemanticCache connected to Chroma at {settings.chroma_persist_directory}")

        self.embedder = SentenceTransformer(settings.embedding_model_name)
        logger.info(f"Embedding model loaded: {settings.embedding_model_name}")

        self.redis = redis.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            decode_responses=True
        )
        logger.info(f"SemanticCache LRU Redis connection established")

        self._insert_counter = 0

    def _get_embedding(self, text: str) -> list:
        embedding = self.embedder.encode(text).tolist()
        logger.debug(f"Generated embedding of dimension {len(embedding)} for text: {text[:50]}...")
        return embedding

    async def get(self, query: str, context: str = "") -> Tuple[Optional[str], Optional[str], Optional[float]]:
        full_text = f"{context}|{query}"
        query_embedding = self._get_embedding(full_text)

        #results will be
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["metadatas", "distances"]
        )

        if results['ids'] and results['distances'][0]:
            distance = results['distances'][0][0]
            similarity = 1 - distance

            if similarity >= settings.similarity_threshold:
                metadata = results['metadatas'][0][0]
                response = metadata['response']
                cached_query = metadata['query']

                doc_id = results['ids'][0][0]
                await self.redis.zadd("semantic_cache:lru", {doc_id: time.time()})

                logger.info(f"Semantic cache HIT (similarity={similarity:.3f})")
                return response, cached_query, similarity

        logger.debug(f"Semantic cache MISS")
        return None, None, None

    async def add(self, query: str, context: str, response: str) -> None:
        full_text = f"{context}|{query}"
        embedding = self._get_embedding(full_text)
        doc_id = str(uuid.uuid4())

        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[{
                "query": full_text,
                "response": response,
                "created_at": time.time()
            }]
        )

        await self.redis.zadd("semantic_cache:lru", {doc_id: time.time()})

        self._insert_counter += 1
        if self._insert_counter >= settings.semantic_cache_lru_check_frequency:
            await self._evict_if_needed()
            self._insert_counter = 0

        logger.debug(f"Semantic cache ADD for query: {full_text[:50]}...")

    async def _evict_if_needed(self) -> None:
        current_size = await self.redis.zcard("semantic_cache:lru")

        if current_size > settings.max_semantic_cache_entries:
            to_remove = current_size - settings.max_semantic_cache_entries
            logger.info(f"Semantic cache size {current_size} exceeds limit. Evicting {to_remove} oldest entries.")

            for _ in range(to_remove):
                popped = await self.redis.zpopmin("semantic_cache:lru", count=1)
                if not popped:
                    break
                doc_id, score = popped[0]
                try:
                    self.collection.delete(ids=[doc_id])
                    logger.debug(f"Evicted semantic cache entry {doc_id}")
                except Exception as e:
                    logger.warning(f"Failed to delete {doc_id} from Chroma: {e}")

            logger.info(f"Eviction complete. New size: {await self.redis.zcard('semantic_cache:lru')}")

    async def stats(self) -> dict:
        chroma_count = self.collection.count()
        redis_count = await self.redis.zcard("semantic_cache:lru")

        # Consistency check
        if chroma_count != redis_count:
            logger.warning(
                f"Inconsistency between Chroma ({chroma_count}) and Redis LRU set ({redis_count}). "
                f"Difference: {abs(chroma_count - redis_count)} entries. Consider running repair."
            )

        return {
            "semantic_cache_entries_chroma": chroma_count,
            "semantic_cache_entries_redis": redis_count,
            "max_entries": settings.max_semantic_cache_entries,
            "similarity_threshold": settings.similarity_threshold,
            "consistent": chroma_count == redis_count,  # optional flag
        }

    async def close(self):  #Close Redis and Chroma connections
        # 1. Close the Redis connection
        await self.redis.close()
        logger.info("SemanticCache Redis connection closed")

        # 2. Close the Chroma client to flush state and release SQLite locks
        self.chroma_client.close()
        logger.info("SemanticCache Chroma client connection closed")