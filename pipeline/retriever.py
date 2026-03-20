"""Hybrid retrieval combining keyword search and semantic similarity.

In production, the semantic path uses Pinecone/Weaviate embeddings
and the keyword path uses BM25 or Elasticsearch. For offline testing,
both paths use term-overlap heuristics against the local knowledge base.

The hybrid approach catches cases where:
- Pure semantic misses exact-match terms ($94.8B, AAPL, Q4 2024)
- Pure keyword misses paraphrased meaning ("revenue growth" vs "sales increased")
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from data.knowledge_base import DOCUMENTS, search, get_by_company

log = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    doc_id: str
    content: str
    source: str
    relevance_score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    total_searched: int
    retrieval_method: str

    @property
    def contents(self) -> list[str]:
        return [c.content for c in self.chunks]

    @property
    def source_ids(self) -> list[str]:
        return [c.doc_id for c in self.chunks]


class HybridRetriever:
    """Combines keyword and semantic retrieval for RAG grounding.
    
    Retrieval strategy:
    1. Keyword search: exact term matching, good for numbers/tickers/names
    2. Semantic search: embedding similarity (simulated with term overlap)
    3. Merge & deduplicate: combine results, remove duplicates, re-rank
    4. Company filter: optionally restrict to a specific company's docs
    """

    def __init__(self, top_k: int = 5, keyword_weight: float = 0.4,
                 semantic_weight: float = 0.6):
        self.top_k = top_k
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight

    def retrieve(self, query: str, company_filter: Optional[str] = None) -> RetrievalResult:
        log.info(f"retrieving for: '{query[:60]}...' company={company_filter or 'all'}")

        # Get candidate documents
        if company_filter:
            candidates = get_by_company(company_filter)
        else:
            candidates = DOCUMENTS

        if not candidates:
            log.warning("no documents found for query")
            return RetrievalResult(
                query=query, chunks=[], total_searched=0,
                retrieval_method="hybrid",
            )

        # Score each document
        query_terms = set(query.lower().split())
        scored = []

        for doc in candidates:
            content_lower = doc["content"].lower()
            content_terms = set(content_lower.split())

            # Keyword score: exact term overlap
            keyword_overlap = len(query_terms & content_terms)
            keyword_score = keyword_overlap / max(len(query_terms), 1)

            # Semantic score: longer shared subsequences (simulated)
            # In production this is cosine similarity of embeddings
            bigrams_q = self._bigrams(query.lower())
            bigrams_d = self._bigrams(content_lower)
            bigram_overlap = len(bigrams_q & bigrams_d)
            semantic_score = bigram_overlap / max(len(bigrams_q), 1)

            # Combined hybrid score
            combined = (
                self.keyword_weight * keyword_score
                + self.semantic_weight * semantic_score
            )

            if combined > 0.01:  # minimum relevance threshold
                scored.append((combined, doc))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:self.top_k]

        chunks = [
            RetrievedChunk(
                doc_id=doc["id"],
                content=doc["content"],
                source=doc["source"],
                relevance_score=round(score, 4),
                metadata=doc.get("metadata", {}),
            )
            for score, doc in top
        ]

        log.info(f"retrieved {len(chunks)} chunks from {len(candidates)} candidates")

        return RetrievalResult(
            query=query, chunks=chunks,
            total_searched=len(candidates),
            retrieval_method="hybrid_keyword_semantic",
        )

    @staticmethod
    def _bigrams(text: str) -> set:
        words = text.split()
        return {(words[i], words[i + 1]) for i in range(len(words) - 1)}
