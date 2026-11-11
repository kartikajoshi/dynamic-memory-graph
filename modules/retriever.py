"""
Retriever (owner: Kartika).

Finds the stored facts most relevant to a query using three signals fused
with Reciprocal Rank Fusion (RRF):

    1. BM25 keyword search  (rank-bm25)
    2. Dense embedding search  (sentence-transformers + FAISS)
    3. Graph expansion  (multi-hop walk from mentioned entities)

main.py keeps calling retrieve(), so nothing else has to change.

Usage (from the project root):
    from modules.retriever import retrieve
    facts = retrieve("What medicine do I take?", store)

Quick test (no Ollama needed):
    python -m modules.retriever
"""

import re
import logging

import numpy as np
import faiss
from rank_bm25 import BM25Plus
from sentence_transformers import SentenceTransformer

from config import (
    BM25_TOP_K,
    DENSE_TOP_K,
    EMBEDDING_MODEL,
    GRAPH_TOP_K,
    MAX_HOPS,
    RRF_K,
    TOP_K,
    USER_NODE,
)

logger = logging.getLogger(__name__)

# Lazy-loaded embedding model (loaded on first call to avoid slow imports at startup).
_model = None


def _get_model():
    """Load the sentence-transformer model once and cache it."""
    global _model
    if _model is None:
        logger.info("Loading embedding model %s ...", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "at", "by",
    "is", "are", "was", "were", "be", "been", "am", "do", "does", "did",
    "i", "me", "my", "mine", "you", "your", "what", "which", "who", "whom",
    "when", "where", "why", "how", "can", "could", "should", "would", "will",
    "about", "any", "some", "tell", "know", "please", "it", "its", "this", "that",
}


# ---------- Tokeniser (shared by BM25 and entity extraction) ----------

def tokenize(text):
    """Lowercase, split into words, drop stopwords and strip a plural 's'."""
    words = re.findall(r"[a-z0-9]+", str(text).lower().replace("_", " "))
    tokens = []
    for word in words:
        if word in STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.append(word)
    return tokens


def fact_to_text(fact):
    """Build the searchable text of a fact, including its entity type."""
    object_type = (fact.get("object_type") or "").lower()
    return f"{fact.get('text', '')} {object_type}".strip()


# ---------- BM25 ----------

def bm25_search(query, facts, top_k=BM25_TOP_K):
    """Rank facts by BM25 score; only facts sharing at least one word with the query are kept."""
    if not facts:
        return []
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    corpus = [tokenize(fact_to_text(fact)) for fact in facts]
    scores = BM25Plus(corpus).get_scores(query_tokens)

    query_set = set(query_tokens)
    ranked = [
        (fact, float(score))
        for fact, tokens, score in zip(facts, corpus, scores)
        if query_set & set(tokens)
    ]
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


# ---------- Dense (FAISS) ----------

def dense_search(query, facts, top_k=DENSE_TOP_K):
    """Rank facts by cosine similarity using sentence-transformers and FAISS."""
    if not facts:
        return []

    model = _get_model()
    texts = [fact_to_text(f) for f in facts]

    fact_embeddings = model.encode(texts, normalize_embeddings=True).astype("float32")
    query_embedding = model.encode([query], normalize_embeddings=True).astype("float32")

    dim = fact_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(fact_embeddings)

    k = min(top_k, len(facts))
    scores, indices = index.search(query_embedding, k)

    ranked = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        ranked.append((facts[idx], float(score)))
    return ranked


# ---------- Graph expansion ----------

def _extract_entities(query, facts):
    """Find entity names from facts that appear in the query (simple substring match)."""
    query_lower = query.lower()
    entities = set()
    # Always include User since queries are about the user.
    entities.add(USER_NODE)
    for fact in facts:
        for field in ("subject", "object"):
            name = fact.get(field, "")
            if name and name.lower() in query_lower:
                entities.add(name)
    return entities


def graph_search(query, graph_store, facts, top_k=GRAPH_TOP_K):
    """Expand from entities mentioned in the query and rerank by recency."""
    seed_entities = _extract_entities(query, facts)
    if not seed_entities:
        return []

    expanded = graph_store.expand(seed_entities, max_hops=MAX_HOPS, active_only=True)
    reranked = graph_store.rerank(expanded, top_k=top_k)
    return [(fact, 1.0) for fact in reranked]


# ---------- Reciprocal Rank Fusion ----------

def rrf_fuse(ranked_lists, k=RRF_K):
    """Merge multiple ranked lists using RRF. Each list is [(fact, score), ...]."""
    scores = {}
    fact_map = {}

    for ranked in ranked_lists:
        for rank, (fact, _score) in enumerate(ranked, start=1):
            fact_id = (fact.get("source", ""), fact.get("target", ""), fact.get("key", ""))
            fact_map[fact_id] = fact
            scores[fact_id] = scores.get(fact_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(fact_map[fid], score) for fid, score in fused]


# ---------- Main entry point ----------

def retrieve(query, graph_store, top_k=TOP_K):
    """Return the top_k active facts from the graph most relevant to the query."""
    facts = graph_store.get_all_facts(active_only=True)
    if not facts:
        return []

    bm25_results = bm25_search(query, facts, top_k=BM25_TOP_K)
    dense_results = dense_search(query, facts, top_k=DENSE_TOP_K)
    graph_results = graph_search(query, graph_store, facts, top_k=GRAPH_TOP_K)

    fused = rrf_fuse([bm25_results, dense_results, graph_results], k=RRF_K)

    logger.info("Retrieval: %d BM25, %d dense, %d graph -> %d fused",
                len(bm25_results), len(dense_results), len(graph_results), len(fused))

    return [fact for fact, _ in fused[:top_k]]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = [
        {"text": "User has Diabetes", "object_type": "CONDITION",
         "subject": "User", "object": "Diabetes", "source": "user", "target": "diabetes", "key": 0},
        {"text": "User takes Metformin", "object_type": "DRUG",
         "subject": "User", "object": "Metformin", "source": "user", "target": "metformin", "key": 0},
        {"text": "User lives in Bangalore", "object_type": "PLACE",
         "subject": "User", "object": "Bangalore", "source": "user", "target": "bangalore", "key": 0},
        {"text": "User likes Filter Coffee", "object_type": "PREFERENCE",
         "subject": "User", "object": "Filter Coffee", "source": "user", "target": "filter coffee", "key": 0},
    ]
    for q in ["What drugs do I take?", "Where do I live?", "Any conditions I have?"]:
        print(q)
        bm25 = bm25_search(q, sample, top_k=3)
        dense = dense_search(q, sample, top_k=3)
        fused = rrf_fuse([bm25, dense])
        for fact, score in fused[:3]:
            print(f"    {score:.4f}  {fact['text']}")
        print()
