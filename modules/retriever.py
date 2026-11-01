"""
Retriever (owner: Kartika).

Finds the stored facts most relevant to a query. This version uses BM25
keyword search over the fact texts. RRF fusion with dense (FAISS) and graph
expansion results is added on Nov 1 - main.py keeps calling retrieve(), so
nothing else has to change.

Usage (from the project root):
    from modules.retriever import retrieve
    facts = retrieve("What medicine do I take?", store)

Quick test:
    python -m modules.retriever
"""

import re

from rank_bm25 import BM25Plus

from config import BM25_TOP_K, TOP_K

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "at", "by",
    "is", "are", "was", "were", "be", "been", "am", "do", "does", "did",
    "i", "me", "my", "mine", "you", "your", "what", "which", "who", "whom",
    "when", "where", "why", "how", "can", "could", "should", "would", "will",
    "about", "any", "some", "tell", "know", "please", "it", "its", "this", "that",
}


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


def retrieve(query, graph_store, top_k=TOP_K):
    """Return the top_k active facts from the graph most relevant to the query."""
    facts = graph_store.get_all_facts(active_only=True)
    return [fact for fact, _ in bm25_search(query, facts, top_k=top_k)]


if __name__ == "__main__":
    sample = [
        {"text": "User has Diabetes", "object_type": "CONDITION"},
        {"text": "User takes Metformin", "object_type": "DRUG"},
        {"text": "User lives in Bangalore", "object_type": "PLACE"},
        {"text": "User likes Filter Coffee", "object_type": "PREFERENCE"},
    ]
    for q in ["What drugs do I take?", "Where do I live?", "Any conditions I have?"]:
        print(q)
        for fact, score in bm25_search(q, sample, top_k=3):
            print(f"    {score:.2f}  {fact['text']}")
