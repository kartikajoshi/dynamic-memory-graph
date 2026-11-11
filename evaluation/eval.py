"""
Evaluation framework (owner: Kartika).

Measures retrieval quality using Recall@k on the test conversations, and
runs an ablation study comparing BM25-only, dense-only, graph-only and
the full RRF-fused retriever.

Metrics:
    Recall@k  =  |relevant facts in top-k| / |all relevant facts|

Run from the project root (Ollama must be running):
    python -m evaluation.eval

Quick check (skips LLM, uses pre-built graph):
    python -m evaluation.eval --skip-llm
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime

# Add project root to path so imports work when run as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BM25_TOP_K,
    DENSE_TOP_K,
    GRAPH_TOP_K,
    RECALL_K_VALUES,
    RRF_K,
    TEST_CONVERSATIONS_PATH,
    TOP_K,
)
from modules.extractor import extract
from modules.graph_store import GraphStore
from modules.memory_manager import update_graph
from modules.retriever import (
    bm25_search,
    dense_search,
    graph_search,
    retrieve,
    rrf_fuse,
)


def load_test_conversations(path=TEST_CONVERSATIONS_PATH):
    """Load test conversations from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph_from_conversation(turns, skip_llm=False):
    """Play through conversation turns, extracting triples and building a graph."""
    tmp_path = os.path.join(tempfile.gettempdir(), "dmg_eval_graph.json")
    store = GraphStore(tmp_path)

    for i, turn in enumerate(turns):
        if turn.get("query"):
            continue
        message = turn["message"]
        timestamp = f"2026-11-01T{10 + i:02d}:00:00"

        if skip_llm:
            # Build triples directly from expected_triples strings.
            triples = []
            for triple_str in turn.get("expected_triples", []):
                parts = triple_str.split(" ", 2)
                if len(parts) == 3:
                    triples.append({
                        "subject": parts[0],
                        "predicate": parts[1],
                        "object": parts[2],
                        "object_type": "PREFERENCE",
                    })
        else:
            triples = extract(message)

        update_graph(store, triples, timestamp)

    return store


def recall_at_k(retrieved_facts, expected_texts, k):
    """Compute Recall@k: fraction of expected facts found in the top-k retrieved."""
    if not expected_texts:
        return 1.0  # Nothing to find -> perfect recall.

    retrieved_texts = set()
    for fact in retrieved_facts[:k]:
        retrieved_texts.add(fact["text"].lower())

    found = sum(1 for exp in expected_texts if exp.lower() in retrieved_texts)
    return found / len(expected_texts)


def evaluate_retriever(store, query, expected_facts, retriever_fn, k_values):
    """Run one query through a retriever and return recall at each k."""
    results = retriever_fn(query, store)
    return {k: recall_at_k(results, expected_facts, k) for k in k_values}


def retriever_bm25_only(query, store, top_k=TOP_K):
    """BM25-only retrieval (no dense, no graph)."""
    facts = store.get_all_facts(active_only=True)
    return [fact for fact, _ in bm25_search(query, facts, top_k=top_k)]


def retriever_dense_only(query, store, top_k=TOP_K):
    """Dense-only retrieval (no BM25, no graph)."""
    facts = store.get_all_facts(active_only=True)
    return [fact for fact, _ in dense_search(query, facts, top_k=top_k)]


def retriever_graph_only(query, store, top_k=TOP_K):
    """Graph-expansion-only retrieval (no BM25, no dense)."""
    facts = store.get_all_facts(active_only=True)
    return [fact for fact, _ in graph_search(query, store, facts, top_k=top_k)]


def retriever_full(query, store, top_k=TOP_K):
    """Full RRF-fused retrieval."""
    return retrieve(query, store, top_k=top_k)


def run_evaluation(skip_llm=False):
    """Run the full evaluation and print results."""
    conversations = load_test_conversations()
    k_values = RECALL_K_VALUES

    retrievers = {
        "BM25 only": retriever_bm25_only,
        "Dense only": retriever_dense_only,
        "Graph only": retriever_graph_only,
        "Full RRF": retriever_full,
    }

    all_results = {name: {k: [] for k in k_values} for name in retrievers}
    query_count = 0

    for conv in conversations:
        print(f"\n--- {conv['id']}: {conv['description']} ---")
        store = build_graph_from_conversation(conv["turns"], skip_llm=skip_llm)
        print(f"  Graph: {store.stats()}")

        for turn in conv["turns"]:
            if not turn.get("query"):
                continue

            query = turn["message"]
            expected = turn.get("expected_facts", [])
            query_count += 1
            print(f"\n  Q: {query}")
            print(f"  Expected: {expected}")

            for name, fn in retrievers.items():
                scores = evaluate_retriever(store, query, expected,
                                            lambda q, s, _fn=fn: _fn(q, s), k_values)
                for k in k_values:
                    all_results[name][k].append(scores[k])
                top_facts = fn(query, store)[:3]
                print(f"    {name}: R@{k_values[-1]}={scores[k_values[-1]]:.2f}  "
                      f"top: {[f['text'] for f in top_facts]}")

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY ({query_count} queries across {len(conversations)} conversations)")
    print("=" * 70)

    header = f"{'Retriever':<16}" + "".join(f"{'R@' + str(k):>8}" for k in k_values)
    print(header)
    print("-" * len(header))

    for name in retrievers:
        row = f"{name:<16}"
        for k in k_values:
            scores = all_results[name][k]
            avg = sum(scores) / len(scores) if scores else 0.0
            row += f"{avg:>8.3f}"
        print(row)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DMG evaluation")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM extraction, use expected triples directly")
    args = parser.parse_args()

    run_evaluation(skip_llm=args.skip_llm)
