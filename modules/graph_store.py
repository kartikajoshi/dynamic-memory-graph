"""
Graph store (owner: Arpit).

NetworkX-backed storage for the memory graph. Entities are nodes and facts
are directed edges (subject -> object) labelled with a predicate. The graph
is saved to data/graph.json with networkx node_link_data / node_link_graph.

Oct 31 update: added expand() for multi-hop graph walks and rerank() for
recency-based re-ranking of retrieved facts.

Usage (from the project root):
    from modules.graph_store import GraphStore
    store = GraphStore().load()
    store.add_edge("User", "takes", "Metformin", obj_type="DRUG", created_at="2026-10-28T10:00:00")
    store.save()

    # Multi-hop expansion from a set of entities
    expanded = store.expand({"user", "metformin"}, max_hops=2)

    # Recency rerank
    ranked = store.rerank(expanded)

Quick test:
    python -m modules.graph_store
"""

import json
import os
from datetime import datetime

import networkx as nx
from networkx.readwrite import json_graph

from config import GRAPH_PATH, GRAPH_TOP_K, MAX_HOPS, USER_NODE


def _key(name):
    """Turn an entity name into a node id: lowercase with single spaces."""
    return " ".join(str(name).strip().lower().split())


def _to_json(graph):
    """Convert a graph to a JSON-ready dict (works on old and new NetworkX)."""
    try:
        return json_graph.node_link_data(graph, edges="links")
    except TypeError:
        return json_graph.node_link_data(graph)


def _from_json(data):
    """Rebuild a MultiDiGraph from a node_link dict (works on old and new NetworkX)."""
    try:
        return json_graph.node_link_graph(data, directed=True, multigraph=True, edges="links")
    except TypeError:
        return json_graph.node_link_graph(data, directed=True, multigraph=True)


class GraphStore:
    """Persistent knowledge graph of the user's facts."""

    def __init__(self, path=GRAPH_PATH):
        """Create an empty graph that will be saved to `path`."""
        self.path = path
        self.graph = nx.MultiDiGraph()

    # ---------- Nodes ----------

    def add_node(self, name, node_type=None):
        """Add an entity node (or fill in its type if it already exists); return its id."""
        key = _key(name)
        if not key:
            return None
        if key in self.graph:
            if node_type and not self.graph.nodes[key].get("type"):
                self.graph.nodes[key]["type"] = node_type
        else:
            self.graph.add_node(key, label=str(name).strip(), type=node_type)
        return key

    def has_node(self, name):
        """Return True if an entity with this name exists."""
        return _key(name) in self.graph

    def get_label(self, node_id):
        """Return the display name of a node."""
        return self.graph.nodes[node_id].get("label", node_id)

    # ---------- Edges ----------

    def add_edge(self, subject, predicate, obj, obj_type=None, created_at=None, **attrs):
        """Add a fact edge subject -[predicate]-> obj and return its (source, target, key)."""
        subject_type = "PERSON" if _key(subject) == _key(USER_NODE) else None
        source = self.add_node(subject, subject_type)
        target = self.add_node(obj, obj_type)
        if source is None or target is None:
            return None
        edge_key = self.graph.add_edge(
            source,
            target,
            predicate=predicate,
            created_at=created_at,
            ended_at=None,
            **attrs,
        )
        return (source, target, edge_key)

    def has_edge(self, subject, predicate, obj, active_only=True):
        """Return True if the fact subject -[predicate]-> obj is already stored."""
        return len(self.find_edges(subject, predicate, obj, active_only)) > 0

    def find_edges(self, subject=None, predicate=None, obj=None, active_only=True):
        """Return facts matching any combination of subject / predicate / object."""
        source = _key(subject) if subject is not None else None
        target = _key(obj) if obj is not None else None
        results = []
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            if source is not None and u != source:
                continue
            if target is not None and v != target:
                continue
            if predicate is not None and data.get("predicate") != predicate:
                continue
            if active_only and data.get("ended_at"):
                continue
            results.append(self._edge_to_fact(u, v, k, data))
        return results

    def get_all_facts(self, active_only=True):
        """Return every stored fact as a list of dicts."""
        return self.find_edges(active_only=active_only)

    def _edge_to_fact(self, u, v, k, data):
        """Convert one edge into a fact dict used by the rest of the pipeline."""
        subject = self.get_label(u)
        obj = self.get_label(v)
        predicate = data.get("predicate", "")
        return {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "object_type": self.graph.nodes[v].get("type"),
            "created_at": data.get("created_at"),
            "ended_at": data.get("ended_at"),
            "text": f"{subject} {predicate.replace('_', ' ')} {obj}",
            "source": u,
            "target": v,
            "key": k,
        }

    # ---------- Multi-hop expansion ----------

    def expand(self, seed_entities, max_hops=MAX_HOPS, active_only=True):
        """Walk up to max_hops from seed entities and return all reachable active facts.

        seed_entities: a set of entity names (strings) to start from.
        Returns a list of fact dicts (same format as get_all_facts).
        Deduplicates by (source, target, key).
        """
        frontier = set()
        for entity in seed_entities:
            key = _key(entity)
            if key in self.graph:
                frontier.add(key)

        if not frontier:
            return []

        visited_nodes = set(frontier)
        collected = {}

        for _hop in range(max_hops):
            next_frontier = set()
            for node in frontier:
                for u, v, k, data in self.graph.out_edges(node, keys=True, data=True):
                    if active_only and data.get("ended_at"):
                        continue
                    fact_key = (u, v, k)
                    if fact_key not in collected:
                        collected[fact_key] = self._edge_to_fact(u, v, k, data)
                    if v not in visited_nodes:
                        next_frontier.add(v)

                for u, v, k, data in self.graph.in_edges(node, keys=True, data=True):
                    if active_only and data.get("ended_at"):
                        continue
                    fact_key = (u, v, k)
                    if fact_key not in collected:
                        collected[fact_key] = self._edge_to_fact(u, v, k, data)
                    if u not in visited_nodes:
                        next_frontier.add(u)

            visited_nodes |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

        return list(collected.values())

    # ---------- Recency-based reranking ----------

    def rerank(self, facts, top_k=GRAPH_TOP_K):
        """Rerank facts by recency (newest created_at first) and return the top_k."""
        def _sort_key(fact):
            ts = fact.get("created_at")
            return ts if ts else ""

        sorted_facts = sorted(facts, key=_sort_key, reverse=True)
        return sorted_facts[:top_k]

    # ---------- Housekeeping ----------

    def stats(self):
        """Return node and edge counts."""
        return {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges()}

    def clear(self):
        """Delete everything in the graph (does not touch the file until save())."""
        self.graph = nx.MultiDiGraph()

    # ---------- Persistence ----------

    def save(self, path=None):
        """Write the graph to JSON (default data/graph.json)."""
        path = path or self.path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_to_json(self.graph), f, indent=2)

    def load(self, path=None):
        """Load the graph from JSON if the file exists; return self so calls can be chained."""
        path = path or self.path
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.graph = _from_json(json.load(f))
        return self


if __name__ == "__main__":
    import tempfile

    test_path = os.path.join(tempfile.gettempdir(), "dmg_test_graph.json")
    store = GraphStore(test_path)

    store.add_edge("User", "has", "Diabetes", obj_type="CONDITION", created_at="2026-10-23T10:00:00")
    store.add_edge("User", "takes", "Metformin", obj_type="DRUG", created_at="2026-10-24T09:00:00")
    store.add_edge("User", "lives_in", "Bangalore", obj_type="PLACE", created_at="2026-10-25T08:00:00")
    store.add_edge("User", "has_sister", "Priya", obj_type="PERSON", created_at="2026-10-25T08:00:00")
    store.add_edge("Priya", "lives_in", "Pune", obj_type="PLACE", created_at="2026-10-25T08:05:00")
    store.save()

    reloaded = GraphStore(test_path).load()
    print("Stats:", reloaded.stats())
    print("\nAll facts:")
    for fact in reloaded.get_all_facts():
        print("   ", fact["text"])

    print("\n--- expand() from {'User'}, max_hops=1 ---")
    hop1 = reloaded.expand({"User"}, max_hops=1)
    for fact in hop1:
        print("   ", fact["text"])

    print("\n--- expand() from {'User'}, max_hops=2 ---")
    hop2 = reloaded.expand({"User"}, max_hops=2)
    for fact in hop2:
        print("   ", fact["text"])

    print("\n--- rerank() top 3 ---")
    ranked = reloaded.rerank(hop2, top_k=3)
    for fact in ranked:
        print(f"    {fact['created_at']}  {fact['text']}")
