"""
Graph store (owner: Arpit).

NetworkX-backed storage for the memory graph. Entities are nodes and facts
are directed edges (subject -> object) labelled with a predicate. The graph
is saved to data/graph.json with networkx node_link_data / node_link_graph.

This version covers the basics: add nodes/edges, look up facts, save, load.
expand() (multi-hop walks) and rerank() (recency) are added on Oct 31.

Usage (from the project root):
    from modules.graph_store import GraphStore
    store = GraphStore().load()
    store.add_edge("User", "takes", "Metformin", obj_type="DRUG", created_at="2026-10-28T10:00:00")
    store.save()

Quick test:
    python -m modules.graph_store
"""

import json
import os

import networkx as nx
from networkx.readwrite import json_graph

from config import GRAPH_PATH, USER_NODE


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
    store.add_edge("User", "takes", "Metformin", obj_type="DRUG", created_at="2026-10-23T10:00:00")
    store.save()

    reloaded = GraphStore(test_path).load()
    print("Stats:", reloaded.stats())
    for fact in reloaded.get_all_facts():
        print("   ", fact["text"])
    print("Has 'User takes Metformin':", reloaded.has_edge("user", "takes", "metformin"))
