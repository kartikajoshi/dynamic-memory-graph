"""
Graph visualization helper (owner: Arpit).

Generates a visual representation of the memory graph using NetworkX and
matplotlib. Saves the output as a PNG image for use in the README and
presentations.

Usage (from the project root):
    from modules.graph_viz import visualize_graph
    visualize_graph(store, "screenshots/graph.png")

Quick test:
    python -m modules.graph_viz
"""

import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving to file.
import matplotlib.pyplot as plt
import networkx as nx

from config import GRAPH_PATH, USER_NODE


def visualize_graph(graph_store, output_path="screenshots/graph.png",
                    active_only=True, figsize=(14, 10)):
    """Draw the memory graph and save it as a PNG image.

    Nodes are colour-coded by type, edges are labelled with predicates,
    and ended (inactive) edges are drawn as dashed grey lines.
    """
    graph = graph_store.graph
    if graph.number_of_nodes() == 0:
        print("Graph is empty, nothing to visualize.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("Dynamic Memory Graph", fontsize=16, fontweight="bold", pad=20)

    # Layout.
    pos = nx.spring_layout(graph, seed=42, k=2.0)

    # Colour nodes by type.
    type_colors = {
        "PERSON": "#4CAF50",
        "CONDITION": "#F44336",
        "DRUG": "#2196F3",
        "PLACE": "#FF9800",
        "PREFERENCE": "#9C27B0",
        None: "#607D8B",
    }

    node_colors = []
    node_labels = {}
    for node in graph.nodes():
        node_data = graph.nodes[node]
        node_type = node_data.get("type")
        node_colors.append(type_colors.get(node_type, type_colors[None]))
        label = node_data.get("label", node)
        node_labels[node] = label

    # Draw nodes.
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors,
                           node_size=1500, alpha=0.9, edgecolors="white", linewidths=2)
    nx.draw_networkx_labels(graph, pos, labels=node_labels, ax=ax,
                            font_size=9, font_weight="bold")

    # Separate active and inactive edges.
    active_edges = []
    inactive_edges = []
    edge_labels_active = {}
    edge_labels_inactive = {}

    for u, v, k, data in graph.edges(keys=True, data=True):
        predicate = data.get("predicate", "")
        if data.get("ended_at"):
            inactive_edges.append((u, v))
            edge_labels_inactive[(u, v)] = predicate.replace("_", " ")
        else:
            active_edges.append((u, v))
            edge_labels_active[(u, v)] = predicate.replace("_", " ")

    # Draw active edges (solid).
    if active_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=active_edges, ax=ax,
                               edge_color="#333333", width=1.5, alpha=0.8,
                               arrows=True, arrowsize=15,
                               connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels_active, ax=ax,
                                     font_size=7, font_color="#333333")

    # Draw inactive edges (dashed grey).
    if inactive_edges and not active_only:
        nx.draw_networkx_edges(graph, pos, edgelist=inactive_edges, ax=ax,
                               edge_color="#BDBDBD", width=1.0, alpha=0.5,
                               style="dashed", arrows=True, arrowsize=10,
                               connectionstyle="arc3,rad=0.1")

    # Legend.
    legend_items = []
    for type_name, color in type_colors.items():
        if type_name is None:
            type_name = "Other"
        legend_items.append(plt.Line2D([0], [0], marker="o", color="w",
                                       markerfacecolor=color, markersize=10,
                                       label=type_name))
    ax.legend(handles=legend_items, loc="upper left", fontsize=9,
              title="Entity types", title_fontsize=10)

    ax.axis("off")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Graph saved to {output_path}")


if __name__ == "__main__":
    from modules.graph_store import GraphStore
    import tempfile

    # Build a sample graph for testing.
    tmp_path = os.path.join(tempfile.gettempdir(), "dmg_viz_test.json")
    store = GraphStore(tmp_path)
    store.add_edge("User", "has", "Diabetes", obj_type="CONDITION", created_at="2026-10-23T10:00:00")
    store.add_edge("User", "takes", "Metformin", obj_type="DRUG", created_at="2026-10-23T10:00:00")
    store.add_edge("User", "lives_in", "Bangalore", obj_type="PLACE", created_at="2026-10-24T09:00:00")
    store.add_edge("User", "likes", "Filter Coffee", obj_type="PREFERENCE", created_at="2026-10-24T09:00:00")
    store.add_edge("User", "has_sister", "Priya", obj_type="PERSON", created_at="2026-10-25T08:00:00")
    store.add_edge("Priya", "lives_in", "Pune", obj_type="PLACE", created_at="2026-10-25T08:05:00")

    output = "screenshots/graph.png"
    visualize_graph(store, output)
    print(f"Test graph visualization saved to {output}")
