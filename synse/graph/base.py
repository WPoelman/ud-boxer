import json
from os import PathLike
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx

__all__ = ["BaseGraph"]

_ID = Tuple[str, int]
NODE = Tuple[_ID, Dict[str, Any]]
EDGE = Tuple[_ID, _ID, Dict[str, Any]]


class BaseGraph(nx.DiGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    def from_string(self, input_text: str):
        """Method to construct nodes and edges from input text"""
        return NotImplemented

    def to_json(self, path: PathLike):
        """Export the graph to json and save it at the provided path"""
        json_data = nx.readwrite.node_link_data(self)
        with open(path, "w") as f:
            json.dump(json_data, f)

        return self

    def from_json(self, path: PathLike):
        """Read json from the provided path and construct the graph"""
        with open(path) as f:
            json_data = json.load(f)
        self = nx.readwrite.node_link_graph(json_data)

        return self

    def show(
        self,
        node_label_key: str = "token",
        edge_label_key: str = "token",
        save_path: PathLike = None,
    ):
        node_labels = {
            n: data[node_label_key] for n, data in self.nodes.items()
        }
        edge_labels = {
            n: data[edge_label_key] for n, data in self.edges.items()
        }

        # pos = nx.drawing.nx_pydot.graphviz_layout(U, prog="dot")
        # pos = nx.drawing.nx_pydot.graphviz_layout(U)
        pos = nx.circular_layout(self)
        nx.draw_networkx_labels(self, pos, labels=node_labels)
        nx.draw_networkx_edge_labels(self, pos, edge_labels=edge_labels)
        nx.draw(
            self,
            pos,
            node_size=1500,
            node_color="grey",
            font_size=8,
            font_weight="bold",
        )
        plt.show()
