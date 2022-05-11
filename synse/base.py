from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx

from synse.misc import ensure_ext

__all__ = [
    "_ID",
    "NODE",
    "EDGE",
    "BaseEnum",
    "BaseGraph",
]


class BaseEnum(str, Enum):
    @classmethod
    def all_values(cls) -> List[str]:
        return [i for i in cls]

    def __str__(self):
        return str(self.value)


_ID = Tuple[str, int]
NODE = Tuple[_ID, Dict[str, Any]]
EDGE = Tuple[_ID, _ID, Dict[str, Any]]


class BaseGraph(nx.DiGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    def from_string(self, input_text: str):
        """Method to construct nodes and edges from input text"""
        raise NotImplementedError("Cannot be called directly.")

    @property
    def type_style_mapping(self):
        """Style per node and/or edge type to use in dot export"""
        raise NotImplementedError("Cannot be called directly.")

    @staticmethod
    def _node_label(node_data) -> str:
        raise NotImplementedError("Overwrite this to create a node label.")

    @staticmethod
    def _edge_label(edge_data) -> str:
        raise NotImplementedError("Overwrite this to create an edge label.")

    def to_png(self, save_path):
        """Creates a dot graph png and saves it at the provided path"""
        # This is possible, but it's a pain to select the proper labels and
        # format. It's easier to create it 'manually'.
        # P = nx.drawing.nx_pydot.to_pydot(self)
        import pydot

        p_graph = pydot.Dot()

        token_count: Dict[str, int] = dict()
        node_dict = dict()
        for node_id, node_data in self.nodes.items():
            # Need to do some trickery so no duplicate nodes get added, for
            # example when a sense occurs > 1 times. Example:
            # pmb-4.0.0/data/en/bronze/p00/d0075
            # The tuple ids themselves are not great here.
            tok = node_data["token"]
            if tok in token_count:
                token_count[tok] += 1
                token_id = f"{tok}-{token_count[tok]}"
            else:
                token_id = f"{tok}"
                token_count[tok] = 0
            node_dict[node_id] = token_id

            p_graph.add_node(
                pydot.Node(
                    token_id,
                    **{
                        "label": f'{self._node_label(node_data).replace(":", "-")}',
                        **self.type_style_mapping[node_data["type"]],
                    },
                )
            )

        for (from_id, to_id), edge_data in self.edges.items():
            p_graph.add_edge(
                pydot.Edge(
                    node_dict[from_id],
                    node_dict[to_id],
                    **{
                        "label": f'{self._edge_label(edge_data).replace(":", "-")}',
                        **self.type_style_mapping[edge_data["type"]],
                    },
                )
            )

        final_path = ensure_ext(save_path, ".png")

        # pydot does not like a Path object
        p_graph.write(str(final_path.resolve()), format="png")

        del p_graph

        return self
