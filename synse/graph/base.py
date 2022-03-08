import json
from os import PathLike
from typing import Any, Dict, List, Tuple

import networkx as nx

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
        """Style per node abd/or edge type to use in dot export"""
        raise NotImplementedError("Cannot be called directly.")

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

    def to_png(self, save_path):
        """Creates a dot graph png and saves it at the provided path"""
        # This is possible, but it's a pain to select the proper labels and
        # format. It's easier to create it 'manually'.
        # P = nx.drawing.nx_pydot.to_pydot(self)
        import pydot

        p_graph = pydot.Dot(save_path)

        if type(save_path) != str:
            # pydot does not like a Path object
            save_path = str(save_path)

        token_count: Dict[str, int] = dict()
        node_dict = dict()
        for node_id, node_data in self.nodes.items():
            # Need to do some trickery so no duplicate nodes get added, for
            # example when a sense occurs > 1 times. Example:
            # pmb-4.0.0/data/en/bronze/p00/d0075
            # The tuple ids themselves are not great here.
            tok = node_data["token"].replace(":", "-")
            if tok in token_count:
                token_count[tok] += 1
                token_id = f"{tok}-{token_count[tok]}"
            else:
                token_id = tok
                token_count[tok] = 0
            node_dict[node_id] = token_id

            label = [tok]

            if lemma := node_data.get("lemma"):
                label.append(lemma)
            if upos := node_data.get("upos"):
                label.append(upos)
            if xpos := node_data.get("xpos"):
                label.append(xpos)

            p_graph.add_node(
                pydot.Node(
                    token_id,
                    **{
                        **self.type_style_mapping[node_data["type"]],
                        "label": "\n".join(label),
                    },
                )
            )

        for (from_id, to_id), edge_data in self.edges.items():
            p_graph.add_edge(
                pydot.Edge(
                    node_dict[from_id],
                    node_dict[to_id],
                    **{
                        "label": edge_data["token"].replace(":", "-"),
                        **self.type_style_mapping[edge_data["type"]],
                    },
                )
            )
        p_graph.write(save_path, format="png")

        del p_graph

        return self


class GraphTransformer:
    @staticmethod
    def transform(G: BaseGraph) -> BaseGraph:
        raise NotImplemented


class NodeRemover(GraphTransformer):
    @staticmethod
    def transform(G: BaseGraph) -> BaseGraph:
        for node_id, data in G.nodes.items():
            pass
        return G


class GraphModifier:
    def __init__(self, transformations: List[GraphTransformer] = None) -> None:
        self.transformations = transformations or []

    def transform(self, G: BaseGraph) -> BaseGraph:
        for transformation in self.transformations:
            G = transformation.transform(G)
        return G
