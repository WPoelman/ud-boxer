from typing import List

from synse.graph import BaseGraph
from synse.sbn import SBN_EDGE_TYPE, SBN_NODE_TYPE, SBNGraph
from synse.ud import UD_EDGE_TYPE, UD_NODE_TYPE, UDGraph
from synse.ud.ud_spec import UDSpecBasic


class GraphTransformer:
    @staticmethod
    def transform(G: BaseGraph) -> BaseGraph:
        raise NotImplemented


class NodeRemover(GraphTransformer):
    DEP_RELS_TO_REMOVE = {
        UDSpecBasic.DepRels.DET,
        UDSpecBasic.DepRels.PUNCT,
    }

    @staticmethod
    def transform(G: BaseGraph) -> BaseGraph:
        node_ids_to_remove, edge_ids_to_remove = set(), set()
        for edge_id, edge_data in G.edges.items():
            _, to_node_id = edge_id
            if edge_data["deprel"] in NodeRemover.DEP_RELS_TO_REMOVE:
                node_ids_to_remove.add(to_node_id)
                edge_ids_to_remove.add(edge_id)

        G.remove_nodes_from(node_ids_to_remove)
        G.remove_edges_from(edge_ids_to_remove)
        return G


# Misschien NameResolver (combine names consisting of > 1 tokens)
class POSResolver(GraphTransformer):
    @staticmethod
    def transform(G: BaseGraph) -> BaseGraph:
        # rename_mapping = dict()
        nodes_to_add, edges_to_add = [], []

        for node_id, node_data in G.nodes.items():
            # Expand PROPN into 2 nodes with an edge
            if node_data["upos"] == UDSpecBasic.POS.PROPN:
                new_node_id = (SBN_NODE_TYPE.NAME_CONSTANT, 1)
                nodes_to_add.append(
                    (
                        new_node_id,
                        {
                            "type": UD_NODE_TYPE.TOKEN,
                            "token": node_data["token"],
                        },
                    )
                )
                edges_to_add.append(
                    (
                        node_id,
                        new_node_id,
                        {
                            "type": UD_EDGE_TYPE.DEPENDENCY_RELATION,
                            "token": "Name",
                        },  # TODO: get roles from spec
                    )
                )

        G.add_nodes_from(nodes_to_add)
        G.add_edges_from(edges_to_add)

        return G


class GraphModifier:
    def __init__(self, transformations: List[GraphTransformer] = None) -> None:
        self.transformations = transformations or []

    # TODO: Add option to provide existing UDGraph and SBNGraph to extract
    # mappings from
    def extract_mappings(self, U: UDGraph, S: SBNGraph) -> SBNGraph:
        pass

    def transform(self, G: BaseGraph) -> BaseGraph:
        for transformation in self.transformations:
            G = transformation.transform(G)
        return G
