import logging

from synse.base import BaseGraph
from synse.sbn import SBN_EDGE_TYPE, SBN_NODE_TYPE, SBNGraph
from synse.ud_spec import UDSpecBasic

logger = logging.getLogger(__name__)

TIME_EDGE_MAPPING = {
    UDSpecBasic.Feats.Tense.FUT: "TSU",
    UDSpecBasic.Feats.Tense.IMP: "TPR",  # Not in English?
    UDSpecBasic.Feats.Tense.PAST: "TPR",
    UDSpecBasic.Feats.Tense.PQP: "TPR",  # Not in English
    UDSpecBasic.Feats.Tense.PRES: "EQU",  # or TIN (it's still happening)
}

# Possible useful to extract Role mappings from:
#   - https://universaldependencies.org/u/feat/Degree.html for equality
#   - https://universaldependencies.org/u/feat/Person.html for speaker constant etc.
#   - https://universaldependencies.org/u/feat/Aspect.html for temporal relations, in addition to verb tense noted above
#   - https://universaldependencies.org/u/feat/Mood.html similar to Aspect and Tense


class GraphTransformer:
    @staticmethod
    def transform(G: BaseGraph, **kwargs) -> BaseGraph:
        raise NotImplemented


class BoxRemover(GraphTransformer):
    @staticmethod
    def transform(G: SBNGraph, **kwargs) -> BaseGraph:
        edges_to_remove = {
            edge_id
            for edge_id, edge_data in G.edges.items()
            if edge_data["type"]
            in (SBN_EDGE_TYPE.BOX_CONNECT, SBN_EDGE_TYPE.BOX_BOX_CONNECT)
        }
        nodes_to_remove = {
            node_id
            for node_id, node_data in G.nodes.items()
            if node_data["type"] == SBN_NODE_TYPE.BOX
        }

        G.remove_edges_from(edges_to_remove)
        G.remove_nodes_from(nodes_to_remove)

        return G
