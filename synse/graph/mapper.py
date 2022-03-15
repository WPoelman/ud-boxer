"""
Possible approach:
    - try to make UD graph a monomorphic (sub)graph of SBN graph by
    removing and adding nodes.
        - add time
        - remove punct, det, etc. nodes
        - combine named entities
    - then learn mappings using the 'mapping' from the networkx matcher
    for the node and edge labels and store those.

    If no monomphism can be found, try to go as far as possible using the
    matcher.candidate_pairs_iter() and try to find relevant mappings there
    and store those as 'silver' mappings.

    After 'training' / extracting the mappings we can recursively try to
    apply the mappings and find new ones using the procedure described above.
    Maybe relevant mappings from other examples help the current one to 
    become a better match (monomorphism) and extract even more mappings.

    This can go on recursively until no new mappings are found, all the
    while storing the counts of how many times a mapping occurred.
"""

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Dict

from networkx.algorithms.isomorphism import DiGraphMatcher

from synse.graph import BaseGraph
from synse.graph.rewrite import BoxRemover, EdgeConnector, NodeRemover, POSResolver
from synse.sbn import SBN_EDGE_TYPE, SBN_NODE_TYPE, SBNGraph
from synse.sbn.sbn import SBN_NODE_TYPE
from synse.sbn.sbn_spec import SUPPORTED_LANGUAGES
from synse.ud import UD_SYSTEM, UDGraph
from synse.ud.ud import UD_NODE_TYPE, Collector


# NOTE: Not used at the moment, currently just looking at structure.
def node_match(ud_node, sbn_node):
    # (
    #   token_u == token_s or
    #   Levenshtein(token_u, token_s) or
    #   lemma_u == lemma_s or
    #   date/time mappings from words to numbers (e.g. data/en/gold/p64/d2903) or
    #   wn_pos == xpos mapped to simple wn pos (verb / aux variants -> v, noun variants / numbers etc -> n etc.)
    # )
    if (
        ud_node["type"] == UD_NODE_TYPE.ROOT
        and sbn_node["type"] == SBN_NODE_TYPE.BOX
    ):
        return True

    if (
        (ud_lemma := ud_node["lemma"])
        and (sbn_lemma := sbn_node.get("wn_lemma"))
        and ud_lemma == sbn_lemma
    ):
        return True

    if ud_node["token"] == sbn_node["token"]:
        return True

    return False


class MapExtractor:
    def __init__(self) -> None:
        # TODO: make edge mapping type
        self.edge_mappings: Dict[str, Dict[str, Dict[str, int]]] = {
            "deprel2role": dict(),
            "deprel2token": dict(),
            "token2token": dict(),
        }
        # This is a good place to see what node / edge types need to be added
        # it's a lot nicer to map based on types + consistent data (lemma) than
        # random stuff.
        self.node_mappings: Dict[str, Dict[str, Dict[str, int]]] = {
            "token2sense": dict(),
            "token2const": dict(),  # maybe add more explicit const types (needs to happen at parse stage as well)
            "token2nameconst": dict(),
            "token2box": dict(),
        }

    # Use I as intermediate representation to do recursive calls, i.e. a graph like accumilator
    def extract_mappings(
        self,
        U: UDGraph,
        S: SBNGraph,
        I: UDGraph = None,
        depth: int = 5,
        count: int = 0,
        debug: bool = False,
    ):

        if count >= depth:
            return I

        if not I:
            I = deepcopy(U)

        # First try to 'disable' the box nodes, leaving only the first box node
        # and box connect edge, to resemble the UD root
        #
        # TODO: another way of doing is to connect the UD nodes to the
        # respective sentence they belong to. Might be better since then CONTINUATION etc.
        # are easier to catch.
        # Not a fan of two mutable graphs currently, ideally only I should be mutable
        # or I should be a 'fresh' graph that gets nodes & edges added to it.
        #
        # Another approach, just for extracting the mappings, could be to apply both
        # transformations and extract possible mappings. This could catch those
        # 'box' relation mappings maybe?
        S = BoxRemover.transform(S, ud_root_lemma=I.root_node()["lemma"])
        # if count == 0:
        I = NodeRemover.transform(I)
        I = POSResolver.transform(I)
        # I = EdgeConnector.transform(I, self.edge_mappings)

        matcher = DiGraphMatcher(I, S)

        # Isomorphic is stronger, but we might get more mappings using monomorphism.
        # At the moment the quality of both is not guaranteed.
        if matcher.subgraph_is_isomorphic():
            self.store_mappings(I, S, matcher.mapping)
        elif matcher.subgraph_is_monomorphic():
            # Maybe check alternatives if there are any:
            # print(list(matcher.subgraph_monomorphisms_iter()))
            self.store_mappings(I, S, matcher.mapping)

        I_n_nodes, S_n_nodes = len(I.nodes), len(S.nodes)
        I_n_edges, S_n_edges = len(I.edges), len(S.edges)

        """
        Scenarios:
            - U has the same number of nodes and edges as S -> check if it's monomorphism and extract / apply mappings
            - U has more nodes than S -> remove / merge nodes until it does or find subgraph
              (within a threshold) (indirectly also deals with edges when nodes are merged)
            - S has more nodes than U -> expand nodes (names/compounds/time?)
            - S has more edges than U -> try stored mappings to add more edges (sbn roles)
        """
        if debug:
            if I_n_nodes == S_n_nodes:
                if I_n_edges == S_n_edges:
                    print(
                        "Same number of nodes and edges, try monomorphism, maybe use mappings if this does not work (i.e. the edges are incorrect)."
                    )
                elif I_n_edges < S_n_edges:
                    print(
                        "Same number of nodes, but not enough edges, try stored mappings to add more"
                    )
                else:
                    print(
                        "Same number of nodes, but more edges, is this even possible?"
                    )
            elif I_n_nodes < S_n_nodes:
                print(
                    "Not enough nodes, try to remove some nodes from sbn (only boxes for now) or expand existing UD nodes"
                )
            else:
                if I_n_edges == S_n_edges:
                    print(
                        "Too many nodes and the same number of edges, is that even possible?"
                    )
                elif I_n_edges < S_n_edges:
                    print(
                        "Too many nodes, but not enough edges, try to remove / merge nodes and (next is implicitly handled if we do it recursively) then try stored mappings to add more"
                    )
                else:
                    print(
                        "Too many nodes and too many edges, remove / merge nodes"
                    )

        S.to_png("intermediate_step_sbn_1.png")
        I.to_png("intermediate_step_ud_1.png")
        return I

    def store_mappings(self, I: UDGraph, S: SBNGraph, mapping: Dict[Any, Any]):
        for ud_from_id, ud_to_id, edge_data in I.edges.data():
            # The mappings from the DiGraphMatcher only go over the node,
            # it could be possible to figure out the correct edges from the
            # (sub)graph match we're currently looking at, but in that case
            # the id is incorrect resulting in a key error, which accomplishes
            # the same :)
            try:
                sbn_edge = S.edges[mapping[ud_from_id], mapping[ud_to_id]]
            except KeyError:
                continue

            deprel = edge_data.get("deprel")
            ud_token = edge_data.get("token")
            sbn_token = sbn_edge.get("token")
            sbn_edge_type = sbn_edge.get("type")

            # TODO: clean this up a bit with a defaultdict or something, for later.
            if deprel and sbn_edge_type == SBN_EDGE_TYPE.ROLE:
                if deprel in self.edge_mappings["deprel2role"]:
                    self.edge_mappings["deprel2role"][deprel][sbn_token] = (
                        self.edge_mappings["deprel2role"][deprel].get(
                            sbn_token, 0
                        )
                        + 1
                    )
                else:
                    self.edge_mappings["deprel2role"][deprel] = {sbn_token: 1}
            elif deprel:
                if deprel in self.edge_mappings["deprel2token"]:
                    self.edge_mappings["deprel2token"][deprel][sbn_token] = (
                        self.edge_mappings["deprel2token"][deprel].get(
                            sbn_token, 0
                        )
                        + 1
                    )
                else:
                    self.edge_mappings["deprel2token"][deprel] = {sbn_token: 1}
            elif ud_token and sbn_token:
                if ud_token in self.edge_mappings["token2token"]:
                    self.edge_mappings["token2token"][ud_token][sbn_token] = (
                        self.edge_mappings["token2token"][ud_token].get(
                            sbn_token, 0
                        )
                        + 1
                    )
                else:
                    self.edge_mappings["token2token"][ud_token] = {
                        sbn_token: 1
                    }
            else:
                print("No token or deprel (?)")

        for ud_node_id, sbn_node_id in mapping.items():
            ud_node_data = I.nodes[ud_node_id]
            sbn_node_data = S.nodes[sbn_node_id]

            ud_node_token = ud_node_data["token"]
            sbn_node_token = sbn_node_data["token"]
            sbn_node_type = sbn_node_data["type"]

            if sbn_node_type == SBN_NODE_TYPE.SENSE:
                key = "token2sense"
            elif sbn_node_type == SBN_NODE_TYPE.CONSTANT:
                key = "token2const"
            elif sbn_node_type == SBN_NODE_TYPE.NAME_CONSTANT:
                key = "token2nameconst"
            elif sbn_node_type == SBN_NODE_TYPE.BOX:
                key = "token2box"
            else:
                print(f"Unknown sbn node type found: {sbn_node_type}")
                continue

            if ud_node_token in self.node_mappings[key]:
                self.node_mappings[key][ud_node_token][sbn_node_token] = (
                    self.node_mappings[key][ud_node_token].get(
                        sbn_node_token, 0
                    )
                    + 1
                )
            else:
                self.node_mappings[key][ud_node_token] = {sbn_node_token: 1}

    def transform(self, G: BaseGraph) -> BaseGraph:
        # apply stored mappings to create SBN graph from UD graph
        # U -> S
        pass