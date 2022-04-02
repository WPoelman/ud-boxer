import logging
from collections import Counter
from typing import List

import networkx as nx

from synse.graph import BaseGraph
from synse.sbn import SBN_EDGE_TYPE, SBN_NODE_TYPE, SBNGraph
from synse.ud import UD_EDGE_TYPE, UD_NODE_TYPE, UDGraph
from synse.ud.ud_spec import UDSpecBasic

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


class NodeRemover(GraphTransformer):
    DEP_RELS_TO_REMOVE = {
        UDSpecBasic.DepRels.DET,
        UDSpecBasic.DepRels.PUNCT,
        UDSpecBasic.DepRels.AUX,
        UDSpecBasic.DepRels.AUX_PASS,
        UDSpecBasic.DepRels.CASE,
        UDSpecBasic.DepRels.EXPL,
    }

    @staticmethod
    def transform(G: BaseGraph, **kwargs) -> BaseGraph:
        node_ids_to_remove, edge_ids_to_remove = set(), set()
        for edge_id, edge_data in G.edges.items():
            if edge_data.get("deprel") in NodeRemover.DEP_RELS_TO_REMOVE:
                _, to_node_id = edge_id
                node_ids_to_remove.add(to_node_id)
                edge_ids_to_remove.add(edge_id)
            # TODO: check if this explicit root is a good idea.
            # Coordinate with removing boxes probably
            elif edge_data.get("type") == UD_EDGE_TYPE.EXPLICIT_ROOT:
                from_node_id, _ = edge_id
                node_ids_to_remove.add(from_node_id)
                edge_ids_to_remove.add(edge_id)

        G.remove_nodes_from(node_ids_to_remove)
        G.remove_edges_from(edge_ids_to_remove)
        return G


class NodeExpander(GraphTransformer):
    MULTI_TOKEN_DEP_RELS = (
        UDSpecBasic.DepRels.FLAT,
        UDSpecBasic.DepRels.FLAT_NAME,
    )

    @staticmethod
    def transform(G: UDGraph, **kwargs) -> BaseGraph:
        nodes_to_add, edges_to_add = [], []
        nodes_to_remove, edges_to_remove = set(), set()
        idx = len(G) + 1

        for node_id, node_data in G.nodes.items():
            if node_id in nodes_to_remove:
                continue

            node_pos = node_data.get("upos")
            # Expand PROPN into 2 nodes with an edge
            if node_pos == UDSpecBasic.POS.PROPN:
                # figure out something better here
                new_node_id = (SBN_NODE_TYPE.CONSTANT, idx)
                idx += 1

                full_name = [node_data["token"]]
                # These are multi word tokens, often names.
                # We need to reconstruct them here into a single node.
                for node_id_decs in nx.descendants(G, node_id):
                    # if (G.nodes[node_id_decs].get("deprel") not in NodeExpander.MULTI_TOKEN_DEP_RELS):
                    # break

                    nodes_to_remove.add(node_id_decs)
                    edges_to_remove.add((node_id, node_id_decs))
                    full_name.append(G.nodes[node_id_decs]["token"])

                nodes_to_add.append(
                    (
                        new_node_id,
                        {
                            "type": UD_NODE_TYPE.TOKEN,
                            "token": " ".join(full_name),
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
            # Try to add the time node
            # TODO: use NER or something else to see if an actual date or time
            # is present in the graph. Also make sure not too time nodes get added.
            if node_data.get("deprel") == UDSpecBasic.DepRels.ROOT:
                # Take 'now' as the baseline time constant
                time_relation = "now"

                # Try the tense of the root first, maybe change later
                if tense := node_data["feats"].get("Tense"):
                    time_relation = TIME_EDGE_MAPPING[tense]
                # Otherwise let the majority of tokens vote
                else:
                    all_tenses = [
                        node_data.get("Tense")
                        for _, node_data in G.nodes.items()
                        if node_data.get("Tense")
                    ]
                    if len(all_tenses) > 0:
                        most_freq, _ = Counter(all_tenses).most_common(1)[0]
                        time_relation = TIME_EDGE_MAPPING[most_freq]

                time_sense_id = (SBN_NODE_TYPE.SENSE, idx)
                idx += 1
                nodes_to_add.append(
                    (
                        time_sense_id,
                        {
                            "type": UD_NODE_TYPE.TOKEN,
                            "token": "time.n.08",  # default time sense
                        },
                    )
                )
                edges_to_add.append(
                    (
                        node_id,
                        time_sense_id,
                        {
                            "type": UD_EDGE_TYPE.DEPENDENCY_RELATION,
                            "token": "Time",  # TODO: get from spec?
                        },
                    )
                )

                # Figure something out here
                time_const_id = (SBN_NODE_TYPE.SENSE, idx)
                idx += 1
                nodes_to_add.append(
                    (
                        time_const_id,
                        {
                            "type": UD_NODE_TYPE.TOKEN,
                            "token": "now",  # TODO: different time stuff here (dates, NER? etc.)
                        },
                    )
                )
                edges_to_add.append(
                    (
                        time_sense_id,
                        time_const_id,
                        {
                            "type": UD_EDGE_TYPE.DEPENDENCY_RELATION,
                            "token": time_relation,
                        },
                    )
                )

        G.add_nodes_from(nodes_to_add)
        G.add_edges_from(edges_to_add)

        G.remove_nodes_from(nodes_to_remove)
        G.remove_edges_from(edges_to_remove)

        return G


# Misschien NameResolver (combine names consisting of > 1 tokens)
class POSResolver(GraphTransformer):
    @staticmethod
    def transform(G: UDGraph, **kwargs) -> BaseGraph:

        # # TODO: create a mapping class / dict that stores transformations which
        # # can be applied later on (create edge abc, delete node x, etc)
        # nodes_to_add, edges_to_add = [], []
        # nodes_to_remove, edges_to_remove = set(), set()

        # for node_id, node_data in G.nodes.items():
        #     if node_id in nodes_to_remove:
        #         continue

        #     if node_data.get("deprel") == UDSpecBasic.DepRels.NMOD_POSS:
        #         # Find the subject of the sentence
        #         subj_node = None
        #         for test_node in G.nodes.data():
        #             _, test_node_data = test_node
        #             if test_node_data.get("deprel") in [
        #                 UDSpecBasic.DepRels.NSUBJ,
        #                 UDSpecBasic.DepRels.NSUBJ_PASS,
        #             ]:
        #                 subj_node = test_node
        #                 break

        #         if subj_node:
        #             subj_node_id, _ = subj_node

        #             nodes_to_remove.add(node_id)
        #             parent = next(G.predecessors(node_id))
        #             edges_to_remove.update(G.out_edges([parent]))
        #             # Not sure if possessive is always the user, might also be
        #             # something else, but that is for the next step once the graph
        #             # structure matches (and we have mappings).
        #             # Probably a NodeLabeler or EdgeLabeler that renames the
        #             # graph by itself? Or with specific rename mappings
        #             # (DiGraphMatcher monomorphism mappings?)
        #             edges_to_add.append(
        #                 (
        #                     parent,
        #                     subj_node_id,
        #                     {
        #                         "type": UD_EDGE_TYPE.DEPENDENCY_RELATION,
        #                         "token": "User",  # TODO: get from spec
        #                     },
        #                 )
        #             )

        # G.add_nodes_from(nodes_to_add)
        # G.add_edges_from(edges_to_add)

        # G.remove_nodes_from(nodes_to_remove)
        # G.remove_edges_from(edges_to_remove)

        return G


class BoxRemover(GraphTransformer):
    @staticmethod
    def transform(G: SBNGraph, **kwargs) -> BaseGraph:
        # NOTE: remove all box nodes and edges, commented out parts kept the
        # starting box and starting box edge connecting to the (possible) root.
        # This was very inconsistent. For now try this.

        # possible_lemma_id = None
        # for node_id, node_data in G.nodes.data():
        #     if "wn_lemma" in node_data and node_data["wn_lemma"] == kwargs.get(
        #         "ud_root_lemma"
        #     ):
        #         possible_lemma_id = node_id
        #         break

        # if not possible_lemma_id:
        #     logger.info(
        #         "No possible lemma match found, removing all box info, just in case"
        #     )

        edges_to_remove = set()
        for edge_id, edge_data in G.edges.items():
            # from_id, to_id = edge_id
            # if to_id == possible_lemma_id or to_id == possible_lemma_id:
            #     continue

            if edge_data["type"] == SBN_EDGE_TYPE.BOX_CONNECT:
                edges_to_remove.add(edge_id)

        nodes_to_remove = set()
        # initial_box_id = (SBN_NODE_TYPE.BOX, 0)
        for node_id, node_data in G.nodes.items():
            # Don't remove the initial box node
            # if node_id == initial_box_id:
            #     continue

            if node_data["type"] == SBN_NODE_TYPE.BOX:
                nodes_to_remove.add(node_id)

        G.remove_edges_from(edges_to_remove)
        G.remove_nodes_from(nodes_to_remove)

        return G


class EdgeConnector:
    @staticmethod
    def transform(G: UDGraph, mappings, **kwargs) -> BaseGraph:
        # use the top 3 (or however many) most frequent candidates per found
        # mapping check if the edge exists in the Graph if there are not enough
        # edges (similar to the POS resolver)
        # If an edge does not exist yet, connect it. Do this for all possible
        # mappings and run the extraction again. This will (hopefully) result
        # in more isomorphic graphs, thus resulting in more mappings etc. etc.
        # The recursion can stop at a certain depth or when no (or few) new
        # mappings are found.
        #
        # For nodes this is probably a lot more 'manual':
        # (expanding name constants, time, combined entities etc,)
        return G
