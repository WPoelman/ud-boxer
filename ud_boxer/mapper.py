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

import logging
from os import PathLike
from pathlib import Path
from typing import Dict, List

import pandas as pd
from networkx.algorithms.isomorphism import DiGraphMatcher

from ud_boxer.rewrite import BoxRemover
from ud_boxer.sbn import SBNGraph
from ud_boxer.ud_spec import UDSpecBasic

logger = logging.getLogger(__name__)

__all__ = [
    "MapExtractor",
]


class MapExtractor:
    def __init__(self) -> None:
        self.edge_mapping_records: List[Dict[str, str]] = []
        self.node_mapping_records: List[Dict[str, str]] = []
        self.relevant_edge_keys = {
            "token",
            "lemma",
            "upos",
            "xpos",
            *UDSpecBasic.Feats.KEYS,
        }
        self.relevant_node_keys = {"token", "lemma", "upos", "xpos"}

    def extract(self, G: SBNGraph, T: SBNGraph, doc_id: str):
        """
        G: source gold graph
        T: target graph to test
        """
        G = BoxRemover.transform(G)
        T = BoxRemover.transform(T)

        matcher = DiGraphMatcher(G, T)
        if matcher.subgraph_is_isomorphic():
            mapping = matcher.mapping

            for g_from_id, g_to_id, g_edge in G.edges.data():
                # The mappings from the DiGraphMatcher only go over the node,
                # it could be possible to figure out the correct edges from the
                # (sub)graph match we're currently looking at, but in that case
                # the id is incorrect resulting in a key error, which accomplishes
                # the same :)
                try:
                    target_from = mapping[g_from_id]
                    target_to = mapping[g_to_id]
                    target_edge = T.edges[target_from, target_to]
                except KeyError:
                    continue

                # altijd de nodes opslaan lemma -> sense etc.
                # van target lemma of lemma pos naar gold sense

                # Only store correct edge mappings
                if target_edge["token"] == g_edge["token"]:
                    # The Grew step introduced an already exact mapping, meaning
                    # we also don't have any additional UD information
                    if g_edge != target_edge:
                        self.edge_mapping_records.append(
                            {
                                **{
                                    f"from_node_{k}": v
                                    for k, v in T.nodes[target_from].items()
                                    if k in self.relevant_edge_keys
                                },
                                **{
                                    f"to_node_{k}": v
                                    for k, v in T.nodes[target_to].items()
                                    if k in self.relevant_edge_keys
                                },
                                "deprel": target_edge.get("deprel"),
                                "label": g_edge["token"],
                                "doc_id": doc_id,
                            }
                        )

                self.node_mapping_records.append(
                    {
                        **{
                            f"gold_node_{k}": v
                            for k, v in G.nodes[g_from_id].items()
                            if k in self.relevant_node_keys
                        },
                        **{
                            f"ud_node_{k}": v
                            for k, v in T.nodes[target_from].items()
                            if k in self.relevant_node_keys
                        },
                        **{
                            f"gold_node_{k}": v
                            for k, v in G.nodes[g_to_id].items()
                            if k in self.relevant_node_keys
                        },
                        **{
                            f"ud_node_{k}": v
                            for k, v in T.nodes[target_to].items()
                            if k in self.relevant_node_keys
                        },
                    }
                )

    def export_csv(self, output_path: PathLike) -> None:
        for item in ["node", "edge"]:
            path = (
                Path(output_path).parent
                / f"{Path(output_path).stem}_{item}.csv"
            )
            pd.DataFrame().from_records(
                getattr(self, f"{item}_mapping_records")
            ).drop_duplicates().to_csv(path, index=False)
