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
from typing import Dict, List

import pandas as pd
from networkx.algorithms.isomorphism import DiGraphMatcher

from synse.rewrite import BoxRemover
from synse.sbn import SBNGraph
from synse.ud_spec import UDSpecBasic

logger = logging.getLogger(__name__)


class MapExtractor:
    def __init__(self) -> None:
        self.mapping_records: List[Dict[str, str]] = []
        self.relevant_keys = {
            "token",
            "lemma",
            "deprel",
            "upos",
            "xpos",
            *UDSpecBasic.Feats.KEYS,
        }

    def extract(self, S: SBNGraph, T: SBNGraph, doc_id: str):
        """
        S: source gold graph
        T: target graph to test
        """
        S = BoxRemover.transform(S)
        T = BoxRemover.transform(T)

        matcher = DiGraphMatcher(S, T)
        if matcher.subgraph_is_isomorphic():
            mapping = matcher.mapping

            for from_id, to_id, source_edge in S.edges.data():
                # The mappings from the DiGraphMatcher only go over the node,
                # it could be possible to figure out the correct edges from the
                # (sub)graph match we're currently looking at, but in that case
                # the id is incorrect resulting in a key error, which accomplishes
                # the same :)
                try:
                    target_from = mapping[from_id]
                    target_to = mapping[to_id]
                    target_edge = T.edges[target_from, target_to]
                except KeyError:
                    continue

                # The Grew step introduced an already exact mapping, meaning
                # we also don't have any additional UD information
                # if source_edge == target_edge:
                #    continue

                self.mapping_records.append(
                    {
                        **{
                            f"from_node_{k}": v
                            for k, v in T.nodes[target_from].items()
                            if k in self.relevant_keys
                        },
                        **{
                            f"to_node_{k}": v
                            for k, v in T.nodes[target_to].items()
                            if k in self.relevant_keys
                        },
                        "label": source_edge["token"],
                        "doc_id": doc_id,
                    }
                )

    def export_csv(self, output_path: PathLike) -> None:
        pd.DataFrame().from_records(
            self.mapping_records
        ).drop_duplicates().to_csv(output_path, index=False)
