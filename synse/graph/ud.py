from os import PathLike
from typing import Any, Dict, Optional, Tuple, Union

from stanza.utils.conll import CoNLL
from synse.graph.base import BaseGraph

__all__ = ["UDGraph"]


class UD_NODE_TYPE(str):
    """Node types"""

    # NOTE: possibly use POS tags as node types directly
    SENTENCE = "sentence"
    TOKEN = "token"
    ROOT = "root"


class UD_EDGE_TYPE(str):
    """Edge types"""

    # NOTE: possibly use dependency relations as edge type directly
    SENTENCE_CONNECT = "sentence-connect"
    DEPENDENCY_RELATION = "dependency-relation"


class UDGraph(BaseGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    def from_conll(self, conll_path: PathLike):
        """Construct the graph using the provided conll file"""
        sentences, _ = CoNLL.conll2dict(conll_path)

        nodes, edges = [], []
        for sentence_idx, sentence in enumerate(sentences):
            # Explicitly add the root node for each sentence
            root_id = (sentence_idx, UD_NODE_TYPE.ROOT, 0)
            nodes.append(
                (
                    root_id,
                    {
                        "id": root_id,
                        "text": "ROOT",
                        "token": "ROOT",
                        "lemma": None,
                        "upos": None,
                        "xpos": None,
                        "feats": None,
                        "head": None,
                    },
                )
            )
            for token in sentence:
                # Ids are read in as tuples, but currently there are no parses
                # with multiple or duplicate ids (not sure when that happens,
                # with pre-annotated docs maybe?)
                assert (
                    len(token["id"]) == 1
                ), f"Multiple ids found, cannot parse this currently"

                tok_id = (sentence_idx, UD_NODE_TYPE.TOKEN, token["id"][0])
                tok_data = {**token, **{"id": tok_id, "token": token["text"]}}

                if token["head"] == 0:
                    head_id = (sentence_idx, UD_NODE_TYPE.ROOT, token["head"])
                else:
                    head_id = (sentence_idx, UD_NODE_TYPE.TOKEN, token["head"])
                edge_data = {"token": token["deprel"]}

                nodes.append((tok_id, tok_data))
                edges.append((head_id, tok_id, edge_data))

        self.add_nodes_from(nodes)
        self.add_edges_from(edges)

        return self
