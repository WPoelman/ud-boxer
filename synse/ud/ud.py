from enum import Enum
from os import PathLike
from typing import Any, Dict, Optional, Tuple, Union

from stanza.utils.conll import CoNLL

from synse.graph import BaseGraph


class UD_NODE_TYPE(str, Enum):
    """Node types"""

    # NOTE: possibly use POS tags as node types directly
    SENTENCE = "sentence"
    TOKEN = "token"
    ROOT = "root"


class UD_EDGE_TYPE(str, Enum):
    """Edge types"""

    # NOTE: possibly use dependency relations as edge type directly
    SENTENCE_CONNECT = "sentence-connect"
    DEPENDENCY_RELATION = "dependency-relation"


class UDGraph(BaseGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    def from_path(self, conll_path: PathLike):
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
                        "_id": root_id,
                        "token": "ROOT",
                        "lemma": None,
                        "upos": None,
                        "xpos": None,
                        "feats": None,
                        "connl_id": None,
                        "type": UD_NODE_TYPE.ROOT,
                    },
                )
            )
            if sentence_idx > 0:
                edges.append(
                    (
                        (sentence_idx - 1, UD_NODE_TYPE.ROOT, 0),
                        root_id,
                        {
                            "token": "new-sentence",
                            "type": UD_EDGE_TYPE.SENTENCE_CONNECT,
                        },
                    )
                )

            for token in sentence:
                # Ids are read in as tuples, but currently there are no parses
                # with multiple or duplicate ids (not sure when that happens,
                # with pre-annotated docs maybe?)
                assert (
                    len(token["id"]) == 1
                ), f"Multiple ids found, cannot parse this currently."

                tok_id = (sentence_idx, UD_NODE_TYPE.TOKEN, token["id"][0])
                tok_data = {
                    "_id": tok_id,
                    "token": token["text"],
                    "lemma": token.get("lemma"),
                    "upos": token.get("upos"),
                    "xpos": token.get("xpos"),
                    "feats": token.get("feats"),
                    "connl_id": token.get("id"),
                    "type": UD_NODE_TYPE.ROOT,
                }

                if token["head"] == 0:
                    head_id = (sentence_idx, UD_NODE_TYPE.ROOT, token["head"])
                else:
                    head_id = (sentence_idx, UD_NODE_TYPE.TOKEN, token["head"])
                edge_data = {
                    "token": token["deprel"],
                    "type": UD_EDGE_TYPE.DEPENDENCY_RELATION,
                }

                nodes.append((tok_id, tok_data))
                edges.append((head_id, tok_id, edge_data))

        self.add_nodes_from(nodes)
        self.add_edges_from(edges)

        return self

    @property
    def type_style_mapping(self):
        return {
            UD_NODE_TYPE.SENTENCE: {},
            UD_NODE_TYPE.TOKEN: {},
            UD_NODE_TYPE.ROOT: {"shape": "box"},
            UD_EDGE_TYPE.SENTENCE_CONNECT: {"style": "dotted"},
            UD_EDGE_TYPE.DEPENDENCY_RELATION: {},
        }
