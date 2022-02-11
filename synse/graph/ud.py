from os import PathLike
from synse.graph.base import BaseGraph
from stanza.utils.conll import CoNLL

__all__ = ["UDGraph"]


class UDGraph(BaseGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    def from_conll(self, conll_path: PathLike):

        # TODO: very ugly currently, just to test

        sents, _ = CoNLL.conll2dict(conll_path)

        if len(sents) > 1:
            # Continuation box relation ?
            raise ValueError("Document with multiple sentence found, skipping")

        IGNORE_NODE = {"deprel", "misc"}
        ID = "id"
        nodes, edges = [], []

        for tok in sents[0]:
            # Ids are read in as tuples, but currently there are no parses with
            # multiple or duplicate ids (not sure when that happens,
            # with pre-annotated docs maybe?)
            assert (
                len(tok["id"]) == 1
            ), f"Multiple ids found, cannot parse this currently"
            tok_id = tok["id"][0]

            nodes.append(
                (
                    0,
                    {
                        "id": 0,
                        "text": "ROOT",
                        "token": "ROOT",
                        "upos": None,
                        "xpos": None,
                        "feats": None,
                        "head": None,
                    },
                )
            )

            nodes.append(
                (
                    tok_id,
                    {
                        k if k != "lemma" else "token": v if k != ID else v[0]
                        for k, v in tok.items()
                        if k not in IGNORE_NODE
                    },
                )
            )
            edges.append((tok["head"], tok_id, {"token": tok["deprel"]}))

        self.add_nodes_from(nodes)
        self.add_edges_from(edges)

        return self
