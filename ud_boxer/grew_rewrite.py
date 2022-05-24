# Grew has no stubs & mixed types everywhere, no need to bother mypy with that.
# mypy: ignore-errors
import tempfile
from os import PathLike
from pathlib import Path
from typing import List

import grew
from ud_boxer.config import Config
from ud_boxer.graph_resolver import GraphResolver
from ud_boxer.sbn import SBNGraph
from ud_boxer.sbn_spec import SBN_EDGE_TYPE, SBN_NODE_TYPE, SBNError

__all__ = [
    "Grew",
]

# Placeholder to dynamically build a grs file for a specific language.
LANGUAGE_PLACEHOLDER = "$$LANGUAGE$$"


class Grew:
    def __init__(
        self,
        grs_path: PathLike = Config.GRS_PATH,
        language: Config.SUPPORTED_LANGUAGES = Config.SUPPORTED_LANGUAGES.EN,
    ) -> None:
        grew.init()
        self.current_grs_path = self._build_grs(grs_path, language)
        # note that this is an index for internal grew use
        self.grs: int = grew.grs(str(self.current_grs_path))

    def run(self, conll_path: PathLike, strat: str = "main") -> SBNGraph:
        # This is not ideal since we need to deserialize the file 2x, once
        # here and once 'inside' GREW. It might be worth it to convert the
        # sentence(s) to a GREW graph(s) directly. We need to this though since
        # GREW throws an error when providing a conll-u file with > 1 sentence.
        sentences = Path(conll_path).read_text().rstrip().split("\n\n")

        if len(sentences) > 1:
            graphs = []
            for sent in sentences:
                with tempfile.NamedTemporaryFile("w") as f:
                    Path(f.name).write_text(sent)
                    grew_graph = grew.graph(f.name)
                    results = grew.run(self.grs, grew_graph, strat)
                graphs.append(SBNGraph().from_grew(results[0]))
            final_graph = self.merge_graphs(graphs)
        else:
            grew_graph = grew.graph(str(conll_path))
            result = grew.run(self.grs, grew_graph, strat)
            final_graph = SBNGraph().from_grew(result[0])

        return final_graph

    @staticmethod
    def merge_graphs(graphs: List[SBNGraph]) -> SBNGraph:
        """
        Merge n SBNGraphs into a single SBNGraph. This is done by grabbing the
        first graph (A) in the list and 'glueing' subsequent graphs to it. This
        means no nodes or edges are lost, in contrast to the set-like merge
        operations in networkx. Even though there might be overlap in terms of
        ids (multiple box-0's), the result keeps all boxes and ensures all ids
        are correctly added to A. The strategy of connecting boxes is by using
        the most common box indicator (currently).
        """
        A = graphs.pop(0)
        for B in graphs:
            nodes, edges = [], []
            node_mapping = dict()

            for node_id, node_data in B.nodes.items():
                if node_data["type"] == SBN_NODE_TYPE.BOX:
                    active_box = A._active_box_id
                    node = A.create_node(
                        node_data["type"],
                        A._active_box_token,
                        GraphResolver.filter_item_data(node_data),
                    )
                    edges.append(
                        A.create_edge(
                            active_box,
                            node[0],
                            SBN_EDGE_TYPE.BOX_BOX_CONNECT,
                            # TODO: determine this dynamically?
                            Config.DEFAULT_BOX_CONNECT,
                        )
                    )
                else:
                    node = A.create_node(
                        node_data["type"],
                        node_data["token"],
                        GraphResolver.filter_item_data(node_data),
                    )

                nodes.append(node)
                node_mapping[node_id] = node[0]

            for (from_node, to_node), edge_data in B.edges.items():
                edges.append(
                    A.create_edge(
                        node_mapping[from_node],
                        node_mapping[to_node],
                        edge_data["type"],
                        edge_data["token"],
                        GraphResolver.filter_item_data(edge_data),
                    )
                )

            A.add_nodes_from(nodes)
            A.add_edges_from(edges)

        # This would be very strange, but just in case.
        if not A._check_is_dag():
            raise SBNError(
                "Merged SBNgraphs are cyclic, incorrect box connects?"
            )

        return A

    @staticmethod
    def _build_grs(
        grs_path: PathLike, language: Config.SUPPORTED_LANGUAGES
    ) -> Path:
        """Build the grs string with the current language"""
        # Not sure if this is the best way to do this, but it works.
        # I tried returning the grs string, which is also possible, but then
        # the separate language grs files cannot be found.
        grs_str = Path(grs_path).read_text()
        final_grs = grs_str.replace(LANGUAGE_PLACEHOLDER, language)
        current_grs = Path(grs_path).parent / "working_do_not_remove.grs"
        current_grs.write_text(final_grs)

        return current_grs

    def __del__(self):
        self.current_grs_path.unlink()
