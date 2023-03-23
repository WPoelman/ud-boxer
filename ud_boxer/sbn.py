from __future__ import annotations
import re
import logging
from collections import Counter
import os
from copy import deepcopy
from os import PathLike
from pathlib import Path
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import Any, Dict, List, Optional, Tuple, Union
# from helpers import pmb_generator
import networkx as nx
import penman
from tqdm import tqdm
from base import BaseEnum, BaseGraph
from graph_resolver import GraphResolver
from misc import ensure_ext
from penman.graph import Graph
from penman_model import pm_model
from sbn_spec import (
    SBN_EDGE_TYPE,
    SBN_NODE_TYPE,
    SBNError,
    SBNSpec,
    split_comments,
    split_single,
    split_synset_id,
)

import spacy

# load_model = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


# def get_lemma(text):
#     doc = load_model(text)
#     return [token.lemma_ for token in doc]


logger = logging.getLogger(__name__)

RESOLVER = GraphResolver()

__all__ = [
    "SBN_ID",
    "SBNGraph",
    "sbn_graphs_are_isomorphic",
]

# Node / edge ids, unique combination of type and index / count for the current
# document.
SBN_ID = Tuple[Union[SBN_NODE_TYPE, SBN_EDGE_TYPE], int]


class SBNSource(BaseEnum):
    # The SBNGraph is created from an SBN file that comes from the PMB directly
    PMB = "PMB"
    # The SBNGraph is created from GREW output
    GREW = "GREW"
    # The SBNGraph is created from a self generated SBN file
    INFERENCE = "INFERENCE"
    # The SBNGraph is created from a seq2seq generated SBN line
    SEQ2SEQ = "SEQ2SEQ"
    # We don't know the source or it is 'constructed' manually
    UNKNOWN = "UNKNOWN"


class SBNGraph(BaseGraph):
    def __init__(
            self,
            incoming_graph_data=None,
            source: SBNSource = SBNSource.UNKNOWN,
            **attr,
    ):
        super().__init__(incoming_graph_data, **attr)
        self.is_dag: bool = False
        self.is_possibly_ill_formed: bool = False
        self.source: SBNSource = source

    def from_path(self, path: PathLike) -> SBNGraph:
        """Construct a graph from the provided filepath."""
        return self.from_string(Path(path).read_text())

    def from_string(self, input_string: str) -> SBNGraph:
        """Construct a graph from a single SBN string."""
        # Determine if we're dealing with an SBN file with newlines (from the
        # PMB for instance) or without (from neural output).
        if "\n" not in input_string:
            input_string = split_single(input_string)
            # print(input_string)

        lines = split_comments(input_string)
        sbn_info = [(x.split(), y) for x, y in lines]
        sbn_info_reference = deepcopy(sbn_info)
        sbn_node_reference_with_boxes_info = [(x[0], y) for x, y in sbn_info_reference]
        sbn_node_reference = [(x[0], y) for x, y in sbn_info_reference if x[0] not in SBNSpec.NEW_BOX_INDICATORS]
        sbn_node_reference2 = [x[0] for x, _ in sbn_info_reference if x[0] not in SBNSpec.NEW_BOX_INDICATORS]
        sbn_node_reference_with_boxes = [x[0] for x, _ in sbn_info_reference]
        # sbn_info.sort(key=lambda t: ('v' in t[0][0], len(t[0])), reverse=True)

        if not lines:
            raise SBNError(
                "SBN doc appears to be empty, cannot read from string"
            )

        self.__init_type_indices()

        starting_box = self.create_node(
            SBN_NODE_TYPE.BOX, self._active_box_token
        )

        nodes, edges = [starting_box], []
        max_wn_idx = len(lines) - 1
        reference_nodes = []
        reference_nodes_without_null = []
        count = -1
        for idx, (basic_node, basic_comment) in enumerate(sbn_node_reference_with_boxes_info):
            if synset_match := SBNSpec.SYNSET_PATTERN.match(basic_node):
                count += 1
                synset_node = self.create_node(
                    SBN_NODE_TYPE.SYNSET,
                    basic_node,
                    {
                        "wn_lemma": synset_match.group(),
                        "comment": basic_comment,
                    },
                )
                nodes.append(synset_node)
                reference_nodes_without_null.append((SBN_NODE_TYPE.SYNSET, count))
                reference_nodes.append((SBN_NODE_TYPE.SYNSET, count, basic_node))
                # reference_nodes.append(synset_node[0])
            elif basic_node in SBNSpec.NEW_BOX_INDICATORS:
                reference_nodes.append('null')
            else:
                raise SBNError(
                    "The structure of sbn is not correct!"
                )

        for j, (sbn_line, comment) in enumerate(sbn_info):
            while len(sbn_line) > 0:
                token = sbn_line[0]

                if SBNSpec.SYNSET_PATTERN.match(token):
                    sbn_line.pop(0)
                else:
                    sub_token = sbn_line.pop(0)
                    if (is_role := sub_token in SBNSpec.ROLES) or (
                            sub_token in SBNSpec.DRS_OPERATORS
                    ):
                        if not sbn_line:
                            raise SBNError(
                                f"Missing target for '{sub_token}' in line {sbn_line}"
                            )

                        target = sbn_line.pop(0)

                        edge_type = (
                            SBN_EDGE_TYPE.ROLE
                            if is_role
                            else SBN_EDGE_TYPE.DRS_OPERATOR
                        )

                        current_node = (reference_nodes[j][0], reference_nodes[j][1])
                        if index_match := SBNSpec.INDEX_PATTERN.match(target):
                            idx = self._try_parse_idx(index_match.group(0))
                            if SBNSpec.MIN_SYNSET_IDX <= current_node[1] + idx <= max_wn_idx:
                                target_node = reference_nodes_without_null[current_node[1] + idx]
                                # if sub_token in SBNSpec.INVERTIBLE_ROLES:
                                #     role_edge = self.create_edge(
                                #         target_node,
                                #         current_node,
                                #         edge_type,
                                #         sub_token[:-2],
                                #     )
                                #     edges.append(role_edge)
                                # else:
                                role_edge = self.create_edge(
                                    current_node,
                                    target_node,
                                    edge_type,
                                    sub_token,
                                )

                                edges.append(role_edge)
                            else:
                                # A special case where a constant looks like an idx
                                # Example:
                                # pmb-4.0.0/data/en/silver/p15/d3131/en.drs.sbn
                                # This is detected by checking if the provided
                                # index points at an 'impossible' line (synset) in
                                # the file.

                                # NOTE: we have seen that the neural parser does
                                # this very (too) frequently, resulting in arguably
                                # ill-formed graphs.
                                self.is_possibly_ill_formed = True

                                const_node = self.create_node(
                                    SBN_NODE_TYPE.CONSTANT,
                                    target,
                                    {"comment": comment},
                                )
                                role_edge = self.create_edge(
                                    current_node,
                                    const_node[0],
                                    edge_type,
                                    sub_token,
                                )
                                nodes.append(const_node)
                                edges.append(role_edge)

                        elif SBNSpec.NAME_CONSTANT_PATTERN.match(target):
                            name_parts = [target]

                            # Some names contain whitspace and need to be
                            # reconstructed
                            while not target.endswith('"'):
                                target = sbn_line.pop(0)
                                name_parts.append(target)

                            # This is faster than constantly creating new strings
                            name = " ".join(name_parts)

                            name_node = self.create_node(
                                SBN_NODE_TYPE.CONSTANT,
                                name,
                                {"comment": comment},
                            )
                            role_edge = self.create_edge(
                                current_node,
                                name_node[0],
                                SBN_EDGE_TYPE.ROLE,
                                sub_token,
                            )

                            nodes.append(name_node)
                            edges.append(role_edge)
                        else:

                            const_node = self.create_node(
                                SBN_NODE_TYPE.CONSTANT,
                                target,
                                {"comment": comment},
                            )
                            role_edge = self.create_edge(
                                current_node,
                                const_node[0],
                                SBN_EDGE_TYPE.ROLE,
                                sub_token,
                            )

                            nodes.append(const_node)
                            edges.append(role_edge)

                    elif sub_token in SBNSpec.NEW_BOX_INDICATORS:

                        # In the entire dataset there are no indices for box
                        # references other than -1. Maybe they are needed later and
                        # the exception triggers if something different comes up.
                        if not sbn_line:
                            raise SBNError(
                                f"Missing box index in line: {sbn_line}"
                            )

                        if (box_index := self._try_parse_idx(sbn_line.pop(0))) != -1:
                            raise SBNError(
                                f"Unexpected box index found '{box_index}'"
                            )
                        sbn_node_reference_without_comment = [x for x, _ in sbn_info_reference]
                        sbn_node_reference_without_comment.sort(key=lambda t: ('v' in t[0], len(t)), reverse=True)
                        target_node = [x for x in reference_nodes if sbn_node_reference_without_comment[0][0] in x][0]
                        active_id = (target_node[0], target_node[1])

                        if sub_token in SBNSpec.NEW_BOX_INDICATORS_2VERB:

                            freq = len([x for x in sbn_node_reference_with_boxes if x == sub_token])
                            if freq > 1:

                                active_id = (reference_nodes[j + 1][0], reference_nodes[j + 1][1])
                                new_node = self.create_node(
                                    SBN_NODE_TYPE.CONSTANT,
                                    "+1",
                                    {"comment": comment},
                                )
                                new_edge = self.create_edge(
                                    active_id,
                                    new_node[0],
                                    SBN_EDGE_TYPE.DRS_OPERATOR,
                                    sub_token,
                                )
                                nodes.append(new_node)
                                edges.append(new_edge)

                            else:
                                new_node = self.create_node(
                                    SBN_NODE_TYPE.CONSTANT,
                                    "+1",
                                    {"comment": comment},
                                )
                                new_edge = self.create_edge(
                                    active_id,
                                    new_node[0],
                                    SBN_EDGE_TYPE.DRS_OPERATOR,
                                    sub_token,
                                )
                                nodes.append(new_node)
                                edges.append(new_edge)

                        else:
                            pre, after = [x for x, _ in sbn_info_reference][:j], [x for x, _ in sbn_info_reference][
                                                                                 j + 1:]
                            pre = sorted(pre, key=lambda x: (len(x), 'v' in x[0]), reverse=True)
                            after = sorted(after, key=lambda x: (len(x), 'v' in x[0]), reverse=True)
                            preverb_id_node = [x for x in reference_nodes if pre[0][0] in x][0]
                            afterverb_id_node = [x for x in reference_nodes if after[0][0] in x][0]
                            preverb_id = (preverb_id_node[0], preverb_id_node[1])
                            afterverb_id = (afterverb_id_node[0], afterverb_id_node[1])

                            new_edge = self.create_edge(
                                preverb_id,
                                afterverb_id,
                                SBN_EDGE_TYPE.ROLE,
                                token,
                            )
                            edges.append(new_edge)


                    else:
                        raise SBNError(
                            f"Invalid token found '{sub_token}' in line: {sbn_line}"
                        )
        # print([(edge[0],edge[1]) for edge in edges])
        # print(edges)
        from_node = set([edge[0] for edge in edges])
        to_node = set([edge[1] for edge in edges if edge[1][0] == SBN_NODE_TYPE.SYNSET])
        predicate_nodes = [x for x in from_node if x not in to_node]

        # assert len(to_node) <= len(from_node)

        for predicate_node in predicate_nodes:
            new_box_syn_edge = self.create_edge(
                self._active_box_id,
                predicate_node,
                SBN_EDGE_TYPE.BOX_CONNECT,
            )
            edges.append(new_box_syn_edge)

        self.add_nodes_from(nodes)
        self.add_edges_from(edges)

        self._check_is_dag()
        return self

    def from_grew(self, grew_graph: Dict[str, List[Any]]) -> SBNGraph:
        """Create an SBNGraph from a grew output format graph."""
        self.__init_type_indices()

        starting_box = self.create_node(
            SBN_NODE_TYPE.BOX, self._active_box_token
        )

        nodes, edges = [starting_box], []
        id_mapping: Dict[str, SBN_ID] = dict()

        # First collect all nodes and create a mapping from the grew ids to
        # the current graph ids.
        for grew_node_id, (node_data, _) in grew_graph.items():
            node_components = RESOLVER.node_token_type(node_data)
            node = self.create_node(*node_components)
            id_mapping[grew_node_id] = node[0]
            nodes.append(node)

        self.add_nodes_from(nodes)

        # With this we 'climb' back up the box nodes to find the correct place
        # to connect the new box. This only considers the starting box as
        # a starting point. Other box constructions are not supported currently
        box_count = self.type_indices[SBN_NODE_TYPE.BOX]
        for grew_from_node_id, (_, grew_edges) in grew_graph.items():
            from_id = id_mapping[grew_from_node_id]
            from_type = from_id[0]

            if from_type == SBN_NODE_TYPE.BOX:
                box_box_edge = self.create_edge(
                    self._prev_box_id(box_count),
                    from_id,
                    SBN_EDGE_TYPE.BOX_BOX_CONNECT,
                    self.nodes[from_id]["token"],
                )
                edges.append(box_box_edge)
                box_count -= 1
                continue  # <-- NOTICE THIS PLEASE, THIS SKIPS THE OUT EDGES

            if from_type == SBN_NODE_TYPE.SYNSET:
                box_edge = self.create_edge(
                    self._active_box_id,
                    from_id,
                    SBN_EDGE_TYPE.BOX_CONNECT,
                )
                edges.append(box_edge)

            for edge_name, grew_to_node_id in grew_edges:
                to_id = id_mapping[grew_to_node_id]
                edge_components = RESOLVER.edge_token_type(
                    edge_name,
                    self.nodes,
                    from_id,
                    to_id,
                )

                edge = self.create_edge(from_id, to_id, *edge_components)
                edges.append(edge)

        self.add_edges_from(edges)

        self._check_is_dag()

        return self

    def create_edge(
            self,
            from_node_id: SBN_ID,
            to_node_id: SBN_ID,
            type: SBN_EDGE_TYPE,
            token: Optional[str] = None,
            meta: Optional[Dict[str, Any]] = None,
    ):
        """Create an edge, if no token is provided, the id will be used."""
        edge_id = self._id_for_type(type)
        meta = meta or dict()
        return (
            from_node_id,
            to_node_id,
            {
                "_id": str(edge_id),
                "type": type,
                "type_idx": edge_id[1],
                "token": token or str(edge_id),  # xiulin: it seems it's always str(edge_id)?
                **meta,
            },
        )

    def create_node(
            self,
            type: SBN_NODE_TYPE,
            token: Optional[str] = None,
            meta: Optional[Dict[str, Any]] = None,
    ):
        """Create a node, if no token is provided, the id will be used."""
        node_id = self._id_for_type(type)
        meta = meta or dict()
        return (
            node_id,
            {
                "_id": str(node_id),
                "type": type,
                "type_idx": node_id[1],
                "token": token or str(node_id),
                **meta,
            },
        )

    def to_sbn(self, path: PathLike, add_comments: bool = False) -> Path:
        """Writes the SBNGraph to a file in sbn format"""
        final_path = ensure_ext(path, ".sbn")
        final_path.write_text(self.to_sbn_string(add_comments))
        return final_path

    def to_sbn_string(self, add_comments: bool = False) -> str:
        """Creates a string in sbn format from the SBNGraph"""
        result = []
        synset_idx_map: Dict[SBN_ID, int] = dict()
        line_idx = 0

        box_nodes = [
            node for node in self.nodes if node[0] == SBN_NODE_TYPE.BOX
        ]
        for box_node_id in box_nodes:
            box_box_connect_to_insert = None
            for edge_id in self.out_edges(box_node_id):
                _, to_node_id = edge_id
                to_node_type, _ = to_node_id

                edge_data = self.edges.get(edge_id)
                if edge_data["type"] == SBN_EDGE_TYPE.BOX_BOX_CONNECT:
                    if box_box_connect_to_insert:
                        raise SBNError(
                            "Found box connected to multiple boxes, "
                            "is that possible?"
                        )
                    else:
                        box_box_connect_to_insert = edge_data["token"]

                if to_node_type in (
                        SBN_NODE_TYPE.SYNSET,
                        SBN_NODE_TYPE.CONSTANT,
                ):
                    if to_node_id in synset_idx_map:
                        raise SBNError(
                            "Ambiguous synset id found, should not be possible"
                        )

                    synset_idx_map[to_node_id] = line_idx
                    temp_line_result = [to_node_id]
                    for syn_edge_id in self.out_edges(to_node_id):
                        _, syn_to_id = syn_edge_id

                        syn_edge_data = self.edges.get(syn_edge_id)
                        if syn_edge_data["type"] not in (
                                SBN_EDGE_TYPE.ROLE,
                                SBN_EDGE_TYPE.DRS_OPERATOR,
                        ):
                            raise SBNError(
                                f"Invalid synset edge connect found: "
                                f"{syn_edge_data['type']}"
                            )

                        temp_line_result.append(syn_edge_data["token"])

                        syn_node_to_data = self.nodes.get(syn_to_id)
                        syn_node_to_type = syn_node_to_data["type"]
                        if syn_node_to_type == SBN_NODE_TYPE.SYNSET:
                            temp_line_result.append(syn_to_id)
                        elif syn_node_to_type == SBN_NODE_TYPE.CONSTANT:
                            temp_line_result.append(syn_node_to_data["token"])
                        else:
                            raise SBNError(
                                f"Invalid synset node connect found: "
                                f"{syn_node_to_type}"
                            )

                    result.append(temp_line_result)
                    line_idx += 1
                elif to_node_type == SBN_NODE_TYPE.BOX:
                    pass
                else:
                    raise SBNError(f"Invalid node id found: {to_node_id}")

            if box_box_connect_to_insert:
                result.append([box_box_connect_to_insert, "-1"])

        # Resolve the indices and the correct synset tokens and create the sbn
        # line strings for the final string
        final_result = []
        if add_comments:
            final_result.append(
                (
                    f"{SBNSpec.COMMENT_LINE} SBN source: {self.source.value}",
                    " ",
                )
            )
        current_syn_idx = 0
        for line in result:
            tmp_line = []
            comment_for_line = None

            for token_idx, token in enumerate(line):
                # There can never be an index at the first token of a line, so
                # always start at the second token.
                if token_idx == 0:
                    # It is a synset id that needs to be converted to a token
                    if token in synset_idx_map:
                        node_data = self.nodes.get(token)
                        tmp_line.append(node_data["token"])
                        comment_for_line = comment_for_line or (
                            node_data["comment"]
                            if "comment" in node_data
                            else None
                        )
                        current_syn_idx += 1
                    # It is a regular token
                    else:
                        tmp_line.append(token)
                # It is a synset which needs to be resolved to an index
                elif token in synset_idx_map:
                    target = synset_idx_map[token] - current_syn_idx + 1
                    # In the PMB dataset, an index of '0' is written as '+0',
                    # so do that here as well.
                    tmp_line.append(
                        f"+{target}" if target >= 0 else str(target)
                    )
                # It is a regular token
                else:
                    tmp_line.append(token)

            if add_comments and comment_for_line:
                tmp_line.append(f"{SBNSpec.COMMENT}{comment_for_line}")

            # This is a bit of trickery to vertically align synsets just as in
            # the PMB dataset.
            if len(tmp_line) == 1:
                final_result.append((tmp_line[0], " "))
            else:
                final_result.append((tmp_line[0], " ".join(tmp_line[1:])))

        # More formatting and alignment trickery.
        max_syn_len = max(len(s) for s, _ in final_result) + 1
        sbn_string = "\n".join(
            f"{synset: <{max_syn_len}}{rest}".rstrip(" ")
            for synset, rest in final_result
        )

        return sbn_string

    def to_penman(
            self, path: PathLike, evaluate_sense: bool = False, strict: bool = True
    ) -> PathLike:
        """
        Writes the SBNGraph to a file in Penman (AMR-like) format.

        See `to_penman_string` for an explanation of `strict`.
        """
        final_path = ensure_ext(path, ".penman")
        final_path.write_text(self.to_penman_string(evaluate_sense, strict))
        return final_path

    def to_penman_string(
            self, evaluate_sense: bool = False, strict: bool = True
    ) -> str:
        """
        Creates a string in Penman (AMR-like) format from the SBNGraph.

        The 'evaluate_sense; flag indicates if the sense number is included.
        If included, the evaluation indirectly also targets the task of word
        sense disambiguation, which might not be desirable. Example:

            (b0 / "box"
                :member (s0 / "synset"
                    :lemma "person"
                    :pos "n"
                    :sense "01")) # Would be excluded when False

        The 'strict' option indicates how to handle possibly ill-formed graphs.
        Especially when indices point at impossible synsets. Cyclic graphs are
        also ill-formed, but these are not even allowed to be exported to
        Penman.

        FIXME: the DRS/SBN constants technically don't need a variable. As long
        as this is consistent between the gold and generated data, it's not a
        problem.
        """
        if not self.is_dag:
            raise SBNError(
                "Exporting a cyclic SBN graph to Penman is not possible."
            )

        if strict and self.is_possibly_ill_formed:
            raise SBNError(
                "Strict evaluation mode, possibly ill-formed graph not "
                "exported."
            )

        # Make a copy just in case since strange side-effects such as token
        # changes are no fun to debug.
        G = deepcopy(self)

        prefix_map = {
            SBN_NODE_TYPE.BOX: ["b", 0],
            SBN_NODE_TYPE.CONSTANT: ["c", 0],
            SBN_NODE_TYPE.SYNSET: ["s", 0],
        }
        # print(G.nodes.items())
        for node_id, node_data in G.nodes.items():

            pre, count = prefix_map[node_data["type"]]
            prefix_map[node_data["type"]][1] += 1  # type: ignore
            G.nodes[node_id]["var_id"] = f"{pre}{count}"

            # A box is always an instance of the same type (or concept), the
            # specification of what that type does is shown by the
            # box-box-connection, such as NEGATION or EXPLANATION.
            if node_data["type"] == SBN_NODE_TYPE.BOX:
                G.nodes[node_id]["token"] = "root"

        for edge in G.edges:
            # Add a proper token to the box connectors
            if G.edges[edge]["type"] == SBN_EDGE_TYPE.BOX_CONNECT:
                G.edges[edge]["token"] = "member"

        def __to_penman_str(S: SBNGraph, current_n, visited, out_str, tabs):
            # print(current_n)
            node_data = S.nodes[current_n]
            # print(node_data)
            var_id = node_data["var_id"]
            # print(visited)
            if var_id in visited:
                out_str += var_id
                return out_str

            indents = tabs * "\t"

            node_tok = node_data["token"]
            if node_data["type"] == SBN_NODE_TYPE.SYNSET:
                if not (components := split_synset_id(node_tok)):
                    raise SBNError(f"Cannot split synset id: {node_tok}")

                lemma, pos, sense = [self.quote(i) for i in components]

                out_str += f'({var_id} / {".".join(components)}'
                # print(self.quote("synset"))
                # out_str += f"\n{indents}:lemma {lemma}"
                # out_str += f"\n{indents}:pos {pos}"

                # if evaluate_sense:
                #     out_str += f"\n{indents}:sense {sense}"
            else:
                out_str += f"({var_id} / {self.quote(node_tok)}"

            if S.out_degree(current_n) > 0:
                for edge_id in S.edges(current_n):
                    edge_name = S.edges[edge_id]["token"]
                    if edge_name in SBNSpec.INVERTIBLE_ROLES:
                        # print(edge_name)
                        # SMATCH can invert edges that end in '-of'.
                        # This means that,
                        #   A -[AttributeOf]-> B
                        #   B -[Attribute]-> A
                        # are treated the same, but they need to be in the
                        # right notation for this to work.
                        edge_name = edge_name.replace("Of", "-of")

                    _, child_node = edge_id
                    out_str += f"\n{indents}:{edge_name} "
                    out_str = __to_penman_str(
                        S, child_node, visited, out_str, tabs + 1
                    )
            out_str += ")"
            visited.add(var_id)

            return out_str

        # Assume there always is the starting box to serve as the "root"
        starting_node = (SBN_NODE_TYPE.BOX, 0)
        final_result = __to_penman_str(G, starting_node, set(), "", 1)

        try:
            g = penman.decode(final_result)

            if errors := pm_model.errors(g):
                raise penman.DecodeError(str(errors))
            assert len(g.edges()) == len(self.edges), "Wrong number of edges"
        except (penman.DecodeError, AssertionError) as e:
            raise SBNError(f"Generated Penman output is invalid: {e}")

        return final_result

    def __init_type_indices(self):
        self.type_indices = {
            SBN_NODE_TYPE.SYNSET: 0,
            SBN_NODE_TYPE.CONSTANT: 0,
            SBN_NODE_TYPE.BOX: 0,
            SBN_EDGE_TYPE.ROLE: 0,
            SBN_EDGE_TYPE.DRS_OPERATOR: 0,
            SBN_EDGE_TYPE.BOX_CONNECT: 0,
            SBN_EDGE_TYPE.BOX_BOX_CONNECT: 0,
        }

    def _id_for_type(
            self, type: Union[SBN_EDGE_TYPE, SBN_NODE_TYPE]
    ) -> SBN_ID:
        _id = (type, self.type_indices[type])
        self.type_indices[type] += 1
        return _id

    def _check_is_dag(self) -> bool:
        self.is_dag = nx.is_directed_acyclic_graph(self)
        if not self.is_dag:
            logger.warning(
                "Initialized cyclic SBN graph, this will work for most tasks, "
                "but can cause problems later on when exporting to Penman for "
                "instance."
            )
        return self.is_dag

    @staticmethod
    def _try_parse_idx(possible_idx: str) -> int:
        """Try to parse a possible index, raises an SBNError if this fails."""
        try:
            return int(possible_idx)
        except ValueError:
            raise SBNError(f"Invalid index '{possible_idx}' found.")

    @staticmethod
    def quote(in_str: str) -> str:
        """Consistently quote a string with double quotes"""
        if in_str.startswith('"') and in_str.endswith('"'):
            return in_str

        if in_str.startswith("'") and in_str.endswith("'"):
            return f'"{in_str[1:-1]}"'

        return f'"{in_str}"'

    @property
    def _active_synset_id(self) -> SBN_ID:
        return (
            SBN_NODE_TYPE.SYNSET,
            self.type_indices[SBN_NODE_TYPE.SYNSET] - 1)

    def _active_node_synset_id(self, target_node, reference) -> SBN_ID:

        return (SBN_NODE_TYPE.SYNSET,
                reference.index(target_node))

    @property
    def _active_box_id(self) -> SBN_ID:
        return (SBN_NODE_TYPE.BOX,
                self.type_indices[SBN_NODE_TYPE.BOX] - 1)

    # def 

    def _prev_box_id(self, offset: int) -> SBN_ID:
        n = self.type_indices[SBN_NODE_TYPE.BOX]
        return (
            SBN_NODE_TYPE.BOX,
            max(0, min(n, n - offset)),  # Clamp so we always have a valid box
        )

    @property
    def _active_box_token(self) -> str:
        return f"B-{self.type_indices[SBN_NODE_TYPE.BOX]}"

    @staticmethod
    def _node_label(node_data) -> str:
        return node_data["token"]
        # return "\n".join(f"{k}={v}" for k, v in node_data.items())

    @staticmethod
    def _edge_label(edge_data) -> str:
        return edge_data["token"]
        # return "\n".join(f"{k}={v}" for k, v in edge_data.items())

    @property
    def type_style_mapping(self):
        """Style per node type to use in dot export"""
        return {
            SBN_NODE_TYPE.SYNSET: {},
            SBN_NODE_TYPE.CONSTANT: {"shape": "none"},
            SBN_NODE_TYPE.BOX: {"shape": "box", "label": ""},
            SBN_EDGE_TYPE.ROLE: {},
            SBN_EDGE_TYPE.DRS_OPERATOR: {},
            SBN_EDGE_TYPE.BOX_CONNECT: {"style": "dotted", "label": ""},
            SBN_EDGE_TYPE.BOX_BOX_CONNECT: {},
        }


def sbn_graphs_are_isomorphic(A: SBNGraph, B: SBNGraph) -> bool:
    """
    Checks if two SBNGraphs are isomorphic this is based on node and edge
    ids as well as the 'token' meta data per node and edge
    """

    # Type and count are already compared implicitly in the id comparison that
    # is done in the 'is_isomorphic' function. The tokens are important to
    # compare since some constants (names, dates etc.) need to be reconstructed
    # properly with their quotes in order to be valid.
    def node_cmp(node_a, node_b) -> bool:
        return node_a["token"] == node_b["token"]

    def edge_cmp(edge_a, edge_b) -> bool:
        return edge_a["token"] == edge_b["token"]

    return nx.is_isomorphic(A, B, node_cmp, edge_cmp)


def main(starting_path):
    pre_map = {
        SBN_NODE_TYPE.BOX: "b",
        SBN_NODE_TYPE.CONSTANT: "c",
        SBN_NODE_TYPE.SYNSET: "s",
    }
    error = 0
    with open('correct_penman.txt', 'w') as penman_link:
        for filepath in pmb_generator(
                starting_path, "**/*.sbn", desc_tqdm="Generating Penman files "
        ):
            alignment = {}
            nodes_info = []
            edges_info = []
            comments = []
            comment_taken = []
            comment_node_pair_info = ()
            comment_node=[]
            none_node_pair = {}

            # TODO punctuations
            # punctuations= {'.','?', '!'}
            try:
                G = SBNGraph().from_path(filepath)
                for edge_info in G.edges.data(True):
                    current_id = pre_map[edge_info[0][0]] + str(edge_info[0][1])
                    to_id = pre_map[edge_info[1][0]] + str(edge_info[1][1])
                    if current_id != 'b0':
                        edges_info.append((current_id, to_id))

                for info in G.nodes.data(True):

                    if 'comment' in list(info[1].keys()):
                        node_type = info[0][0]
                        var_id = pre_map[node_type] + str(info[0][1])
                        token = info[1]['token']
                        comment = info[1]['comment']
                        nodes_info.append((var_id, token, comment))
                        if comment != None:
                            comments.append(comment)
                            if comment not in comment_taken:
                                comment_node_pair_info.append(comment,[(var_id, token)])
                                comment_node.append(var_id)
                                comment_taken.append(comment)
                            else:
                                comment_node_pair_info[comment].append((var_id, token))
                                comment_node.append(var_id)

                comment_list = ' '.join(comments).split()
                for token, node_info in comment_node_pair_info.items():
                    split_tokens = token.split()
                    if len(split_tokens) == 1 and len(node_info)==1:
                        alignment['node_name'] = node_info[0][0]
                        token_id = comment_list.index(token)
                        alignment['token_id'] = token_id
                        # comment_list[token_id] ='null'

                    elif len(split_tokens) == 1 and len(node_info)>1:
                        alignment['node_name'] = [x[0] for x in node_info]
                        alignment['token_id'] = comment_list.index(token)

                print(comment_node_pair_info)

                for n in nodes_info:
                    if n[-1]==None:
                        possible_nodes = []
                        def find_node(var_id1):
                            target_n = [x[0] for x in edges_info if x[1] == var_id1][0]
                            possible_nodes.append((var_id1, target_n))
                            if target_n not in comment_node:
                                find_node(target_n)
                            else:
                                return possible_nodes

                        find_node(var_id)
                        for k, v in comment_node_pair_info.items():
                            cluster = [x[0] for x in v]
                            if possible_nodes[-1][-1] in cluster:
                                possible_nodes = [x for y in possible_nodes for x in y]
                                possible_nodes = list(dict.fromkeys(possible_nodes))
                                v.append(possible_nodes)



                output_path = Path(f"{filepath.parent}/{filepath.stem}.penman")
                G.to_penman(output_path)
                with open(output_path, 'r') as penman_file:
                    penman_string = penman_file.read()

                    b0_freq = Counter(penman_string.split())[':member']
                    if int(b0_freq) > 1:
                        print(f'isolate node {output_path}')
                    triple = penman.decode(penman_string).triples
                    output_path_penmaninfo = Path(f"{filepath.parent}/{filepath.stem}.penmaninfo")
                    with open(output_path_penmaninfo, 'w') as penmaninfo:
                        for t in triple:
                            penmaninfo.write(f'{t}\n')
                print(f'correct{output_path}')
                penman_link.write(f"{output_path_penmaninfo}\n")

            except SBNError:

                error += 1
                print(error)
                print(f'error {filepath}')
                print(SBNError)


def pmb_generator(
        starting_path: PathLike,
        pattern: str,
        # By default we don't want to regenerate predicted output
        exclude: str = "predicted",
        disable_tqdm: bool = False,
        desc_tqdm: str = "",
):
    """Helper to glob over the pmb dataset"""
    path_glob = Path(starting_path).glob(pattern)
    return tqdm(
        (p for p in path_glob if exclude not in str(p)),
        disable=disable_tqdm,
        desc=desc_tqdm,
    )


if __name__ == "__main__":
    starting_path = Path('/Users/shirleenyoung/Desktop/TODO/MA_Thesis/ud-boxer/ud_boxer/test_data')
    with logging_redirect_tqdm():
        main(starting_path)
