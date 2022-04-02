from __future__ import annotations

from copy import deepcopy
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx

from synse.graph import BaseGraph
from synse.sbn.sbn_spec import SBNError, SBNSpec, split_comments
from synse.ud.ud import UD_NODE_TYPE, UDGraph


class SBN_NODE_TYPE(str, Enum):
    """Node types"""

    SENSE = "wordnet-sense"
    NAME_CONSTANT = "name-constant"
    CONSTANT = "constant"
    BOX = "box"


class SBN_EDGE_TYPE(str, Enum):
    """Edge types"""

    ROLE = "role"
    DRS_OPERATOR = "drs-operator"
    BOX_CONNECT = "box-connect"
    BOX_BOX_CONNECT = "box-box-connect"


# Node / edge ids, unique combination of type and index / count for the current
# document.
SBN_ID = Tuple[Union[SBN_NODE_TYPE, SBN_EDGE_TYPE], int]


class SBNGraph(BaseGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    def from_path(self, path: PathLike) -> SBNGraph:
        """Construct a graph from the provided filepath."""
        return self.from_string(Path(path).read_text())

    def from_string(self, input_string: str) -> SBNGraph:
        """Construct a graph from a single SBN string."""
        lines = split_comments(input_string)

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

        for sbn_line, comment in lines:
            tokens = sbn_line.split()

            tok_count = 0
            while len(tokens) > 0:
                # Try to 'consume' all tokens from left to right
                token: str = tokens.pop(0)

                # No need to check all tokens for this since only the first
                # might be a sense id.
                if tok_count == 0 and (
                    sense_match := SBNSpec.WORDNET_SENSE_PATTERN.match(token)
                ):
                    sense_node = self.create_node(
                        SBN_NODE_TYPE.SENSE,
                        token,
                        {
                            "wn_lemma": sense_match.group(1),
                            "wn_pos": sense_match.group(2),
                            "wn_id": sense_match.group(3),
                            "comment": comment,
                        },
                    )
                    box_edge = self.create_edge(
                        self._active_box_id,
                        self._active_sense_id,
                        SBN_EDGE_TYPE.BOX_CONNECT,
                    )

                    nodes.append(sense_node)
                    edges.append(box_edge)
                elif token in SBNSpec.NEW_BOX_INDICATORS:
                    # In the entire dataset there are no indices for box
                    # references other than -1. Maybe they are needed later and
                    # the exception triggers if something different comes up.
                    if box_index := int(tokens.pop(0)) != -1:
                        raise SBNError(
                            f"Unexpected box index found {box_index}"
                        )

                    current_box_id = self._active_box_id

                    # Connect the current box to the one indicated by the index
                    new_box = self.create_node(
                        SBN_NODE_TYPE.BOX, self._active_box_token
                    )
                    box_edge = self.create_edge(
                        current_box_id,
                        self._active_box_id,
                        SBN_EDGE_TYPE.BOX_BOX_CONNECT,
                        token,
                    )

                    nodes.append(new_box)
                    edges.append(box_edge)
                elif (is_role := token in SBNSpec.ROLES) or (
                    token in SBNSpec.DRS_OPERATORS
                ):
                    target = tokens.pop(0)
                    edge_type = (
                        SBN_EDGE_TYPE.ROLE
                        if is_role
                        else SBN_EDGE_TYPE.DRS_OPERATOR
                    )

                    if index_match := SBNSpec.INDEX_PATTERN.match(target):
                        idx = int(index_match.group(0))
                        active_id = self._active_sense_id
                        target_idx = active_id[1] + idx
                        to_id = (active_id[0], target_idx)

                        if SBNSpec.MIN_SENSE_IDX <= target_idx <= max_wn_idx:
                            role_edge = self.create_edge(
                                self._active_sense_id,
                                to_id,
                                edge_type,
                                token,
                            )

                            edges.append(role_edge)
                        else:
                            # This is special case where a constant looks like an
                            # idx. Example:
                            # pmb-4.0.0/data/en/silver/p15/d3131/en.drs.sbn
                            # This is detected by checking if the provided index
                            # points at an 'impossible' line (sense) in the file.
                            const_node = self.create_node(
                                SBN_NODE_TYPE.CONSTANT,
                                target,
                                {"comment": comment},
                            )
                            role_edge = self.create_edge(
                                self._active_sense_id,
                                const_node[0],
                                edge_type,
                                token,
                            )
                            nodes.append(const_node)
                            edges.append(role_edge)
                    elif SBNSpec.NAME_CONSTANT_PATTERN.match(target):
                        name_parts = [target]

                        # Some names contain whitspace and need to be
                        # reconstructed
                        while not target.endswith('"'):
                            target = tokens.pop(0)
                            name_parts.append(target)

                        # This is faster than constantly creating new strings
                        name = " ".join(name_parts)

                        name_node = self.create_node(
                            SBN_NODE_TYPE.CONSTANT,
                            name,
                            {"comment": comment},
                        )
                        role_edge = self.create_edge(
                            self._active_sense_id,
                            name_node[0],
                            SBN_EDGE_TYPE.ROLE,
                            token,
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
                            self._active_sense_id,
                            const_node[0],
                            SBN_EDGE_TYPE.ROLE,
                            token,
                        )

                        nodes.append(const_node)
                        edges.append(role_edge)
                else:
                    raise SBNError(
                        f"Invalid token found '{token}' in line: {sbn_line}"
                    )

        self.add_nodes_from(nodes)
        self.add_edges_from(edges)

        return self

    def from_ud(self, U: UDGraph) -> SBNGraph:
        self.__init_type_indices()

        starting_box = self.create_node(
            SBN_NODE_TYPE.BOX, self._active_box_token
        )
        nodes, edges = [starting_box], []
        ud_sbn_id_mapping = dict()

        for node_id, node_data in U.nodes.data():
            # distinguish names and other const types (?) or just use a single
            # const type for all constants, then the following is already correct
            if (
                node_data["type"] == UD_NODE_TYPE.ROOT
            ):  # TODO: move this out of here
                continue
            is_sense = U.out_degree(node_id) != 0
            node_type = (
                SBN_NODE_TYPE.SENSE if is_sense else SBN_NODE_TYPE.CONSTANT
            )

            node_data["type"] = node_type
            if lemma := node_data.get("lemma"):
                # TODO: part of speech mapping xpos -> wn
                node_data["token"] = f"{lemma}.n.01"
            new_node = self.create_node(
                node_type, node_data["token"], node_data
            )
            ud_sbn_id_mapping[node_id] = new_node[0]

            nodes.append(new_node)
            if is_sense:
                edges.append(
                    self.create_edge(
                        self._active_box_id,
                        new_node[0],
                        SBN_EDGE_TYPE.BOX_CONNECT,
                    )
                )

        # assume the boxes are not present
        for node_id, node_data in U.nodes.data():
            if (
                node_data["type"] == UD_NODE_TYPE.ROOT
            ):  # TODO: move this out of here
                continue
            for edge_id in U.out_edges(node_id):
                edge_data = U.edges.get(edge_id)
                from_id, to_id = edge_id
                edge_data["type"] = SBN_EDGE_TYPE.ROLE

                # refactor so token gets picked from meta or from id
                edges.append(
                    self.create_edge(
                        ud_sbn_id_mapping[from_id],
                        ud_sbn_id_mapping[to_id],
                        SBN_EDGE_TYPE.ROLE,
                        edge_data["token"],
                        edge_data,
                    )
                )
        self.add_nodes_from(nodes)
        self.add_edges_from(edges)
        print(self.nodes.data())

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
                **dict(
                    _id=str(edge_id),
                    type=type,
                    type_idx=edge_id[1],
                    token=token or str(edge_id),
                ),
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
                **dict(
                    _id=str(node_id),
                    type=type,
                    type_idx=node_id[1],
                    token=token or str(node_id),
                ),
                **meta,
            },
        )

    def to_sbn(self, path: PathLike, add_comments: bool = False) -> PathLike:
        """Writes the SBNGraph to an file in sbn format"""
        path = (
            Path(path) if str(path).endswith(".sbn") else Path(f"{path}.sbn")
        )

        path.write_text(self.to_sbn_string(add_comments))
        return path

    def to_sbn_string(self, add_comments: bool = False) -> str:
        """Creates a string in sbn format from the SBNGraph"""
        result = []
        sense_idx_map: Dict[SBN_ID, int] = dict()
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
                            "Found box connected to multiple boxes, is that possible?"
                        )
                    else:
                        box_box_connect_to_insert = edge_data["token"]

                # if to_node_type == SBN_NODE_TYPE.SENSE:
                if to_node_type in (
                    SBN_NODE_TYPE.SENSE,
                    SBN_NODE_TYPE.CONSTANT,
                ):
                    if to_node_id in sense_idx_map:
                        raise SBNError(
                            "Ambiguous sense id found, should not be possible"
                        )

                    sense_idx_map[to_node_id] = line_idx
                    temp_line_result = [to_node_id]
                    for sense_edge_id in self.out_edges(to_node_id):
                        _, sense_to_id = sense_edge_id

                        sense_edge_data = self.edges.get(sense_edge_id)
                        if sense_edge_data["type"] not in (
                            SBN_EDGE_TYPE.ROLE,
                            SBN_EDGE_TYPE.DRS_OPERATOR,
                        ):
                            raise SBNError(
                                f"Invalid sense edge connect found: {sense_edge_data['type']}"
                            )

                        temp_line_result.append(sense_edge_data["token"])

                        sense_node_to_data = self.nodes.get(sense_to_id)
                        sense_node_to_type = sense_node_to_data["type"]
                        if sense_node_to_type not in (
                            SBN_NODE_TYPE.CONSTANT,
                            SBN_NODE_TYPE.SENSE,
                        ):
                            raise SBNError(
                                f"Invalid sense node connect found: {sense_node_to_type}"
                            )

                        if sense_node_to_type == SBN_NODE_TYPE.SENSE:
                            temp_line_result.append(sense_to_id)
                        else:
                            temp_line_result.append(
                                sense_node_to_data["token"]
                            )

                    result.append(temp_line_result)
                    line_idx += 1
                elif to_node_type == SBN_NODE_TYPE.BOX:
                    pass
                else:
                    raise SBNError(f"Invalid node id found: {to_node_id}")

            if box_box_connect_to_insert:
                result.append([box_box_connect_to_insert, "-1"])

        # Resolve the indices and the correct sense tokens and create the sbn
        # line strings for the final string
        final_result = []
        current_sense_idx = 0
        for line_idx, line in enumerate(result):
            tmp_line = []
            comment_for_line = None

            for token_idx, token in enumerate(line):
                # There can never be an index at the first token of a line, so
                # always start at the second token.
                if token_idx == 0:
                    # It is a sense id that needs to be converted to a token
                    if token in sense_idx_map:
                        node_data = self.nodes.get(token)
                        tmp_line.append(node_data["token"])
                        comment_for_line = comment_for_line or (
                            node_data["comment"]
                            if "comment" in node_data
                            else None
                        )
                        current_sense_idx += 1
                    # It is a regular token
                    else:
                        tmp_line.append(token)
                # It is a sense which needs to be resolved to an index
                elif token in sense_idx_map:
                    target = sense_idx_map[token] - current_sense_idx + 1
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
            # TODO: vertically align tokens just as in the dataset?
            # See: https://docs.python.org/3/library/string.html#format-specification-mini-language
            final_result.append(" ".join(tmp_line))

        sbn_string = "\n".join(final_result)
        return sbn_string

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
        for grew_node_id, (node_data, grew_edges) in grew_graph.items():
            is_leaf = len(grew_edges) == 0
            node = self.create_node(
                SBN_NODE_TYPE.CONSTANT if is_leaf else SBN_NODE_TYPE.SENSE,
                # TODO: make sure in pre-processing 'type' and 'token' are added
                node_data.get("token", None) or node_data.get("lemma", None),
                node_data,
            )
            nodes.append(node)
            id_mapping[grew_node_id] = node[0]

            if not is_leaf:
                box_edge = self.create_edge(
                    self._active_box_id,
                    self._active_sense_id,
                    SBN_EDGE_TYPE.BOX_CONNECT,
                )
                edges.append(box_edge)

        for grew_from_node_id, (_, grew_edges) in grew_graph.items():
            for edge_name, grew_to_node_id in grew_edges:
                # TODO: introduce new type (NOT_LABELED_YET) in case it's not a
                # role or drs operator (so still a deprel)?
                edge_type = (
                    SBN_EDGE_TYPE.ROLE
                    if edge_name in SBNSpec.ROLES
                    else SBN_EDGE_TYPE.DRS_OPERATOR
                )
                edge = self.create_edge(
                    id_mapping[grew_from_node_id],
                    id_mapping[grew_to_node_id],
                    edge_type,
                    edge_name,
                )
                edges.append(edge)

        self.add_nodes_from(nodes)
        self.add_edges_from(edges)

        return self

    def to_amr(self, path: PathLike, add_comments: bool = False) -> PathLike:
        """Writes the SBNGraph to an file in AMR format"""
        path = (
            Path(path) if str(path).endswith(".amr") else Path(f"{path}.amr")
        )

        path.write_text(self.to_amr_string(add_comments))
        return path

    def to_amr_string(self, add_comments: bool = False) -> str:
        """Creates a string in amr format from the SBNGraph"""
        # Maybe use penman library to test validity
        # import penman

        # Make a copy just in case since strange side-effects such as token
        # changes are no fun to debug.
        G = deepcopy(self)
        var_ids = []

        for node in G.nodes:
            abbr = G.nodes[node]["token"].replace('"', "").strip()[0].lower()

            if abbr not in var_ids:
                G.nodes[node]["var_id"] = abbr
                var_ids.append(abbr)
            else:
                previous = [x for x in var_ids if x.startswith(abbr)]
                G.nodes[node]["var_id"] = abbr + str(len(previous) + 1)
                var_ids.append(abbr)

        for edge in G.edges:
            if G.edges[edge]["type"] == SBN_EDGE_TYPE.BOX_CONNECT:
                G.edges[edge]["token"] = "member"

        def __to_amr_str(
            S: SBNGraph, current_node, visited, text_format, tabs
        ):
            node_data = S.nodes[current_node]
            if current_node not in visited:
                text_format += f'({node_data["var_id"]} / {node_data["token"]}'

                if S.out_degree(current_node) == 0:
                    text_format += ")"
                    visited.add(current_node)
                else:
                    indents = tabs * "\t"
                    for edge_id in S.edges(current_node):
                        _, child_node = edge_id
                        text_format += (
                            f'\n{indents}:{S.edges[edge_id]["token"]} '
                        )
                        text_format = __to_amr_str(
                            S, child_node, visited, text_format, tabs + 1
                        )
                    text_format += ")"
                    visited.add(current_node)
            else:
                text_format += node_data["var_id"]
            return text_format

        # For now assume there always is the starting box to server as the "root"
        # (not really since it's a DAG, but for AMR a starting point is needed)
        starting_node = (SBN_NODE_TYPE.BOX, 0)
        return __to_amr_str(G, starting_node, set(), "", 1)

    def __init_type_indices(self):
        self.type_indices = {
            SBN_NODE_TYPE.SENSE: 0,
            SBN_NODE_TYPE.CONSTANT: 0,
            SBN_NODE_TYPE.BOX: 0,
            SBN_EDGE_TYPE.ROLE: 0,
            SBN_EDGE_TYPE.DRS_OPERATOR: 0,
            SBN_EDGE_TYPE.BOX_CONNECT: 0,
            SBN_EDGE_TYPE.BOX_BOX_CONNECT: 0,
        }

    def _id_for_type(
        self, type: Union[SBN_NODE_TYPE, SBN_EDGE_TYPE]
    ) -> SBN_ID:
        _id = (type, self.type_indices[type])
        self.type_indices[type] += 1
        return _id

    @property
    def _active_sense_id(self) -> SBN_ID:
        return (
            SBN_NODE_TYPE.SENSE,
            self.type_indices[SBN_NODE_TYPE.SENSE] - 1,
        )

    @property
    def _active_box_id(self) -> SBN_ID:
        return (SBN_NODE_TYPE.BOX, self.type_indices[SBN_NODE_TYPE.BOX] - 1)

    @property
    def _active_box_token(self) -> str:
        return f"B-{self.type_indices[SBN_NODE_TYPE.BOX]}"

    @staticmethod
    def _node_label(node_data) -> str:
        return node_data["token"]
        # return f'{node_data["type"].value}\n{node_data["token"]}'

    @staticmethod
    def _edge_label(edge_data) -> str:
        return edge_data["token"]
        # return f'{edge_data["type"].value}\n{edge_data["token"]}'

    @property
    def type_style_mapping(self):
        """Style per node type to use in dot export"""
        return {
            SBN_NODE_TYPE.SENSE: {},
            SBN_NODE_TYPE.CONSTANT: {"shape": "none"},
            SBN_NODE_TYPE.BOX: {"shape": "box"},
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
