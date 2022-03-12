from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from uuid import uuid4

import networkx as nx

from synse.graph import BaseGraph
from synse.sbn.sbn_spec import SBNSpec, get_doc_id, split_comments


class SBN_NODE_TYPE(str, Enum):
    """Node types"""

    SENSE = "wordnet-sense"
    NAME_CONSTANT = "name-constant"
    CONSTANT = "constant"
    BOX = "box"


class SBN_EDGE_TYPE(str, Enum):
    """Edge types"""

    ROLE = "role"
    BOX_CONNECT = "box-connect"
    BOX_BOX_CONNECT = "box-box-connect"


# Node / edge ids, unique combination of type and index / count for the current
# document.
SBN_ID = Tuple[Union[SBN_NODE_TYPE, SBN_EDGE_TYPE], int]


class SBNGraph(BaseGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

        # TODO: maybe move to BaseGraph
        self.doc_id = None

    def from_path(self, path: PathLike, doc_id: str = None):
        """Construct a graph from the provided filepath."""
        sbn_str = Path(path).read_text()
        self.__try_to_set_doc_id(doc_id, path=path, sbn_str=sbn_str)
        return self.from_string(sbn_str)

    def from_string(self, input_string: str, doc_id: str = None):
        """Construct a graph from a single SBN string."""
        self.__try_to_set_doc_id(doc_id, sbn_str=input_string)

        lines = split_comments(input_string)

        assert lines, "SBN doc appears to be empty, cannot read from string"

        self.type_indices = {
            SBN_NODE_TYPE.SENSE: 0,
            SBN_NODE_TYPE.NAME_CONSTANT: 0,
            SBN_NODE_TYPE.CONSTANT: 0,
            SBN_NODE_TYPE.BOX: 0,
            SBN_EDGE_TYPE.ROLE: 0,
            SBN_EDGE_TYPE.BOX_CONNECT: 0,
            SBN_EDGE_TYPE.BOX_BOX_CONNECT: 0,
        }

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
                    # the assert triggers if something different comes up.
                    box_index = int(tokens.pop(0))
                    assert (
                        box_index == -1
                    ), f"Unexpected box index found {box_index}"

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
                elif token in SBNSpec.ROLES:
                    target = tokens.pop(0)

                    if index_match := SBNSpec.INDEX_PATTERN.match(target):
                        idx = int(index_match.group(0))
                        active_id = self._active_sense_id
                        target_idx = active_id[1] + idx
                        to_id = (active_id[0], target_idx)

                        if SBNSpec.MIN_SENSE_IDX <= target_idx <= max_wn_idx:
                            role_edge = self.create_edge(
                                self._active_sense_id,
                                to_id,
                                SBN_EDGE_TYPE.ROLE,
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
                                SBN_EDGE_TYPE.ROLE,
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
                            SBN_NODE_TYPE.NAME_CONSTANT,
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
                    raise ValueError(
                        f"Invalid token found '{token}' in line: {sbn_line}"
                    )

        self.add_nodes_from(nodes)
        self.add_edges_from(edges)

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
            dict(
                _id=str(edge_id),
                type=type,
                type_idx=edge_id[1],
                token=token or str(edge_id),
                **meta,
            ),
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
            dict(
                _id=str(node_id),
                type=type,
                type_idx=node_id[1],
                token=token or str(node_id),
                **meta,
            ),
        )

    def _id_for_type(
        self, type: Union[SBN_NODE_TYPE, SBN_EDGE_TYPE]
    ) -> SBN_ID:
        _id = (type, self.type_indices[type])
        self.type_indices[type] += 1
        return _id

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
                        raise AssertionError(
                            "Found box connected to multiple boxes, is that possible?"
                        )
                    else:
                        box_box_connect_to_insert = edge_data["token"]

                if to_node_type == SBN_NODE_TYPE.SENSE:
                    assert (
                        to_node_id not in sense_idx_map
                    ), "Ambiguous sense id found, should not be possible"

                    sense_idx_map[to_node_id] = line_idx
                    temp_line_result = [to_node_id]
                    for sense_edge_id in self.out_edges(to_node_id):
                        _, sense_to_id = sense_edge_id

                        sense_edge_data = self.edges.get(sense_edge_id)
                        assert (
                            sense_edge_data["type"] == SBN_EDGE_TYPE.ROLE
                        ), f"Invalid sense edge connect found: {sense_edge_data['type']}"

                        temp_line_result.append(sense_edge_data["token"])

                        sense_node_to_data = self.nodes.get(sense_to_id)
                        sense_node_to_type = sense_node_to_data["type"]
                        assert sense_node_to_type in (
                            SBN_NODE_TYPE.CONSTANT,
                            SBN_NODE_TYPE.NAME_CONSTANT,
                            SBN_NODE_TYPE.SENSE,
                        ), f"Invalid sense node connect found: {sense_node_to_type}"

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
                    raise ValueError(f"Invalid node id found: {to_node_id}")

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
                    tmp_line.append(
                        # In PMB dataset, an index of '0' is written as '+0',
                        # so do that here as well.
                        f"+{target}"
                        if target >= 0
                        else str(target)
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

    @property
    def type_style_mapping(self):
        """Style per node type to use in dot export"""
        return {
            SBN_NODE_TYPE.SENSE: {},
            SBN_NODE_TYPE.NAME_CONSTANT: {"shape": "none"},
            SBN_NODE_TYPE.CONSTANT: {"shape": "none"},
            SBN_NODE_TYPE.BOX: {"shape": "box"},
            SBN_EDGE_TYPE.ROLE: {},
            SBN_EDGE_TYPE.BOX_CONNECT: {"style": "dotted", "label": ""},
            SBN_EDGE_TYPE.BOX_BOX_CONNECT: {},
        }

    def __try_to_set_doc_id(
        self, doc_id: str = None, path: PathLike = None, sbn_str: str = None
    ):
        # TODO: probably remove this and always use uuid or force the user to
        # provide a valid id instead of the graph figuring it out.
        if self.doc_id:
            return

        if doc_id:
            self.doc_id = doc_id
        elif path:
            self.doc_id = get_doc_id(filepath=path)
        elif sbn_str:
            self.doc_id = get_doc_id(sbn_str=sbn_str)
        else:
            self.doc_id = f"no-id-received-{uuid4()}"


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
