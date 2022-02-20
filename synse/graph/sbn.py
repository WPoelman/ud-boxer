from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from synse.graph.base import BaseGraph
from synse.graph.sbn_spec import SBNSpec, get_doc_id, split_comments

__all__ = ["SBN_NODE_TYPE", "SBN_EDGE_TYPE", "SBNGraph"]


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
        if not self.doc_id and doc_id:
            self.doc_id = doc_id
        else:
            self.doc_id = get_doc_id(filepath=path)
        return self.from_string(Path(path).read_text(), doc_id)

    def from_string(self, input_string: str, doc_id: str = None):
        """Construct a graph from a single SBN string."""
        # TODO: maybe put this in a nicer spot (and possibly preserve comments
        # so the sbn can be reconstructed from the graph, including
        # the comments).
        if not self.doc_id and doc_id:
            self.doc_id = doc_id
        else:
            self.doc_id = get_doc_id(sbn_str=input_string)
        lines = split_comments(input_string)

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
