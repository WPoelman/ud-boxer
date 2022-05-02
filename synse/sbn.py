from __future__ import annotations

import json
import logging
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import penman

from synse.base import BaseEnum, BaseGraph
from synse.config import Config
from synse.grew_spec import GrewSpec
from synse.penman_model import pm_model
from synse.sbn_spec import SBNError, SBNSpec, split_comments, split_wn_sense
from synse.ud_spec import UPOS_WN_POS_MAPPING

logger = logging.getLogger(__name__)


__all__ = [
    "SBN_NODE_TYPE",
    "SBN_EDGE_TYPE",
    "SBNGraph",
    "sbn_graphs_are_isomorphic",
]

# TODO: move this to a better place + don't use older mappings
# just for testing purposes now. Maybe move to GREW class?
with open(Config.EDGE_MAPPINGS_PATH) as f:
    # Sort options so the most frequent mapping is at the front
    EDGE_MAPPINGS = {
        k: sorted(list(v.items()), key=lambda i: i[1], reverse=True)
        for k, v in json.load(f).items()
    }


class SBN_NODE_TYPE(BaseEnum):
    """Node types"""

    SENSE = "sense"
    CONSTANT = "constant"
    BOX = "box"


class SBN_EDGE_TYPE(BaseEnum):
    """Edge types"""

    ROLE = "role"
    DRS_OPERATOR = "drs-operator"
    BOX_CONNECT = "box-connect"
    BOX_BOX_CONNECT = "box-box-connect"


# These are the protected fields in the node and edge data that need special
# care in certain places, such as when merging SBNGraphs.
PROTECTED_FIELDS = ["_id", "type", "type_idx", "token"]

# Node / edge ids, unique combination of type and index / count for the current
# document.
SBN_ID = Tuple[Union[SBN_NODE_TYPE, SBN_EDGE_TYPE], int]


class SBNGraph(BaseGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.is_dag: bool = None

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
                            # This is special case where a constant looks like
                            # an idx. Example:
                            # pmb-4.0.0/data/en/silver/p15/d3131/en.drs.sbn
                            # This is detected by checking if the provided
                            # index points at an 'impossible' line (sense) in
                            # the file.
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
        for grew_node_id, (node_data, grew_edges) in grew_graph.items():
            if not (
                (node_type_raw := node_data.get("type", None))
                and (node_tok := node_data.get("token", None))
            ):
                raise SBNError(
                    f"All nodes need the 'type' and 'token' features.\n"
                    f"Node data: {node_data}"
                )

            # Try to get the correct type from the provided data
            node_type = SBN_NODE_TYPE.from_str(node_type_raw)

            # Otherwise try to figure out what this is
            if not node_type and node_type_raw == GrewSpec.UNDEFINED:
                # The token was added, but forgot to add type.
                if SBNSpec.WORDNET_SENSE_PATTERN.match(node_tok):
                    node_type = SBN_NODE_TYPE.SENSE
                # Try to format the token as a sense. This assumes unwanted
                # nodes (DET, PUNCT, AUX) are already removed.
                elif "upos" in node_data:
                    # TODO: some POS tags indicate constants (NUM, PROPN, etc)
                    # Maybe fix that here as well.
                    wn_pos = UPOS_WN_POS_MAPPING[node_data["upos"]]
                    node_tok = f"{node_tok.lower()}.{wn_pos}.01"
                    node_type = SBN_NODE_TYPE.SENSE
                    node_data["token"] = node_tok
                # When the previous checks cannot determine if it's a sense or
                # not, consider it to be a constant.
                else:
                    node_type = SBN_NODE_TYPE.CONSTANT
            else:
                raise SBNError(f"Unusable node type: {node_type}")

            node = self.create_node(node_type, node_tok, node_data)
            id_mapping[grew_node_id] = node[0]
            nodes.append(node)

        for grew_from_node_id, (_, grew_edges) in grew_graph.items():
            # NOTE: if we also introduce boxes on the grew side, we need to
            # figure out how to connect those here.
            if id_mapping[grew_from_node_id][0] == SBN_NODE_TYPE.BOX:
                box_box_edge = self.create_edge(
                    starting_box[0],
                    id_mapping[grew_from_node_id],
                    SBN_EDGE_TYPE.BOX_BOX_CONNECT,
                    self.nodes[id_mapping[grew_from_node_id]]["token"],
                )
                edges.append(box_box_edge)

            if id_mapping[grew_from_node_id][0] == SBN_NODE_TYPE.SENSE:
                box_edge = self.create_edge(
                    starting_box[0],
                    id_mapping[grew_from_node_id],
                    SBN_EDGE_TYPE.BOX_CONNECT,
                )
                edges.append(box_edge)

            for edge_name, grew_to_node_id in grew_edges:
                edge_type = None

                if edge_name in SBNSpec.ROLES:
                    edge_type = SBN_EDGE_TYPE.ROLE
                elif edge_name in SBNSpec.DRS_OPERATORS:
                    edge_type = SBN_EDGE_TYPE.DRS_OPERATOR
                elif edge_name in SBNSpec.NEW_BOX_INDICATORS:
                    edge_type = SBN_EDGE_TYPE.BOX_CONNECT
                # The type cannot be determined from the name, figure out what
                # an appropriate edge label might be.
                else:
                    if edge_name in EDGE_MAPPINGS:
                        edge_name = EDGE_MAPPINGS[edge_name][0][0]
                        # TODO: This type info should probably be included in
                        # the mappings.
                        if edge_name in SBNSpec.ROLES:
                            edge_type = SBN_EDGE_TYPE.ROLE
                        elif edge_name in SBNSpec.DRS_OPERATORS:
                            edge_type = SBN_EDGE_TYPE.DRS_OPERATOR
                        else:
                            raise SBNError(f"Invalid mapping {edge_name}!")
                    else:
                        # This is pure guesswork, figure something better out
                        # here, all edge mapping based on triples?
                        # POS -> deprel -> POS? Default can be most frequent
                        # role probably.
                        edge_type = SBN_EDGE_TYPE.ROLE
                        edge_name = Config.DEFAULT_ROLE

                edge = self.create_edge(
                    id_mapping[grew_from_node_id],
                    id_mapping[grew_to_node_id],
                    edge_type,
                    edge_name,
                )
                edges.append(edge)

        self.add_nodes_from(nodes)
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
                "token": token or str(edge_id),
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

    def to_sbn(self, path: PathLike, add_comments: bool = False) -> PathLike:
        """Writes the SBNGraph to a file in sbn format"""
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
                            "Found box connected to multiple boxes, "
                            "is that possible?"
                        )
                    else:
                        box_box_connect_to_insert = edge_data["token"]

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
                                f"Invalid sense edge connect found: "
                                f"{sense_edge_data['type']}"
                            )

                        temp_line_result.append(sense_edge_data["token"])

                        sense_node_to_data = self.nodes.get(sense_to_id)
                        sense_node_to_type = sense_node_to_data["type"]
                        if sense_node_to_type == SBN_NODE_TYPE.SENSE:
                            temp_line_result.append(sense_to_id)
                        elif sense_node_to_type == SBN_NODE_TYPE.CONSTANT:
                            temp_line_result.append(
                                sense_node_to_data["token"]
                            )
                        else:
                            raise SBNError(
                                f"Invalid sense node connect found: "
                                f"{sense_node_to_type}"
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
        for line in result:
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

            # This is a bit of trickery to vertically align senses just as in
            # the PMB dataset.
            if len(tmp_line) == 1:
                final_result.append((tmp_line[0], " "))
            else:
                final_result.append((tmp_line[0], " ".join(tmp_line[1:])))

        # More formatting and alignment trickery.
        max_sense_len = max(len(s) for s, _ in final_result) + 1
        sbn_string = "\n".join(
            f"{sense: <{max_sense_len}}{rest}".rstrip(" ")
            for sense, rest in final_result
        )

        return sbn_string

    def to_penman(self, path: PathLike, split_sense: bool = False) -> PathLike:
        """
        Writes the SBNGraph to a file in Penman (AMR-like) format.

        See `to_penman_string` for an explanation of `split_sense`.
        """
        path = (
            Path(path)
            if str(path).endswith(".penman")
            else Path(f"{path}.penman")
        )

        path.write_text(self.to_penman_string(split_sense))
        return path

    def to_penman_string(self, lenient: bool = False) -> str:
        """
        Creates a string in Penman (AMR-like) format from the SBNGraph.

        The `lenient` flag indicates whether to include the sense number.
        If included, the evaluation indirectly also targets the task of word
        sense disambiguation, which might not be desirable. Example:
            Without sense:
                (b0 / "box"
                    :member (s0 / "sense"
                        :lemma (s1 / "person")
                        :pos (s2 / "n")
                        :sense (s3 / "01"))) # Would be excluded when lenient
        """
        if not self.is_dag:
            raise SBNError(
                "Exporting a cyclic SBN graph to Penman is not possible"
            )

        # Make a copy just in case since strange side-effects such as token
        # changes are no fun to debug.
        G = deepcopy(self)

        prefix_map = {
            SBN_NODE_TYPE.BOX: ["b", 0],
            SBN_NODE_TYPE.CONSTANT: ["c", 0],
            SBN_NODE_TYPE.SENSE: ["s", 0],
        }

        for node_id, node_data in G.nodes.items():
            pre, count = prefix_map[node_data["type"]]
            prefix_map[node_data["type"]][1] += 1  # type: ignore
            G.nodes[node_id]["var_id"] = f"{pre}{count}"

            # A box is always an instance of the same type (or concept), the
            # specification of what that type does is shown by the
            # box-box-connection, such as NEGATION or EXPLANATION.
            if node_data["type"] == SBN_NODE_TYPE.BOX:
                G.nodes[node_id]["token"] = "box"

        for edge in G.edges:
            # Add a proper token to the box connectors
            if G.edges[edge]["type"] == SBN_EDGE_TYPE.BOX_CONNECT:
                G.edges[edge]["token"] = "member"
            # Since the graph is possibly in the 'wrong' direction, convert
            # the non 'Of*' relations here.
            # elif G.edges[edge]["type"] == SBN_EDGE_TYPE.ROLE:
            # if G.edges[edge]["token"] in SBNSpec.REVERSABLE_ROLES:
            # G.edges[edge]["token"] = f'{G.edges[edge]["token"]}Of'

        def __to_penman_str(S: SBNGraph, current_n, visited, out_str, tabs):
            node_data = S.nodes[current_n]
            var_id = node_data["var_id"]
            if var_id in visited:
                out_str += var_id
                return out_str

            indents = tabs * "\t"
            node_tok = node_data["token"]
            if node_data["type"] == SBN_NODE_TYPE.SENSE:
                if not (components := split_wn_sense(node_tok)):
                    raise SBNError(f"Cannot split sense: {node_tok}")

                lemma, pos, sense = [self.quote(i) for i in components]

                out_str += f'({var_id} / {self.quote("sense")}'
                out_str += f"\n{indents}:lemma {lemma}"
                out_str += f"\n{indents}:pos {pos}"

                if not lenient:
                    out_str += f"\n{indents}:sense {sense}"
            # TODO: fix this, the generated parentheses are not always correct
            # elif node_tok in SBNSpec.CONSTANTS:
            #     out_str += f"{self.quote(node_tok)})"
            #     if S.out_degree(current_n) > 0:
            #         raise SBNError("A constant cannot have out edges.")
            else:
                out_str += f"({var_id} / {self.quote(node_tok)}"

            if S.out_degree(current_n) == 0:
                out_str += ")"
                visited.add(var_id)
            else:
                for edge_id in S.edges(current_n):
                    _, child_node = edge_id
                    out_str += f"\n{indents}:{S.edges[edge_id]['token']} "
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
    def quote(in_str: str) -> str:
        """Consistently quote a string with double quotes"""
        if in_str.startswith('"') and in_str.endswith('"'):
            return in_str

        if in_str.startswith("'") and in_str.endswith("'"):
            return f'"{in_str[1:-1]}"'

        return f'"{in_str}"'

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

    @staticmethod
    def _edge_label(edge_data) -> str:
        return edge_data["token"]

    @property
    def type_style_mapping(self):
        """Style per node type to use in dot export"""
        return {
            SBN_NODE_TYPE.SENSE: {},
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
