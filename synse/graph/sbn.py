import re
from typing import Any, Dict, List, Optional, Tuple, Union

from synse.graph.base import BaseGraph

__all__ = ["SBN_NODE_TYPE", "SBN_EDGE_TYPE", "SBNGraph", "split_comments"]

# Tried with enums, but those get somewhat messed up in the json serialization
class SBN_NODE_TYPE(str):
    """Node types"""

    SENSE = "wordnet-sense"
    NAME_CONSTANT = "name-constant"
    CONSTANT = "constant"
    BOX = "box"


class SBN_EDGE_TYPE(str):
    """Edge types"""

    ROLE = "role"
    BOX_CONNECT = "box-connect"


# Node / edge ids, unique combination of type and index / count for the current
# document.
SBN_ID = Tuple[Union[SBN_NODE_TYPE, SBN_EDGE_TYPE], int]


class SBNSpec:
    # Whitespace is essential since there can be % signs in sense ids and comments
    SBN_COMMENT = r" % "
    SBN_COMMENT_LINE = r"%%%"

    NEW_BOX_INDICATORS = "|".join(
        [
            "ALTERNATION",
            "ATTRIBUTION",
            "CONDITION",
            "CONSEQUENCE",
            "CONTINUATION",
            "CONTRAST",
            "EXPLANATION",
            "NECESSITY",
            "NEGATION",
            "POSSIBILITY",
            "PRECONDITION",
            "RESULT",
            "SOURCE",
        ]
    )
    NEW_BOX_PATTERN = re.compile(NEW_BOX_INDICATORS)

    # The lemma match might seem loose, however there can be a lot of different
    # characters in there: 'r/2.n.01', 'ø.a.01', 'josé_maria_aznar.n.01'
    WORDNET_SENSE_PATTERN = re.compile(r"(.+)\.(n|v|a|r)\.(\d+)")
    INDEX_PATTERN = re.compile(r"((-|\+)\d)")
    NAME_CONSTANT_PATTERN = re.compile(r"\"(.+)\"|\"(.+)")

    # Special constants at the 'ending' nodes
    CONSTANTS = "|".join(
        [
            "speaker",
            "hearer",
            "now",
            "unknown_ref",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
    )

    # "'2008'" or "'196X'"" for instance, always in single quotes
    YEAR_CONSTANT = r"\'([\dX]{4})\'"

    # Can be "?", single "+/-" or unsigned digit (explicitly signed digits are
    # assumed to be indices and are matched first!)
    QUANTITY_CONSTANT = r"[\+\-\d\?]"

    # "Tom got an A on his exam": Value -> "A" NOTE: arguably better to catch
    # this with roles, but all other constants are caught.
    VALUE_CONSTANT = r"^[A-Z]$"

    # TODO: add named groups to regex so more specific constant types are kept
    CONSTANTS_PATTERN = re.compile(
        "|".join([YEAR_CONSTANT, QUANTITY_CONSTANT, VALUE_CONSTANT, CONSTANTS])
    )

    # Sense indices start at 0
    MIN_SENSE_IDX = 0


def split_comments(sbn_string: str) -> List[Tuple[str, Optional[str]]]:
    """
    Helper to remove starting comments and split the actual sbn and
    trailing comments per line. Empty comments are converted to None.
    """
    # First split everything in lines
    split_lines = sbn_string.rstrip("\n").split("\n")

    # Separate the actual SBN and the comments
    temp_lines = []
    for line in split_lines:
        # discarded here.
        if line.startswith(SBNSpec.SBN_COMMENT_LINE):
            continue

        # Split lines in (<SBN-line>, <comment>) and filter out empty
        # comments
        items = line.split(SBNSpec.SBN_COMMENT)
        assert len(items) == 2, f"Invalid comment format found: {line}"

        sbn, comment = [item.strip() for item in items]
        temp_lines.append((sbn, comment or None))

    return temp_lines


class SBNGraph(BaseGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    def from_string(self, input_string: str):
        """Construct a graph from a single SBN string."""
        lines = split_comments(input_string)

        self.type_indices = {
            SBN_NODE_TYPE.SENSE: 0,
            SBN_NODE_TYPE.NAME_CONSTANT: 0,
            SBN_NODE_TYPE.CONSTANT: 0,
            SBN_NODE_TYPE.BOX: 0,
            SBN_EDGE_TYPE.ROLE: 0,
            SBN_EDGE_TYPE.BOX_CONNECT: 0,
        }

        starting_box = self.create_node(SBN_NODE_TYPE.BOX)

        nodes, edges = [starting_box], []

        max_wn_idx = len(lines) - 1

        # Not really a stack, asserts fail if it has > 1 item, but it gets
        # treated as a stack to catch possible errors.
        to_do_stack = []
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
                        self._active_box_id(),
                        self._active_sense_id(),
                        SBN_EDGE_TYPE.BOX_CONNECT,
                    )

                    nodes.append(sense_node)
                    edges.append(box_edge)
                elif SBNSpec.NEW_BOX_PATTERN.match(token):
                    # In the entire dataset there are no indices for box
                    # references other than -1. Maybe they are needed later and
                    # the assert triggers if something different comes up.
                    box_index = int(tokens.pop(0))
                    assert (
                        box_index == -1
                    ), f"Unexpected box index found {box_index}"

                    current_box_id = self._active_box_id()

                    # Connect the current box to the one indicated by the index
                    new_box = self.create_node(SBN_NODE_TYPE.BOX)
                    box_edge = self.create_edge(
                        current_box_id,
                        self._active_box_id(),
                        SBN_EDGE_TYPE.BOX_CONNECT,
                        token,
                    )

                    nodes.append(new_box)
                    edges.append(box_edge)
                elif index_match := SBNSpec.INDEX_PATTERN.match(token):
                    idx = int(index_match.group(0))
                    active_id = self._active_sense_id()
                    target_idx = active_id[1] + idx
                    to_id = (active_id[0], target_idx)

                    assert (
                        len(to_do_stack) == 1
                    ), f'Error parsing index step "{token}" in: {sbn_line}'
                    target = to_do_stack.pop()

                    if SBNSpec.MIN_SENSE_IDX <= target_idx <= max_wn_idx:
                        role_edge = self.create_edge(
                            self._active_sense_id(),
                            to_id,
                            SBN_EDGE_TYPE.ROLE,
                            target,
                        )

                        edges.append(role_edge)
                    else:
                        # This is special case where a constant looks like an
                        # idx. Example:
                        # pmb-4.0.0/data/en/silver/p15/d3131/en.drs.sbn
                        # This is detected by checking if the provided index
                        # points at an 'impossible' line (sense) in the file.
                        const_node = self.create_node(
                            SBN_NODE_TYPE.CONSTANT, token, {"comment": comment}
                        )
                        role_edge = self.create_edge(
                            self._active_sense_id(),
                            const_node[0],
                            SBN_EDGE_TYPE.ROLE,
                            target,
                        )
                        box_edge = self.create_edge(
                            self._active_box_id(),
                            const_node[0],
                            SBN_EDGE_TYPE.BOX_CONNECT,
                        )

                        nodes.append(const_node)
                        edges.append(role_edge)
                        edges.append(box_edge)
                elif SBNSpec.NAME_CONSTANT_PATTERN.match(token):
                    name_parts = [token]

                    # Some names contain whitspace and need to be reconstructed
                    while not token.endswith('"'):
                        token = tokens.pop(0)
                        name_parts.append(token)

                    # This is faster than constantly creating new strings
                    name = " ".join(name_parts)

                    # Should be the edge linking this node to the previous
                    assert (
                        len(to_do_stack) == 1
                    ), f'Error parsing name step "{token}" in: {sbn_line}'
                    target = to_do_stack.pop()

                    name_node = self.create_node(
                        SBN_NODE_TYPE.NAME_CONSTANT, name, {"comment": comment}
                    )
                    role_edge = self.create_edge(
                        self._active_sense_id(),
                        name_node[0],
                        SBN_EDGE_TYPE.ROLE,
                        target,
                    )
                    box_edge = self.create_edge(
                        self._active_box_id(),
                        name_node[0],
                        SBN_EDGE_TYPE.BOX_CONNECT,
                    )

                    nodes.append(name_node)
                    edges.append(role_edge)
                    edges.append(box_edge)
                elif constant_match := SBNSpec.CONSTANTS_PATTERN.match(token):
                    assert (
                        len(to_do_stack) == 1
                    ), f'Error parsing const step "{token}" in: {sbn_line}'
                    target = to_do_stack.pop()

                    name_node = self.create_node(
                        SBN_NODE_TYPE.CONSTANT,
                        constant_match.group(0),
                        {"comment": comment},
                    )
                    role_edge = self.create_edge(
                        self._active_sense_id(),
                        name_node[0],
                        SBN_EDGE_TYPE.ROLE,
                        target,
                    )
                    box_edge = self.create_edge(
                        self._active_box_id(),
                        name_node[0],
                        SBN_EDGE_TYPE.BOX_CONNECT,
                    )

                    nodes.append(name_node)
                    edges.append(role_edge)
                    edges.append(box_edge)
                else:
                    # At all times the to_do_stack should be empty or have one
                    # token in it. The asserts above ensure this. The tokens
                    # that end up here are the roles that then get consumed by
                    # the indices.
                    to_do_stack.append(token)

            tok_count += 1

            if len(to_do_stack) > 0:
                raise ValueError(f"Unhandled tokens left: {to_do_stack}\n")

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

    def _active_sense_id(self) -> SBN_ID:
        return (
            SBN_NODE_TYPE.SENSE,
            self.type_indices[SBN_NODE_TYPE.SENSE] - 1,
        )

    def _active_box_id(self) -> SBN_ID:
        return (SBN_NODE_TYPE.BOX, self.type_indices[SBN_NODE_TYPE.BOX] - 1)
