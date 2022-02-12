"""
    Author: Wessel Poelman
    Description: Script to parse SBN files into graphs.

    NOTE: Trying to put this into a more reusable format. 
"""
import re
import time
from argparse import ArgumentParser, Namespace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx

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

# "Tom got an A on his exam": Value -> "A" NOTE: arguably better to catch this
# with roles, but all other constants are caught.
VALUE_CONSTANT = r"^[A-Z]$"

# TODO: add named groups to regex so more specific constant types are kept
CONSTANTS_PATTERN = re.compile(
    "|".join([YEAR_CONSTANT, QUANTITY_CONSTANT, VALUE_CONSTANT, CONSTANTS])
)

# NOTE: probably nicer to use actual dataclasses, but since the indices
# are crucial, it becomes a bit more tricky to keep track of hashes and
# indices / ids. NetworkX does accept any hashable object, for later!
NODE = Tuple[int, Dict[str, Any]]
EDGE = Tuple[int, int, Dict[str, Any]]


class SBN_NODE(Enum):
    """Node types"""

    WORDNET_SENSE = "wordnet-sense"
    NAME_CONSTANT = "name-constant"
    CONSTANT = "name-constant"
    BOX = "box"


class SBN_EDGE(Enum):
    """Edge types"""

    ROLE = "role"
    BOX_CONNECT = "box-connect"


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--starting_path",
        type=str,
        default="data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold",
        help="Path to start recursively searching for sbn files.",
    )
    parser.add_argument(
        "-v",
        "--visualization",
        action="store_true",
        help="Show a visualization of the parsed graph.",
    )
    return parser.parse_args()


def parse_sbn(input_string: str) -> Tuple[List[NODE], List[EDGE]]:
    """Creates a graph from a single SBN string."""

    # First split everything in lines
    split_lines = input_string.rstrip("\n").split("\n")

    # Separate the actual SBN and the comments
    temp_lines = []
    for line in split_lines:
        # The starting lines contain the boxer command information, those are
        # discarded here.
        if line.startswith(SBN_COMMENT_LINE):
            continue

        # Split lines in [<SBN-line>, <comment>] and filter out empty comments
        content = [i.strip() for i in line.split(SBN_COMMENT)]
        comment = content.pop() if len(content) > 1 else None
        temp_lines.append((content[0], comment))

    # TODO: find nicer way to construct box ids (and nodes and edges in
    # general), probably with a graph object that handles this internally.
    active_box_id = ("B", 0)
    starting_box = (
        active_box_id,
        {"type": SBN_NODE.BOX, "token": "".join(map(str, active_box_id))},
    )

    nodes, edges = [starting_box], []
    const_node_id, wn_node_id = 1000, 0  # TODO: this is dumb, find nicer way

    min_wn_id = 0
    max_wn_id = len(temp_lines) - 1

    # Not really a stack, asserts fail if it has > 1 item, but it gets
    # treated as a stack to catch possible errors.
    to_do_stack = []
    for sbn_line, comment in temp_lines:
        tokens = sbn_line.split()

        tok_count = 0
        while len(tokens) > 0:
            # Try to 'consume' all tokens from left to right
            token: str = tokens.pop(0)

            # No need to check all tokens for this since only the first
            # could be a sense id.
            if tok_count == 0 and (
                sense_match := WORDNET_SENSE_PATTERN.match(token)
            ):
                nodes.append(
                    (
                        wn_node_id,
                        {
                            "type": SBN_NODE.WORDNET_SENSE,
                            "token": token,
                            "lemma": sense_match.group(1),
                            "pos": sense_match.group(2),
                            "id": sense_match.group(3),
                            "comment": comment,
                        },
                    )
                )
                edges.append(
                    (
                        active_box_id,
                        wn_node_id,
                        {"type": SBN_EDGE.BOX_CONNECT, "token": "box"},
                    )
                )

                wn_node_id += 1
            elif NEW_BOX_PATTERN.match(token):
                # The next token should be the index where this new box points
                # to.
                box_index = int(tokens.pop(0))

                # In the entire dataset there are no indices for box
                # references other than -1. Maybe they are needed later, for
                # now just assume this is correct (and the assert triggers if
                # something different comes up).
                assert (
                    box_index == -1
                ), f"Unexpected box index found {box_index}"

                # TODO: Again, find nicer way of doing this
                new_box_id = (active_box_id[0], active_box_id[1] + 1)
                new_box = (
                    new_box_id,
                    {
                        "type": SBN_NODE.BOX,
                        "token": "".join(map(str, new_box_id)),
                    },
                )
                nodes.append(new_box)

                # Connect the current box to the one indicated by the index.
                edges.append(
                    (
                        active_box_id,
                        new_box_id,
                        {"type": SBN_EDGE.BOX_CONNECT, "token": token},
                    )
                )

                active_box_id = new_box_id
            elif index_match := INDEX_PATTERN.match(token):
                index = int(index_match.group(0))
                to_id = wn_node_id - 1 + index

                assert (
                    len(to_do_stack) == 1
                ), f'Error parsing index step "{token}" in:\n{sbn_line}'
                target = to_do_stack.pop()

                if min_wn_id <= to_id <= max_wn_id:
                    edges.append(
                        (
                            wn_node_id - 1,
                            to_id,
                            {"type": SBN_EDGE.ROLE, "token": target},
                        )
                    )
                else:
                    # This is special case where a constant looks like an idx.
                    # Example: pmb-4.0.0/data/en/silver/p15/d3131/en.drs.sbn
                    # This is detected by checking if the provided index points
                    # at an 'impossible' line (sense) in the file.
                    nodes.append(
                        (
                            const_node_id,
                            {
                                "type": SBN_NODE.CONSTANT,
                                "token": token,
                                "comment": comment,
                            },
                        )
                    )
                    edges.append(
                        (
                            wn_node_id - 1,
                            const_node_id,
                            {"type": SBN_EDGE.ROLE, "token": target},
                        )
                    )
                    edges.append(
                        (
                            active_box_id,
                            const_node_id,
                            {"type": SBN_EDGE.BOX_CONNECT, "token": "box"},
                        )
                    )

                    const_node_id += 1
            elif NAME_CONSTANT_PATTERN.match(token):
                name_parts = [token]

                # Some names contain whitspace and need to be reconstructed
                while not token.endswith('"'):
                    token = tokens.pop(0)
                    name_parts.append(token)

                # This is faster than constantly creating new strings
                name = " ".join(name_parts)

                assert (
                    len(to_do_stack) == 1
                ), f'Error parsing name const step "{token}" in:\n{sbn_line}'
                # Should be the edge linking this node to the previous
                target = to_do_stack.pop()

                nodes.append(
                    (
                        const_node_id,
                        {
                            "type": SBN_NODE.NAME_CONSTANT,
                            "token": name,
                            "comment": comment,
                        },
                    )
                )
                edges.append(
                    (
                        wn_node_id - 1,
                        const_node_id,
                        {"type": SBN_EDGE.ROLE, "token": target},
                    )
                )
                edges.append(
                    (
                        active_box_id,
                        const_node_id,
                        {"type": SBN_EDGE.BOX_CONNECT, "token": "box"},
                    )
                )

                const_node_id += 1
            elif constant_match := CONSTANTS_PATTERN.match(token):
                assert (
                    len(to_do_stack) == 1
                ), f'Error parsing const step "{token}" in:\n{sbn_line}'
                target = to_do_stack.pop()

                nodes.append(
                    (
                        const_node_id,
                        {
                            "type": SBN_NODE.CONSTANT,
                            "token": constant_match.group(0),
                            "comment": comment,
                        },
                    )
                )
                edges.append(
                    (
                        wn_node_id - 1,
                        const_node_id,
                        {"type": SBN_EDGE.ROLE, "token": target},
                    )
                )
                edges.append(
                    (
                        active_box_id,
                        const_node_id,
                        {"type": SBN_EDGE.BOX_CONNECT, "token": "box"},
                    )
                )

                const_node_id += 1
            else:
                # At all times the to_do_stack should be empty or have one
                # token in it. The asserts above ensure this. The tokens that
                # end up here are the roles that then get consumed by the
                # indices.
                to_do_stack.append(token)

            tok_count += 1
        if len(to_do_stack) > 0:
            raise ValueError(f"Unhandled tokens left: {to_do_stack}\n")

    return nodes, edges


def main():
    args = get_args()

    start = time.perf_counter()
    total, failed = 0, 0
    for filepath in Path(args.starting_path).glob("**/*.sbn"):
        with open(filepath) as f:
            total += 1
            try:
                nodes, edges = parse_sbn(f.read())
                # This is just to test to make sure the nodes and edges get
                # parsed correctly and are in the proper format for nx.
                G = nx.Graph()
                G.add_edges_from(edges)
                G.add_nodes_from(nodes)

                node_labels = {n: data["token"] for n, data in G.nodes.items()}
                edge_labels = {n: data["token"] for n, data in G.edges.items()}

                # This is ugly at the moment, probably need to export to dot
                # and use pydot or pygraphviz to layout correctly and adopt
                # paper style of visualizations. For checking if everything
                # works, it is fine.
                if args.visualization:
                    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
                    nx.draw_networkx_labels(G, pos, labels=node_labels)
                    nx.draw_networkx_edge_labels(
                        G, pos, edge_labels=edge_labels
                    )
                    nx.draw(
                        G,
                        pos,
                        node_size=1500,
                        node_color="grey",
                        font_size=8,
                        font_weight="bold",
                    )
                    plt.show()
                    # plt.savefig("Graph.png", format="PNG")
                    # nx.drawing.nx_pydot.write_dot(G, 'networkx_graph.dot')
            except Exception as e:
                print(f"Unable to parse: {e}")
                print(filepath)
                failed += 1
                continue

    end = round(time.perf_counter() - start, 2)

    print(
        f"""

    Total files:            {total}
    Parsed without errors:  {total - failed}
    Parsed with errors:     {failed}
    Took {end} seconds

    """
    )


if __name__ == "__main__":
    main()
