'''

TODO:
    - Create special nodes for NEGATION that connect to correct previous
      'subgraph' (or previously parsed nodes), currently these are not
      parsed correctly.
'''
import re
import sys
import time

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx

# Whitespace is essential since there can be % signs in sense ids and comments!
SBN_COMMENT = r' % '
SBN_COMMENT_LINE = r'%%%'
CONSTANTS = '|'.join([
    'speaker', 'hearer', 'now', 'unknown_ref',
    'monday', 'tuesday', 'wednesday', 'thursday',
    'friday', 'saturday', 'sunday'
])

NEW_BOX_INDICATORS = '|'.join([
    'ALTERNATION', 'ATTRIBUTION', 'CONDITION', 'CONSEQUENCE', 'CONTINUATION',
    'CONTRAST', 'EXPLANATION', 'NECESSITY', 'NEGATION', 'POSSIBILITY',
    'PRECONDITION', 'RESULT', 'SOURCE'
])
NEW_BOX_PATTERN = re.compile(NEW_BOX_INDICATORS)

# The lemma match might seem loose, however there can be a lot of different
# characters in there: 'r/2.n.01', 'ø.a.01', 'josé_maria_aznar.n.01'
WORDNET_SENSE_PATTERN = re.compile(r'(.+)\.(n|v|a|r)\.(\d+)')
INDEX_PATTERN = re.compile(r'((-|\+)\d)')
NAME_CONSTANT_PATTERN = re.compile(r'\"(.+)\"|\"(.+)')

# "'2008'" or "'196X'"" for instance
YEAR_CONSTANT = r'\'([\dX]{4})\''

# Can be "?", single "+" or unsigned digit (explicitly signed digits are
# assumed to be indices and are matched first!)
QUANTITY_CONSTANT = r'[\+\-\d\?]'

# "Tom got an A on his exam": Value -> "A" NOTE: probably better to catch this
# with roles, but in all the pmb data this is quite rare.
VALUE_CONSTANT = r'^[A-Z]$'

# TODO: add named groups to regex so more specific constant types are kept
CONSTANTS_PATTERN = re.compile(
    '|'.join([YEAR_CONSTANT, QUANTITY_CONSTANT, VALUE_CONSTANT, CONSTANTS])
)

# NOTE: probably nicer to use actual dataclasses, but since the indices
# are crucial, it becomes a bit more tricky to keep track of hashes and
# indices / ids. NetworkX does accept any hashable object, for later!
NODE = Tuple[int, Dict[str, Any]]
EDGE = Tuple[int, int, Dict[str, Any]]


class SBN_NODE(Enum):
    WORDNET_SENSE = 'wordnet_sense'
    NAME_CONSTANT = 'name_constant'
    CONSTANT = 'name_constant'


def parse_sbn(input_string: str) -> Tuple[List[NODE], List[EDGE]]:
    # First split everything in lines
    split_lines = input_string.split('\n')

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

    nodes, edges = [], []
    const_node_id, wn_node_id = 1000, 0  # TODO: this is dumb, find nicer way

    # Not really a stack, if it has > 1 item multiple asserts fail, but it gets
    # treated as a stack
    to_do_stack = []
    for sbn_line, comment in temp_lines:
        tokens = sbn_line.split()

        while len(tokens) > 0:
            # Try to 'consume' all tokens from left to right
            token: str = tokens.pop(0)
            if sense_match := WORDNET_SENSE_PATTERN.match(token):
                nodes.append((
                    wn_node_id,
                    {
                        'type': SBN_NODE.WORDNET_SENSE,
                        'token': token,
                        'lemma': sense_match.group(1),
                        'pos': sense_match.group(2),
                        'id': sense_match.group(3),
                        'comment': comment
                    }
                ))
                wn_node_id += 1
            elif index_match := INDEX_PATTERN.match(token):
                index = int(index_match.group(0))

                assert len(to_do_stack) == 1, \
                    f'Error parsing index step "{token}" in:\n{sbn_line}'
                target = to_do_stack.pop()

                edges.append((
                    wn_node_id - 1,    # from
                    wn_node_id - 1 + index,  # to
                    {'type': target}
                ))
            elif NAME_CONSTANT_PATTERN.match(token):
                name_parts = [token]

                # Some names contain whitspace and need to be reconstructed
                while not token.endswith('\"'):
                    token = tokens.pop(0)
                    name_parts.append(token)

                # This is faster than constantly creating new strings
                name = ' '.join(name_parts)

                # Should be the edge linking this node to the previous
                assert len(to_do_stack) == 1, \
                    f'Error parsing name const step "{token}" in:\n{sbn_line}'
                target = to_do_stack.pop()

                nodes.append((
                    const_node_id,
                    {
                        'type': SBN_NODE.NAME_CONSTANT,
                        'token': name,
                        'comment': comment
                    }
                ))
                edges.append((wn_node_id - 1, const_node_id, {'type': target}))
                const_node_id += 1
            elif constant_match := CONSTANTS_PATTERN.match(token):
                # Should be the edge linking this node to the previous
                assert len(to_do_stack) == 1, \
                    f'Error parsing const step "{token}" in:\n{sbn_line}'
                target = to_do_stack.pop()

                nodes.append((
                    const_node_id,
                    {
                        'type': SBN_NODE.CONSTANT,
                        'token': constant_match.group(0),
                        'comment': comment
                    }
                ))
                edges.append((wn_node_id - 1, const_node_id, {'type': target}))
            else:
                to_do_stack.append(token)

        if len(to_do_stack) > 0:
            raise ValueError(f'Unhandled tokens left: {to_do_stack}\n')

            # NOTE: for now the focus is on the constants and everything
            # 'around' the roles. The roles are 'implicit' in a way.
            # This is some leftover stuff from experiments.
            # elif token.isupper() or SPECIAL_ROLE_PATTERN.match(token):
            #     # The next token should be a constant since indices get
            #     # consumed by the branch above
            #     target = tokens.pop(0)
            #     nodes.append((
            #         node_id,
            #         {
            #             'type': 'constant',
            #             'token': target,
            #             'comment': comment
            #         }
            #     ))
            #     edges.append((node_id - 1, node_id, {'type': token}))
            #     node_id += 1

    return nodes, edges


def main():
    test_path = sys.argv[1]

    start = time.perf_counter()
    total, failed = 0, 0
    for filepath in Path(test_path).glob('**/*.sbn'):
        with open(filepath) as f:
            total += 1
            try:
                nodes, edges = parse_sbn(f.read())
                G = nx.Graph()
                G.add_edges_from(edges)
                G.add_nodes_from(nodes)

                # # pos = nx.drawing.nx_pydot.pydot_layout(G)
                # pos = nx.spiral_layout(G)
                # node_labels = {
                #     n: data['token'] if 'token' in data else 'UNKNOWN!!!'
                #     for n, data in G.nodes.items()
                # }
                # edge_labels = {
                #     n: data['type'] if 'token' in data else 'UNKNOWN!!!'
                #     for n, data in G.edges.items()
                # }

                # nx.draw_networkx_labels(G, pos, labels=node_labels)
                # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

                # nx.drawing.nx_pydot.write_dot(G, 'networkx_graph.dot')
            except Exception as e:
                print(e)
                print(filepath)
                failed += 1
                continue

    end = round(time.perf_counter() - start, 2)

    print(f'''

    Total files:            {total}
    Parsed without errors:  {total - failed}
    Parsed with errors:     {failed}
    Took {end} seconds

    ''')


if __name__ == '__main__':
    main()
