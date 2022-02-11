import random
import re
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Optional

from regraph import NXGraph, Rule, plot_graph, plot_instance, plot_rule

SBN_COMMENT = '%'
WORDNET_SENSE_PATTERN = re.compile(r'([a-z_]+)\.(n|v|a|r)\.(\d+)')
INDEX_PATTERN = re.compile(r'(-|\+)(\d)')


@dataclass
class Node:
    type: str
    meta: Optional[Dict[str, Any]] = None

    def __hash__(self) -> int:
        return hash(self.type, self.meta)


@dataclass
class Edge:
    type: str
    from_node: str
    to_node: str
    meta: Optional[Dict[str, Any]] = None

    def __hash__(self) -> int:
        return hash(self.type, self.from_node, self.to_node, self.meta)


def parse_sbn(input_string):
    nodes, edges = [], []
    split_lines = input_string.strip().split('\n')

    print(input_string + '\n\n')

    parsed_lines = []
    for line in split_lines:
        if line.startswith(SBN_COMMENT):
            continue

        content = [i.strip() for i in line.split(SBN_COMMENT)]
        comment = content.pop() if len(content) > 1 else None
        data = content[0]
        items = data.split()
        parsed_lines.append((items, comment))

    seen_senses = set()
    sense_count = 1

    for line_idx, (tokens, comment) in enumerate(parsed_lines):
        for token_idx, token in enumerate(tokens):
            if result := WORDNET_SENSE_PATTERN.match(token):
                # Use hash or generate uuid or something instead of this.
                # This also doesn't work for the edges at the moment
                if token in seen_senses:
                    node_id = f'{token}-{sense_count}'
                    sense_count += 1
                else:
                    node_id = token
                seen_senses.add(token)

                nodes.append(Node(
                    node_id,
                    'wordnet_sense',
                    {
                        'lemma': result.group(1),
                        'pos': result.group(2),
                        'id': result.group(3),
                        'comment': comment
                    }
                ))
            elif index := INDEX_PATTERN.match(token):
                index = int(index.group(0))
                # Not 100% sure if this direction is always correct (and if
                # it even needs to be directed?)
                if index < 0:
                    edge = Edge(
                        tokens[token_idx - 1],  # id
                        tokens[token_idx - 1],  # id
                        parsed_lines[line_idx + index][0][0],
                        tokens[0],
                    )
                else:
                    edge = Edge(
                        tokens[token_idx - 1],  # id
                        tokens[token_idx - 1],  # id
                        tokens[0],
                        parsed_lines[line_idx + index][0][0],
                    )
                edges.append(edge)
            # Need to add relations/properties such as EQU, CONTINUATION, etc.
            # man.n.01                                % A man    [0-5]
            # time.n.08   EQU now                     % is       [6-8]
            # open.v.01   Agent -2 Time -1 Patient +1 % opening  [9-16]
            # soda.n.02                               % a soda   [17-23]
            #             CONTINUATION -1             % and      [24-27]
            # time.n.08   EQU now                     %
            # drink.v.01  Agent -5 Patient -2 Time -1 % drinking [28-36]
            # entity.n.01 EQU -3                      % it       [37-39]

    return nodes, edges


def get_sbn_files(folder_path):
    return Path(folder_path).glob('**/*.sbn')


def get_random_sbn_filepath(folder_path):
    return random.choice(list(get_sbn_files(folder_path)))


def main():
    filepath = get_random_sbn_filepath(sys.argv[1])

    with open(filepath) as f:
        nodes, edges = parse_sbn(f.read())

    print(nodes)

    graph = NXGraph()

    # This is horrible, just need to use dicts probably
    graph.add_nodes_from([
        (
            n.id,
            {field: getattr(n, field) for field in n.__dataclass_fields__}
        )
        for n in nodes
    ])
    graph.add_edges_from([
        (
            e.from_node, e.to_node,
            {field: getattr(e, field) for field in e.__dataclass_fields__}
        )
        for e in edges
    ])

    title_words = []
    for w in [n.meta['comment'] for n in nodes]:
        # Also not great, but just for testing
        words = w.split('[')[0].strip()
        title_words.append(words)
    plot_graph(graph, title=' '.join(title_words))


if __name__ == '__main__':
    main()
