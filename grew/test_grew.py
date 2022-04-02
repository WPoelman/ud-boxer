import json
import shutil
import sys
from pathlib import Path

import grew


def get_text_format(graph, current_node, visited, text_format, tabs):
    """Takes a graph in GREW format and produces its textual representation

    Parameters: graph - a preprocessed graph in GREW format (needs to have abbreviations
    and a list of incoming relations for each node)
                current_node - the current node for the recursion
                visited - a list of already visited nodes
                text_format - the text format so far
                tabs - the number of tabs to be put at the beginning of each new line

    Output: the visited nodes and text format of the graph so far
    """
    # pp.pprint(graph)
    node_data = graph[current_node][0]
    if current_node not in visited:
        text_format += f'({node_data["abbreviation"]} / {node_data["token"]}'

        if graph[current_node][1] == []:
            text_format += ")"
            visited.add(current_node)
        else:
            indents = tabs * "\t"
            for child_node in graph[current_node][1]:
                text_format += f"\n{indents}:{child_node[0]} "
                text_format = get_text_format(
                    graph, child_node[1], visited, text_format, tabs + 1
                )
            text_format += ")"
            visited.add(current_node)
    else:
        text_format += node_data["abbreviation"]
    return text_format


def amr_grew_to_text(graph):
    """Takes a graph in GREW format and produces its textual representation
    The graph is prepossed so that an abbreviation and a list of incoming
    relations is added to each node. The recursive get_text_format function
    is then called

    Parameters: graph - a graph in GREW format

    Output: the graph in text format
    """

    # For each node add a list of incoming relations by reversing the outgoing ones

    abbreviations = []

    # Create abbreviations for each node
    for node in graph:
        if "token" not in graph[node][0]:
            graph[node][0]["token"] = graph[node][0]["lemma"]
        abbr = graph[node][0]["token"][0].lower()

        # if first letter is not in the abbreviations, the first letter becomes the abbreviation
        if abbr not in abbreviations:
            graph[node][0]["abbreviation"] = abbr
            abbreviations.append(abbr)
        else:
            previous = [x for x in abbreviations if x.startswith(abbr)]
            graph[node][0]["abbreviation"] = abbr + str(len(previous) + 1)
            abbreviations.append(abbr)

    # Make a list in each node where the incoming relations will be kept
    for node in graph:
        graph[node].append([])

    # Fill the incoming relations list
    for node in graph:
        for relation in graph[node][1]:
            graph[relation[1]][2] = [relation[0], node]

    # Find the starting node (root)
    for node in graph:
        if graph[node][2] == []:
            starting_node = node
            break

    # Get the text format of the preprocessed graph with a starting node
    text_format = get_text_format(graph, starting_node, set(), "", 1)
    return text_format


def main():
    grew.init()

    ud_graph = grew.graph(sys.argv[1])

    grs = grew.grs(
        "/home/wessel/Documents/documents/study/1_thesis/project/thesis/grew/main.grs"
    )
    result = grew.run(grs, ud_graph, "main")
    print(json.dumps(result))
    exit()
    for i, res in enumerate(result):
        shutil.copy(
            Path(grew.dot_to_png(res)),
            Path(
                f"/home/wessel/Documents/documents/study/1_thesis/project/thesis/data/grew_output/grew_{i}.png"
            ),
        )
    # print(result)
    for amr_graph in result:
        print(amr_grew_to_text(amr_graph))


if __name__ == "__main__":
    main()
