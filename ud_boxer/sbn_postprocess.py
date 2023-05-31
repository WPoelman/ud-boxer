import penman
from pathlib import Path
from penman.graph import Graph
from penman.codec import PENMANCodec
import copy
from ud_boxer.sbn import pmb_generator
from sbn_scope import SBNGraph
from ud_boxer.sbn_spec import SBN_NODE_TYPE


def get_descendants_beyond_children(node, triples):
    children = [(source, role, target) for (source, role, target) in triples if
                source == node]
    children_copy = copy.deepcopy(children)
    triples = [x for x in triples if x not in children_copy]
    for child in children_copy:
        # parent_node = [(source, role, target) for (source, role, target) in triples if
        #         target == node]
        # for parent in parent_node:
        #     parent_parent_node = [(source, role, target) for (source, role, target) in triples if
        #         target == node]
        #     if parent_node==[] and 'member' not in parent[1]:
        #         children.extend(parent_parent_node)
        #         print(parent_parent_node)
        # Find children of each child (grandchildren) and add them to descendants
        grandchildren, triples = get_descendants_beyond_children(child[2], triples)
        children.extend(grandchildren)
    return children, triples


def add_member(penman_file):
    codec = PENMANCodec()
    penman_string = Path(penman_file).read_text()
    triples = codec.decode(penman_string).triples
    triples_reference = [x for x in triples if ':instance' not in x and x[2][0] not in ['b','c'] and x[0][0]!='c']
    triples_box = sorted(set([x[0] for x in triples if 'b' in x[0] and ':instance' not in x]))

    scope_pair = {}
    for i in range(len(triples_box)):
        triple = triples_box[i]
        scope_pair[triple] = []
        children_cluster, rest_triples = get_descendants_beyond_children(triple, triples_reference)

        children_cluster = [x for x in children_cluster if ':member' not in x]
        scope_pair[triple].extend([x[2] for x in children_cluster])
        for child in children_cluster:
            triples.append((triple, 'member', child[2]))
            to_remove = [(source, role, target) for (source, role, target) in triples_reference if target == child[2]]
            for k in to_remove:
                triples_reference.remove(k)

    print(rest_triples)
    print(scope_pair)
    return codec.encode(Graph(triples))


# def convert_from_penman_to_sbn(penman_triples):
#     note_type_match = {'b':SBN_NODE_TYPE.BOX, 's': SBN_NODE_TYPE.SYNSET, 'c':SBN_NODE_TYPE.CONSTANT}
#     graph = SBNGraph()
#     for triple in penman_triples:
#         if triple[1] == ':instance':
#             graph.create_node(note_type_match[triple[0][0]], triple[2])
#
#
#         else:
#             from_node_id = (note_type_match[triple[0][0]],triple[0][1:])
#             to_node_id = (note_type_match[triple[2][0]],triple[2][1:])
#             graph.create_edge(from_node_id, to_node_id, triple[1])
#
#     return graph


def main(starting_path):
    error=0
    with open('postprocessed_sbn_amr.txt', 'w') as post, open('postprocessed_train_list.txt', 'w') as train :
        path_list = Path(starting_path).read_text().split('\n')
        root_path = '/Users/shirleenyoung/Desktop/TODO/MA_Thesis/pmb-4.0.0/data/en/gold/'
        for path in path_list:
            print(path)
            # try:
            filepath = root_path+path+'/en.drs.penman'
            penman_string = add_member(filepath)
            post.write(penman_string)
            post.write('\n\n')
            train.write(filepath)
            train.write('\n')
            # except:
            #     print('well')
            # except RecursionError as e:
            #     print(path)
            #     error += 1
            #     print(error)
            #     print(f'error {filepath}')
            #     print(f'Error type: {type(e).__name__}')



if __name__ == '__main__':
    main('/Users/shirleenyoung/Desktop/TODO/MA_Thesis/ud-boxer/ud_boxer/en_train.txt')
