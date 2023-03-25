def manipulate(in_list: list[str]) -> list[(str, int)]:
    x_new_list = []
    word_set = set()
    for i, word in enumerate(in_list):
        if word in word_set:
            continue
        word_set.add(word)
        if word[-1].isdigit():
            x_new_list.append((word[:-1], i))
        else:
            x_new_list.append((word, i))

    return x_new_list

def overlap(string_1, string_2):
    result = ''

    # finding the common chars from both strings
    for char in string_1:
        if char in string_2 and not char in result:
            result += char
    return result


# split_tokens = token.split()
# if len(split_tokens) == 1 and len(node_info)==1:
#     alignment['node_name'] = node_info[0][0]
#     token_id = [x for x in x_new_list if x[0] == token[:-1]][0]
#     alignment['token_id'] = token_id[1]
#     x_new_list.remove(token_id)
#     alignment_list.append(alignment)
#     print(f'look at this{alignment}')
# elif len(split_tokens) == 1 and len(node_info)>1:
#     alignment['node_name'] = [x[0] for x in node_info]
#     token_id = [x for x in x_new_list if x[0] == token[:-1]][0]
#     alignment['token_id'] = token_id[1]
#     alignment['lexical_id'] = [x for x in alignment['node_name'] if 'c' in x][0]
#     x_new_list.remove(token_id)
#     alignment_list.append(alignment)
# elif len(split_tokens) >1 and len(node_info)==1:
#     if '_' in node_info[0][1]:
#         more_than_one_node = node_info[0][1].split('_')
#         tok_id = []
#         for nd in more_than_one_node:
#             for tk in x_new_list:
#                 if nd[:2] == tk[0][:2]:
#                     tok_id.append(tk[1])
#                     x_new_list.remove(tk)
#         assert len(tok_id) == len(more_than_one_node)
#         alignment['token_id'] = tok_id
#         alignment['node_name'] = node_info[0][0]
#         alignment_list.append(alignment)
#
#     else:
#         for tka in split_tokens:
#             if tka[:2] == node_info[0][1][:2]:
#                 for tk in x_new_list:
#                     if tka[:-2] in tk[0]:
#                         alignment['token_id'] = tk[1]
#                         alignment['node_name'] = node_info[0][0]
#                         x_new_list.remove(tk)
#                         alignment_list.append(alignment)
#
# elif len(split_tokens) >1 and len(node_info)>1:
#     token_id_list =[]
#     for nd_info in node_info:
#         if '"' in nd_info[1]:
#             for tok in x_new_list:
#                 print('aaaaaaaa')
#                 print(tok)
#                 if tok[0] in nd_info[1]:
#
#                     token_id_list.append(tok[1])
#                     x_new_list.remove(tok)
#                     alignment['token_id'] = token_id_list
#                     alignment['node_name'] =nd_info[0]
#                     alignment_list.append(alignment)
#
#         else:
#             print(f"wft{split_tokens, node_info}")
#
#
#     print('wuwuwuwuwu')
#     print(alignment_list)
#     print('nonononono')
#     print(split_tokens)
#     print(node_info)





