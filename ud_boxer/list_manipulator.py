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
