from collections import Counter
with open('none_lines.txt', 'r') as error:
    lines = [line.split()[0] for line in error.readlines()]
    print(Counter(lines).most_common())

