from graphviz import Digraph

# def create_amr_graph(edges):
#     g = Digraph('G', format='png')
#
#     for edge in edges:
#         g.edge(edge[0], edge[2], label=edge[1])
#
#     g.render('amr_graph', view=True)
#
# edges = [('a', 'ARG0', 'b'), ('b', 'ARG1', 'c'), ('c', 'ARG2', 'd')]
# create_amr_graph(edges)
#
from graphviz import Digraph
#
# # Initialize a directed graph
dot = Digraph('G1')
dot1 = Digraph('G2')
dot2 = Digraph('G3')
dot3 = Digraph('G4')
# # Add nodes with color
dot.node('want', 'want', fontweight='bold', penwidth='2')
dot.node('S1', 'S', fontcolor ='blue')
dot.node('O[S]', 'O[S]', fontcolor='blue')
dot.edge('want', 'S1', label='ARG0')
dot.edge('want', 'O[S]',  label='ARG1')

dot1.node('write', 'write')
dot1.node('person', 'person', fontweight='bold', penwidth='2')
dot1.edge('write', 'person', 'ARG1')

dot2.node('sleep', 'sleep',fontweight='bold', penwidth='2')
dot2.node('S', 'S', fontcolor='blue')
dot2.edge('sleep', 'S', label='ARG0')

dot3.node('m', 'm', fontcolor='blue')
dot3.node('sound', 'sound',fontweight='bold', penwidth='2')
dot3.edge('m', 'sound', label='manner')

dot_combined = Digraph('G_combined')
dot_combined.subgraph(dot)
dot_combined.subgraph(dot1)
dot_combined.subgraph(dot2)
dot_combined.subgraph(dot3)
# dot.node('sleep', 'sleep')
# dot.node('sound', 'sound')
#
# # Add edges with color

# dot.edge('sleep', 'sound', label='manner')
# dot.edge('want', 'sleep', label='ARG1')
# # Save and render the graph to a PDF file
dot_combined.render('combined_graphs.png', view=True)

