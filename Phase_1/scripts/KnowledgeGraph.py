
import networkx as nx
import plotly.graph_objects as go



# Knowledge Graph
class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_relation(self, entity1, entity2, relation):
        self.graph.add_edge(entity1, entity2, relation=relation)

    def get_neighbors(self, entity):
        return list(self.graph.neighbors(entity))

    def extract_subgraph(self, entity, depth=2):
        return nx.ego_graph(self.graph, entity, radius=depth)

    def visualize_graph_interactive(self):
        G = self.graph
        pos = nx.spring_layout(G)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                )
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Interactive Knowledge Graph',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        fig.write_html("knowledge_graph_interactive.html")
        print("Interactive knowledge graph saved as 'knowledge_graph_interactive.html'")        
    def add_node(self, node, **attr):
        self.graph.add_node(node, **attr)

