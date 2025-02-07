import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import networkx as nx
from collections import deque, defaultdict
import ast
import heapq

# Initialize Dash app with Bootstrap
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Register the "cose" layout for Cytoscape
cyto.load_extra_layouts()


class Author:
    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name
        self.publications = []
        self.publication_count = 0

    def add_publication(self, title: str):
        self.publications.append(title)
        self.publication_count += 1


class Edge:
    def __init__(self, author1: Author, author2: Author):
        self.author1 = author1
        self.author2 = author2
        self.weight = 0

    def increment_weight(self):
        self.weight += 1


class BST:
    class Node:
        def __init__(self, author_id, publication_count):
            self.author_id = author_id
            self.publication_count = publication_count
            self.left = None
            self.right = None

    def __init__(self):
        self.root = None

    def insert(self, author_id, publication_count):
        self.root = self._insert_recursive(self.root, author_id, publication_count)

    def _insert_recursive(self, node, author_id, publication_count):
        if not node:
            return self.Node(author_id, publication_count)

        if publication_count < node.publication_count:
            node.left = self._insert_recursive(node.left, author_id, publication_count)
        else:
            node.right = self._insert_recursive(node.right, author_id, publication_count)
        return node

    def delete(self, author_id):
        self.root = self._delete_recursive(self.root, author_id)

    def _delete_recursive(self, node, author_id):
        if not node:
            return None

        if author_id < node.author_id:
            node.left = self._delete_recursive(node.left, author_id)
        elif author_id > node.author_id:
            node.right = self._delete_recursive(node.right, author_id)
        else:
            if not node.left:
                return node.right
            elif not node.right:
                return node.left

            min_node = self._find_min(node.right)
            node.author_id = min_node.author_id
            node.publication_count = min_node.publication_count
            node.right = self._delete_recursive(node.right, min_node.author_id)

        return node

    def _find_min(self, node):
        current = node
        while current.left:
            current = current.left
        return current

    def get_tree_data(self):
        nodes = []
        edges = []
        self._traverse_tree(self.root, nodes, edges)
        return nodes, edges

    def _traverse_tree(self, node, nodes, edges, x=0, y=0, level=0):
        if not node:
            return

        node_id = f"bst_{node.author_id}"
        nodes.append({
            'data': {
                'id': node_id,
                'label': f"ID: {node.author_id}\nPubs: {node.publication_count}",
                'level': level
            },
            'position': {'x': x, 'y': y}
        })

        if node.left:
            left_id = f"bst_{node.left.author_id}"
            edges.append({
                'data': {
                    'source': node_id,
                    'target': left_id
                }
            })
            self._traverse_tree(node.left, nodes, edges, x - 100 / (level + 1), y + 100, level + 1)

        if node.right:
            right_id = f"bst_{node.right.author_id}"
            edges.append({
                'data': {
                    'source': node_id,
                    'target': right_id
                }
            })
            self._traverse_tree(node.right, nodes, edges, x + 100 / (level + 1), y + 100, level + 1)


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.average_publications = 0
        self.bst = None

    def add_node(self, author: Author):
        if author.id not in self.nodes:
            self.nodes[author.id] = author
            self.edges[author.id] = {}
            self._update_average_publications()

    def add_edge(self, author1_id: int, author2_id: int):
        if author1_id not in self.nodes or author2_id not in self.nodes:
            return

        if author2_id not in self.edges[author1_id]:
            edge = Edge(self.nodes[author1_id], self.nodes[author2_id])
            self.edges[author1_id][author2_id] = edge
            self.edges[author2_id][author1_id] = edge

    def _update_average_publications(self):
        if not self.nodes:
            self.average_publications = 0
            return
        total_publications = sum(author.publication_count for author in self.nodes.values())
        self.average_publications = total_publications / len(self.nodes)

    def get_shortest_path(self, start_id: int, end_id: int):
        if start_id not in self.nodes or end_id not in self.nodes:
            return [], float('inf'), []

        distances = {node_id: float('inf') for node_id in self.nodes}
        distances[start_id] = 0
        predecessors = {node_id: None for node_id in self.nodes}

        pq = [(0, start_id)]
        visited = set()
        steps = []

        while pq:
            current_distance, current_id = heapq.heappop(pq)

            if current_id == end_id:
                break

            if current_id in visited:
                continue

            visited.add(current_id)
            steps.append({
                'visited': list(visited),
                'distance': distances.copy()
            })

            for neighbor_id, edge in self.edges[current_id].items():
                if neighbor_id in visited:
                    continue

                weight = 1.0 / edge.weight if edge.weight > 0 else float('inf')
                distance = current_distance + weight

                if distance < distances[neighbor_id]:
                    distances[neighbor_id] = distance
                    predecessors[neighbor_id] = current_id
                    heapq.heappush(pq, (distance, neighbor_id))

        path = []
        current_id = end_id
        while current_id is not None:
            path.append(current_id)
            current_id = predecessors[current_id]
        path.reverse()
        return path if path[0] == start_id else [], distances[end_id], steps

    def get_shortest_paths_from_author(self, author_id):
        if author_id not in self.nodes:
            return {}, []

        distances = {node_id: float('inf') for node_id in self.nodes}
        distances[author_id] = 0
        predecessors = {node_id: None for node_id in self.nodes}

        pq = [(0, author_id)]
        visited = set()
        steps = []

        while pq:
            current_distance, current_id = heapq.heappop(pq)

            if current_id in visited:
                continue

            visited.add(current_id)
            steps.append({
                'visited': list(visited),
                'distances': distances.copy()
            })

            for neighbor_id, edge in self.edges[current_id].items():
                if neighbor_id in visited:
                    continue

                weight = 1.0 / edge.weight if edge.weight > 0 else float('inf')
                distance = current_distance + weight

                if distance < distances[neighbor_id]:
                    distances[neighbor_id] = distance
                    predecessors[neighbor_id] = current_id
                    heapq.heappush(pq, (distance, neighbor_id))

        return distances, steps

    def get_collaboration_queue(self, author_id):
        if author_id not in self.nodes:
            return [], []

        collaborator_queue = []
        for collaborator_id in self.edges[author_id]:
            collaborator = self.nodes[collaborator_id]
            heapq.heappush(collaborator_queue,
                           (-collaborator.publication_count, collaborator_id))

        sorted_collaborators = []
        steps = []
        while collaborator_queue:
            _, collaborator_id = heapq.heappop(collaborator_queue)
            sorted_collaborators.append(collaborator_id)
            steps.append(sorted_collaborators.copy())

        return sorted_collaborators, steps

    def get_collaboration_count(self, author_id):
        if author_id not in self.nodes:
            return 0
        return len(self.edges[author_id])

    def get_most_collaborative_author(self):
        max_collaborations = 0
        most_collaborative_id = None
        steps = []

        for author_id in self.nodes:
            collaborations = len(self.edges[author_id])
            steps.append({
                'author_id': author_id,
                'collaborations': collaborations,
                'is_max': collaborations > max_collaborations
            })

            if collaborations > max_collaborations:
                max_collaborations = collaborations
                most_collaborative_id = author_id

        return most_collaborative_id, max_collaborations, steps

    def get_longest_path(self, start_id):
        if start_id not in self.nodes:
            return [], []

        visited = set()
        longest_path = []
        current_path = []
        steps = []

        def dfs(node_id):
            nonlocal longest_path
            visited.add(node_id)
            current_path.append(node_id)
            steps.append({
                'visited': visited.copy(),
                'current_path': current_path.copy()
            })

            for neighbor_id in self.edges[node_id]:
                if neighbor_id not in visited:
                    dfs(neighbor_id)

            if len(current_path) > len(longest_path):
                longest_path = current_path.copy()

            current_path.pop()
            visited.remove(node_id)

        dfs(start_id)
        return longest_path, steps

    def build_collaboration_bst(self, author_ids):
        self.bst = BST()
        for author_id in author_ids:
            if author_id in self.nodes:
                self.bst.insert(author_id, self.nodes[author_id].publication_count)
        return self.bst.get_tree_data()

    def delete_author(self, author_id):
        if author_id not in self.nodes:
            return

        connected_authors = list(self.edges[author_id].keys())
        for connected_id in connected_authors:
            del self.edges[connected_id][author_id]

        del self.edges[author_id]
        del self.nodes[author_id]
        self._update_average_publications()


class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.author_id_map = {}
        self.next_id = 0

    def get_author_id(self, author_name: str):
        if author_name not in self.author_id_map:
            self.author_id_map[author_name] = self.next_id
            self.next_id += 1
        return self.author_id_map[author_name]

    def parse_coauthors(self, coauthors_str: str):
        try:
            if pd.isna(coauthors_str):
                return []
            return ast.literal_eval(coauthors_str)
        except:
            print(f"Error parsing coauthors: {coauthors_str}")
            return []

    def load_data(self):
        df = pd.read_excel(self.file_path)
        graph = Graph()

        for _, row in df.iterrows():
            coauthors = self.parse_coauthors(row['coauthors'])
            paper_title = row['paper_title']

            for author_name in coauthors:
                author_id = self.get_author_id(author_name)
                if author_id not in graph.nodes:
                    author = Author(author_id, author_name)
                    graph.add_node(author)
                graph.nodes[author_id].add_publication(paper_title)

            for i, author1 in enumerate(coauthors):
                for author2 in coauthors[i + 1:]:
                    id1 = self.get_author_id(author1)
                    id2 = self.get_author_id(author2)
                    graph.add_edge(id1, id2)
                    graph.edges[id1][id2].increment_weight()

        return graph


# Define the Cytoscape stylesheet
cytoscape_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'label': 'data(label)',
            'width': 'data(size)',
            'height': 'data(size)',
            'background-color': '#66b2ff',
            'font-size': '8px',
            'text-wrap': 'wrap',
            'text-max-width': '80px'
        }
    },
    {
        'selector': 'edge',
        'style': {
            'width': 'data(weight)',
            'opacity': 0.5,
            'curve-style': 'bezier'
        }
    },
    {
        'selector': '.highlighted',
        'style': {
            'background-color': '#ff4444',
            'line-color': '#ff0000',
            'opacity': 1,
            'z-index': 9999
        }
    },
    {
        'selector': '.visited',
        'style': {
            'background-color': '#44ff44',
            'opacity': 1
        }
    },
    {
        'selector': '.current-path',
        'style': {
            'background-color': '#4444ff',
            'line-color': '#0000ff',
            'opacity': 1
        }
    }
]

# Layout components
header = html.Div([
    html.H1("Academic Collaboration Network", className="text-center my-4"),
    html.P("Visualize and analyze academic collaboration networks", className="text-center text-muted")
])

requirements_sidebar = html.Div([
    dbc.Button("1. Find Shortest Path", id="req1-button", color="primary", className="mb-2 w-100"),
    dbc.Collapse([
        dbc.Input(id="req1-start", placeholder="Start Author ID", type="number", className="mb-2"),
        dbc.Input(id="req1-end", placeholder="End Author ID", type="number", className="mb-2"),
        dbc.Button("Calculate", id="req1-submit", color="success", className="w-100")
    ], id="req1-collapse"),

    dbc.Button("2. Create Collaboration Queue", id="req2-button", color="primary", className="mb-2 w-100"),
    dbc.Collapse([
        dbc.Input(id="req2-author", placeholder="Author ID", type="number", className="mb-2"),
        dbc.Button("Create Queue", id="req2-submit", color="success", className="w-100")
    ], id="req2-collapse"),

    dbc.Button("3. Build BST", id="req3-button", color="primary", className="mb-2 w-100"),
    dbc.Collapse([
        dbc.Input(id="req3-author", placeholder="Author ID", type="number", className="mb-2"),
        dbc.Button("Build Tree", id="req3-submit", color="success", className="w-100")
    ], id="req3-collapse"),

    dbc.Button("4. All Shortest Paths", id="req4-button", color="primary", className="mb-2 w-100"),
    dbc.Collapse([
        dbc.Input(id="req4-author", placeholder="Author ID", type="number", className="mb-2"),
        dbc.Button("Calculate", id="req4-submit", color="success", className="w-100")
    ], id="req4-collapse"),

    dbc.Button("5. Count Collaborators", id="req5-button", color="primary", className="mb-2 w-100"),
    dbc.Collapse([
        dbc.Input(id="req5-author", placeholder="Author ID", type="number", className="mb-2"),
        dbc.Button("Count", id="req5-submit", color="success", className="w-100")
    ], id="req5-collapse"),
    dbc.Button("6. Most Collaborative Author", id="req6-button", color="primary", className="mb-2 w-100"),
    dbc.Collapse([
        dbc.Button("Find", id="req6-submit", color="success", className="w-100")
    ], id="req6-collapse"),

    dbc.Button("7. Find Longest Path", id="req7-button", color="primary", className="mb-2 w-100"),
    dbc.Collapse([
        dbc.Input(id="req7-author", placeholder="Author ID", type="number", className="mb-2"),
        dbc.Button("Calculate", id="req7-submit", color="success", className="w-100")
    ], id="req7-collapse"),

    html.Div(id="requirement-output", className="mt-3")
])

# Requirement 1: Shortest Path
dbc.Button("1. Find Shortest Path", id="req1-button", color="primary", className="mb-2 w-100"),
dbc.Collapse([
    dbc.Input(id="req1-start", placeholder="Start Author ID", type="number", className="mb-2"),
    dbc.Input(id="req1-end", placeholder="End Author ID", type="number", className="mb-2"),
    dbc.Button("Calculate", id="req1-submit", color="success", className="w-100")
], id="req1-collapse"),
# Requirement 2: Count Collaborators
dbc.Button("2. Count Collaborators", id="req2-button", color="primary", className="mb-2 w-100"),
dbc.Collapse([
    dbc.Input(id="req2-author", placeholder="Author ID", type="number", className="mb-2"),
    dbc.Button("Count", id="req2-submit", color="success", className="w-100")
], id="req2-collapse"),

# Requirement 2: Collaboration Queue
dbc.Button("2. Create Collaboration Queue", id="req2-button", color="primary", className="mb-2 w-100"),
dbc.Collapse([
    dbc.Input(id="req2-author", placeholder="Author ID", type="number", className="mb-2"),
    dbc.Button("Create Queue", id="req2-submit", color="success", className="w-100")
], id="req2-collapse"),

# Requirement 3: Build BST
dbc.Button("3. Build BST", id="req3-button", color="primary", className="mb-2 w-100"),
dbc.Collapse([
    dbc.Input(id="req3-author", placeholder="Author ID", type="number", className="mb-2"),
    dbc.Button("Build Tree", id="req3-submit", color="success", className="w-100")
], id="req3-collapse"),

# Requirement 4: All Shortest Paths
dbc.Button("4. All Shortest Paths", id="req4-button", color="primary", className="mb-2 w-100"),
dbc.Collapse([
    dbc.Input(id="req4-author", placeholder="Author ID", type="number", className="mb-2"),
    dbc.Button("Calculate", id="req4-submit", color="success", className="w-100")
], id="req4-collapse"),

# Requirement 5: Count Collaborators
dbc.Button("5. Count Collaborators", id="req5-button", color="primary", className="mb-2 w-100"),
dbc.Collapse([
    dbc.Input(id="req5-author", placeholder="Author ID", type="number", className="mb-2"),
    dbc.Button("Count", id="req5-submit", color="success", className="w-100")
], id="req5-collapse"),

# Requirement 6: Most Collaborative Author
dbc.Button("6. Most Collaborative Author", id="req6-button", color="primary", className="mb-2 w-100"),
dbc.Collapse([
    dbc.Button("Find", id="req6-submit", color="success", className="w-100")
], id="req6-collapse"),

# Requirement 7: Longest Path
dbc.Button("7. Find Longest Path", id="req7-button", color="primary", className="mb-2 w-100"),
dbc.Collapse([
    dbc.Input(id="req7-author", placeholder="Author ID", type="number", className="mb-2"),
    dbc.Button("Calculate", id="req7-submit", color="success", className="w-100")
], id="req7-collapse"),

html.Div(id="requirement-output", className="mt-3")

# Main layout
app.layout = dbc.Container([
    header,
    dbc.Row([
        dbc.Col([requirements_sidebar], width=3),
        dbc.Col([
            cyto.Cytoscape(
                id='network-graph',
                layout={'name': 'cose'},
                style={'width': '100%', 'height': '800px'},
                elements=[],
                stylesheet=cytoscape_stylesheet
            )
        ], width=9)
    ]),
    dbc.Modal([
        dbc.ModalHeader("Author Details"),
        dbc.ModalBody(id="node-click-output"),
        dbc.ModalFooter([
            dbc.Button("Delete Author", id="delete-author-button", color="danger"),
            dbc.Button("Close", id="close-modal-button", className="ml-2")
        ])
    ], id="node-click-modal")
], fluid=True)


# Callbacks
@app.callback(
    [Output(f"req{i}-collapse", "is_open", allow_duplicate=True) for i in range(1, 8)],
    [Input(f"req{i}-button", "n_clicks") for i in range(1, 8)],
    [State(f"req{i}-collapse", "is_open") for i in range(1, 8)],
    prevent_initial_call=True
)
def toggle_requirement_inputs(*args):
    n_clicks = args[:7]
    is_open = args[7:]
    ctx = callback_context

    if not ctx.triggered:
        return [False] * 7

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    req_num = int(button_id.split("-")[0][3])

    return [not is_open[i] if i == req_num - 1 else False for i in range(7)]


@app.callback(
    [Output("requirement-output", "children"),
     Output("network-graph", "elements", allow_duplicate=True)],
    [Input(f"req{i}-submit", "n_clicks") for i in range(1, 8)],
    [State("req1-start", "value"),
     State("req1-end", "value"),
     State("req2-author", "value"),
     State("req3-author", "value"),
     State("req4-author", "value"),
     State("req5-author", "value"),
     State("req7-author", "value")],
    prevent_initial_call=True
)
def handle_requirement_submit(*args):
    n_clicks = args[:7]
    states = args[7:]
    ctx = callback_context

    if not ctx.triggered:
        return "", create_network_elements(graph)

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Requirement 1: Shortest Path
    if button_id == "req1-submit" and states[0] is not None and states[1] is not None:
        path, distance, steps = graph.get_shortest_path(states[0], states[1])
        if not path:
            return "No path found between these authors", create_network_elements(graph)

        path_names = " → ".join(graph.nodes[id].name for id in path)
        return (
            html.Div([
                html.H5("Shortest Path:"),
                html.P(path_names),
                html.P(f"Path length: {distance:.2f}")
            ]),
            create_network_elements(graph, path)
        )

    # Requirement 2: Collaboration Queue
    elif button_id == "req2-submit" and states[2] is not None:
        queue, steps = graph.get_collaboration_queue(states[2])
        if not queue:
            return "Author not found or has no collaborators", create_network_elements(graph)

        queue_steps = []
        for step in steps:
            queue_steps.append(html.Li([
                "Queue: ",
                ", ".join(graph.nodes[id].name for id in step)
            ]))

        return (
            html.Div([
                html.H5("Collaboration Queue:"),
                html.P(f"Author: {graph.nodes[states[2]].name}"),
                html.H6("Queue Formation Steps:"),
                html.Ul(queue_steps)
            ]),
            create_network_elements(graph, highlight_nodes=queue)
        )

    # Requirement 3: Build BST
    elif button_id == "req3-submit" and states[3] is not None:
        queue, _ = graph.get_collaboration_queue(states[3])
        if not queue:
            return "Author not found or has no collaborators", create_network_elements(graph)

        nodes, edges = graph.build_collaboration_bst(queue)

        return (
            html.Div([
                html.H5("Binary Search Tree:"),
                html.P(f"Built from collaborators of {graph.nodes[states[3]].name}")
            ]),
            nodes + edges
        )

    # Requirement 4: All Shortest Paths
    elif button_id == "req4-submit" and states[4] is not None:
        distances, steps = graph.get_shortest_paths_from_author(states[4])
        if not distances:
            return "Author not found in the network", create_network_elements(graph)

        rows = []
        for target_id, distance in distances.items():
            if target_id != states[4] and distance != float('inf'):
                target_name = graph.nodes[target_id].name
                rows.append(html.Tr([
                    html.Td(target_name),
                    html.Td(f"{distance:.2f}")
                ]))

        return (
            html.Div([
                html.H5("Shortest Paths from Author:"),
                html.P(f"Starting from: {graph.nodes[states[4]].name}"),
                html.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Target Author"),
                            html.Th("Distance")
                        ])
                    ]),
                    html.Tbody(rows)
                ], className="table table-striped")
            ]),
            create_network_elements(graph, highlight_nodes=[states[4]])
        )

    # Requirement 5: Count Collaborators
    elif button_id == "req5-submit" and states[5] is not None:
        count = graph.get_collaboration_count(states[5])
        return (
            html.Div([
                html.H5("Collaboration Count:"),
                html.P(f"Author: {graph.nodes[states[5]].name}"),
                html.P(f"Number of collaborators: {count}")
            ]),
            create_network_elements(graph, highlight_nodes=[states[5]])
        )

    # Requirement 6: Most Collaborative Author
    elif button_id == "req6-submit":
        author_id, count, steps = graph.get_most_collaborative_author()
        if author_id is None:
            return "No authors found in the network", create_network_elements(graph)

        return (
            html.Div([
                html.H5("Most Collaborative Author:"),
                html.P(f"Author: {graph.nodes[author_id].name}"),
                html.P(f"Number of collaborators: {count}")
            ]),
            create_network_elements(graph, highlight_nodes=[author_id])
        )

    # Requirement 7: Longest Path
    elif button_id == "req7-submit" and states[6] is not None:
        path, steps = graph.get_longest_path(states[6])
        if not path:
            return "Author not found in the network", create_network_elements(graph)

        path_names = " → ".join(graph.nodes[id].name for id in path)
        return (
            html.Div([
                html.H5("Longest Path:"),
                html.P(f"Starting from: {graph.nodes[states[6]].name}"),
                html.P(f"Path length: {len(path)} nodes"),
                html.P(path_names)
            ]),
            create_network_elements(graph, highlight_path=path)
        )

    return "", create_network_elements(graph)


@app.callback(
    [Output("node-click-modal", "is_open"),
     Output("node-click-output", "children"),
     Output("network-graph", "elements", allow_duplicate=True)],
    [Input("network-graph", "tapNodeData"),
     Input("close-modal-button", "n_clicks"),
     Input("delete-author-button", "n_clicks")],
    [State("node-click-modal", "is_open"),
     State("network-graph", "elements")],
    prevent_initial_call=True
)
def handle_node_click(node_data, close_clicks, delete_clicks, is_open, elements):
    ctx = callback_context
    if not ctx.triggered:
        return False, None, elements

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "close-modal-button":
        return False, None, elements

    if button_id == "delete-author-button" and node_data:
        author_id = int(node_data["id"])
        graph.delete_author(author_id)
        return False, None, create_network_elements(graph)

    if node_data:
        author = graph.nodes[int(node_data["id"])]
        return True, html.Div([
            html.H5(author.name),
            html.P(f"ID: {author.id}"),
            html.P(f"Publications: {author.publication_count}"),
            html.P(f"Collaborators: {len(graph.edges[author.id])}"),
            html.H6("Publications:"),
            html.Ul([html.Li(pub) for pub in author.publications])
        ]), elements

    return False, None, elements


def create_network_elements(graph, highlight_path=None, highlight_nodes=None):
    elements = []
    highlight_nodes = highlight_nodes or []

    # Add nodes
    for author_id, author in graph.nodes.items():
        size = 20 * (1 + (author.publication_count - graph.average_publications)
                     / graph.average_publications)
        node_class = ""
        if highlight_path and author_id in highlight_path:
            node_class = "highlighted"
        elif author_id in highlight_nodes:
            node_class = "highlighted"

        elements.append({
            'data': {
                'id': str(author_id),
                'label': f"{author.name}\n(ID: {author_id})",
                'size': max(20, size)
            },
            'classes': node_class
        })

    # Add edges
    for author1_id in graph.edges:
        for author2_id, edge in graph.edges[author1_id].items():
            if author1_id < author2_id:  # Avoid duplicate edges
                edge_class = ""
                if highlight_path and author1_id in highlight_path and author2_id in highlight_path:
                    if abs(highlight_path.index(author1_id) - highlight_path.index(author2_id)) == 1:
                        edge_class = "highlighted"

                elements.append({
                    'data': {
                        'source': str(author1_id),
                        'target': str(author2_id),
                        'weight': edge.weight
                    },
                    'classes': edge_class
                })

    return elements


# Initialize global graph variable
graph = None

def main():
    try:
        # Initialize the data
        file_path = "C:/Users/PISHTAAZ SOFTWARE/Desktop/python/book.xlsx"
        loader = DataLoader(file_path)
        global graph
        graph = loader.load_data()

        # Print initial statistics
        print("Data loaded successfully")
        print(f"Total authors: {len(graph.nodes)}")
        print(f"Average publications: {graph.average_publications:.2f}")

        # Run the server
        app.run_server(debug=True, port=8050)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()