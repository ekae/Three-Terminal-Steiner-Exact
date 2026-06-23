import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from geometry_utils import calc_distance


def finalize_graph_creation(G, graph_side_length=1.0, scale_pos=False):
    """
    Standardize the graph G by assigning node types, coordinates, and edge lengths.
    Follows TSPLIB-like node labeling and coordinate mapping.

    :param G: NetworkX Graph object.
    :param end_node_mode: Mode for selecting terminal nodes ("4corners" or "convexhull").
    :param graph_side_length: Scaling factor for coordinates.
    :param scale_pos: Whether to apply the side length scaling to the 'pos' attribute.
    :return: Standardized NetworkX Graph G.
    """
    # Relabel nodes to strings for consistency
    mapping = {node: str(node) for node in G.nodes() if not isinstance(node, str)}
    G = nx.relabel_nodes(G, mapping)
    
    # Assign default types
    for node in G.nodes():
        if 'type' not in G.nodes[node]:
            G.nodes[node]['type'] = 'unknown_node'
    
    # Scale coordinates if requested
    if scale_pos:
        for node in G.nodes():
            pos = G.nodes[node]['pos']
            G.nodes[node]['pos'] = (pos[0] * graph_side_length, pos[1] * graph_side_length)
    
    # Compute edge lengths (EUC_2D convention)
    G = compute_dist_cartesian(G)
    
    return G

def compute_dist_cartesian(G):
    """
    Compute Euclidean distances (EUC_2D) for all edges in G based on 'pos'.
    
    :param G: NetworkX Graph object.
    :return: G with 'length' attribute on edges.
    """
    for u, v in G.edges():
        pos_u = G.nodes[u]['pos']
        pos_v = G.nodes[v]['pos']
        # Euclidean distance rounded to 5 decimals as per legacy behavior
        dist = round(math.dist(pos_u, pos_v), 5)
        G.edges[u, v]['length'] = dist
    return G

def get_pos(G):
    """
    Get the positions of the nodes in the graph for visualization or further processing.

    :return: Dictionary mapping node labels to (x, y) coordinates.
    """
    return nx.get_node_attributes(G, 'pos')

def dist(graph, node_a, node_b):
    """
    Calculate the Euclidean distance between two nodes in the graph.
            
    :param graph: NetworkX Graph object.
    :param node_a: Label of the first node.
    :param node_b: Label of the second node.
    :return: Euclidean distance rounded to 5 decimal places.
    """
    dx = np.abs(graph.nodes[node_a]['pos'][0] - graph.nodes[node_b]['pos'][0])
    dy = np.abs(graph.nodes[node_a]['pos'][1] - graph.nodes[node_b]['pos'][1])
    dist = np.round(np.sqrt(np.square(dx) + np.square(dy)), 5)
    return dist

def draw_graph(G, title="", disc_size=1.0, chosen_edges=None, extra_points=None, extra_discs=None, extra_discs_layers=1):
    """
    A professional visualization tool for the network graph G.
    Displays terminal nodes with extra nodes and communication radii.

    :param G: NetworkX Graph.
    :param title: Plot title.
    :param disc_size: Radius R for communication coverage discs.
    :param chosen_edges: List of edges (u, v) to highlight.
    :param extra_points: List or dictionary of coordinates for extra nodes.
    :param extra_discs: List of coordinates for extra coverage discs.
    :param extra_discs_layers: Number of concentric layers for the discs.
    """
    pos = get_pos(G)
    
    fig, ax = plt.subplots()
    
                               
    # Draw  Nodes
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_size=300, 
                            node_color='white', edgecolors='black', label="Ground Node")

    # Draw Standard Edges
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
    
    
    ############ TEST PURPOSE ###########
    node_labels = {}
    node_pos = nx.get_node_attributes(G, 'pos')
    label_pos = node_pos
    for node, nodedata in G.nodes.items():
        node_labels[node] = node
                
    nx.draw_networkx_labels(G=G, pos=label_pos, labels=node_labels, font_size=12, 
                                font_weight="bold", font_color="k", font_family='serif')
    
    
    # Highlight Edges
    if chosen_edges:
        nx.draw_networkx_edges(G, pos, edgelist=chosen_edges, edge_color="#2CBAFA", width=2.5)

    # Draw Extra Points
    if extra_points:
        if isinstance(extra_points, dict):
            # Handle Dictionary: { 'name': [x, y] }
            for name, coords in extra_points.items():
                ax.scatter(*coords, s=150, edgecolor='b', facecolor='none', marker='^')
                plt.text(coords[0], coords[1], name, fontsize=12, color='red')

        elif isinstance(extra_points, list):
            # Handle List: [[x1, y1], [x2, y2]]
            for point in extra_points:
                ax.scatter(*point, s=50, edgecolor='b', facecolor='b', marker='s')

    ax.set_title(title)
    ax.set_aspect('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()

def draw_point(G, title="", disc_size=1.0, extra_points=None, extra_discs=None, extra_discs_layers=1, extra_vh_line=[], show_node_label = True, show_edges=False, show_edge_label=False):
    """
    Draws the input graph.

    Parameters:
        graph (NetworkX graph): Input graph.
        title (str): Title of the plot.
        edge_label (bool): Whether to display edge labels.

    Returns:
        None
    """
    
    pos = get_pos(G)
    node_labels = {}
    for node, nodedata in G.nodes.items():
        node_labels[node] = node      
    
    length = nx.get_edge_attributes(G, 'length')
    fig, ax = plt.subplots()

    # Draw  Nodes
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_size=300, 
                            node_color='white', edgecolors='black', label="Ground Node")
    # Draw Standard Edges
    if show_edges:
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
    
    # Write Node Labels
    if show_node_label:
            nx.draw_networkx_labels(G=G, pos=pos, labels=node_labels, font_size=12, 
                                    font_weight="bold", font_color="k", font_family='serif')
    
    # Write Edge Labels
    if show_edge_label:
        # prevent displaying 0.0 length edges
        filtered_length = {k: v for k, v in length.items() if float(v) > 0.0}
        nx.draw_networkx_edge_labels(G=G, pos=pos, edge_labels=filtered_length)
        
    # Draw Extra Points
    if extra_points:
        if isinstance(extra_points, dict):
            # Handle Dictionary: { 'name': [x, y] }
            for name, coords in extra_points.items():
                ax.scatter(*coords, s=150, edgecolor='b', facecolor='none', marker='.')
                plt.text(coords[0], coords[1], name, fontsize=12, color='red')

        elif isinstance(extra_points, list):
            # Handle List: [[x1, y1], [x2, y2]]
            for point in extra_points:
                ax.scatter(*point, s=50, edgecolor='b', facecolor='b', marker='s')
    
    # Draw coverage Discs
    if extra_discs:
        for disc_center in extra_discs:
            for i in range(extra_discs_layers):
                circle = Circle(disc_center, radius=disc_size*(i+1), edgecolor='blue', facecolor='none', linestyle='--', alpha=0.3)
                ax.add_patch(circle)
    
    # Draw_Lines
    if extra_vh_line:
        for x in extra_vh_line[0]:
            plt.axvline(x, color='b', linestyle='--')
        for y in extra_vh_line[1]:
            plt.axhline(y, color='r', linestyle='--')
            
    ax.set_title(title, fontsize=14, fontweight='bold')   
    ax.set_aspect(1)
    ax.axis('equal') 
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.axis('on')
    fig.tight_layout()
    plt.show()

def import_tsplib_file(filepath):
    """
    Import node coordinates from a TSPLIB .tsp file.
    
    :param filepath: Path to the .tsp file.
    :return: List of (x, y) tuples.
    """
    coords = []
    with open(filepath, 'r') as f:
        in_node_section = False
        for line in f:
            line = line.strip()
            if line.startswith("NODE_COORD_SECTION"):
                in_node_section = True
                continue
            if line.startswith("EOF") or line.startswith("-1"):
                break
            if in_node_section:
                parts = line.split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
    return coords


def write_tsplib_graph(G, name="network", filepath="output.tsp"):
    """
    Export the graph coordinates to a TSPLIB compatible file.
            
    :param G: NetworkX Graph object.
    :param name: Name of the problem instance.
    :param filepath: Output path for the .tsp file.
    """
    with open(filepath, 'w') as f:
        f.write(f"NAME : {name}\n")
        f.write("TYPE : TSP\n")
        f.write(f"DIMENSION : {G.number_of_nodes()}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, node in enumerate(G.nodes(), 1):
            x, y = G.nodes[node]['pos']
            f.write(f"{i} {x:.5e} {y:.5e}\n")
        f.write("EOF\n")

def setup_three_point_graph(A, B, C):
    """
    Create a graph connecting three points (A, B, C) with edges.

    This function initializes a graph with three nodes representing the points A, B, and C.
    It connects the nodes with the two shorter edges of the three possible edges
    connecting the points A, B, and C.

    :param A: The first point with attributes x and y.
    :param B: The second point with attributes x and y.
    :param C: The third point with attributes x and y.
    :return: A NetworkX graph with three nodes and two edges.
    """
    G = nx.Graph()
    G.add_node('Pa', pos=(A[0], A[1]))
    G.add_node('Pb', pos=(B[0], B[1]))
    G.add_node('Pc', pos=(C[0], C[1]))
    
    # Calculate all pairwise distances
    d_ab = calc_distance(A, B)
    d_bc = calc_distance(B, C)
    d_ac = calc_distance(A, C)
    
    # Define candidates for edges
    candidates = [
        ('Pa', 'Pb', d_ab),
        ('Pb', 'Pc', d_bc),
        ('Pa', 'Pc', d_ac)
    ]
    
    # Sort candidates by distance (length) in ascending order
    candidates.sort(key=lambda x: x[2])
    
    # Add only the two shorter edges
    G.add_edge(candidates[0][0], candidates[0][1], length=candidates[0][2])
    G.add_edge(candidates[1][0], candidates[1][1], length=candidates[1][2])
    
    G = finalize_graph_creation(G)
    return G

def setup_three_star_graph(pa, pb, pc, pj):
    """
    pcreate a graph connecting three points (pa, pb, pc) to a central junction point (pj).

    This function initializes a graph with four nodes representing the points pa, pb, pc, and pj.
    It connects the nodes pa, pb, and pc to the central junction node pj, forming a star topology.

    :param pa: The first point with attributes x and y.
    :param pb: The second point with attributes x and y.
    :param pc: The third point with attributes x and y.
    :param pj: The junction point with attributes x and y.
    :return: pa NetworkX graph with four nodes and three edges.
    """
    G = nx.Graph()
    G.add_node('Pb', pos=(pa[0], pa[1]))
    G.add_node('Pa', pos=(pb[0], pb[1]))
    G.add_node('Pc', pos=(pc[0], pc[1]))
    G.add_node('pj', pos=(pj[0], pj[1]))
    G.add_edge('Pa', 'pj')
    G.add_edge('Pb', 'pj')
    G.add_edge('Pc', 'pj')
    G = finalize_graph_creation(G)
    return G