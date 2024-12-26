# utils.py

import os
import json
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import networkx as nx
import community as community_louvain
from tqdm import tqdm

def load_data(json_path):
    """
    Load a single JSON graph file and convert it into NetworkX and PyTorch Geometric (PyG) Data formats.

    Args:
        json_path (str): Path to the JSON file containing the graph data.

    Returns:
        tuple:
            - G (networkx.Graph): The graph represented using NetworkX.
            - data (torch_geometric.data.Data): The graph data in PyG format, containing node features,
              degrees, edge indices, and edge attributes.
            - edge_to_idx (dict): A mapping from edge tuples (u, v) to their corresponding index in edge_attr.
    
    Notes:
        - The function assumes that each node has 'x' and 'y' coordinates in its properties.
        - Edge attributes include 'dist' (distance), 'speed', and 'jamFactor'.
        - Node features are normalized coordinates along with two constant features [1.0, 1.0].
    """
    # Open and load the JSON file
    with open(json_path, 'r') as f:
        graph_json = json.load(f)
    
    # Extract nodes and edges from the JSON data
    nodes = graph_json['nodes']
    edges = graph_json['edges']

    # Initialize an empty NetworkX graph
    G = nx.Graph()
    
    # Extract 'osmid' (OpenStreetMap ID) for each node to create a mapping from osmid to node index
    node_osmids = [n['properties']['osmid'] for n in nodes]
    osmid_to_index = {osmid: i for i, osmid in enumerate(node_osmids)}
    
    # Add nodes to the NetworkX graph using their indices
    G.add_nodes_from(range(len(nodes)))

    # Initialize lists to store edge tuples and their attributes
    edge_tuples = []
    edge_attrs = []
    
    # Iterate over each edge in the JSON data
    for e in edges:
        # Get the source and target node indices using the osmid mapping
        s = osmid_to_index.get(e['source'])
        t = osmid_to_index.get(e['target'])
        
        # If either source or target node is missing, skip this edge
        if s is None or t is None:
            continue
        
        # Add the edge to the NetworkX graph
        G.add_edge(s, t)
        
        # Extract edge attributes with default values if not present
        dist = e.get('dist', 0.0)
        speed = e.get('speed', 0.0)
        jamFactor = e.get('jamFactor', 0.0)

        # Ensure edge attributes are floats; set to 0.0 if conversion fails
        try:
            dist = float(dist)
        except:
            dist = 0.0
        try:
            speed = float(speed)
        except:
            speed = 0.0
        try:
            jamFactor = float(jamFactor)
        except:
            jamFactor = 0.0

        # Append the edge tuple and its attributes to respective lists
        edge_tuples.append((s, t))
        edge_attrs.append([dist, speed, jamFactor])

    # Create a mapping from edge tuples to their index in edge_attr
    edge_to_idx = {}
    for idx, (s, t) in enumerate(edge_tuples):
        edge_to_idx[(s, t)] = idx
        edge_to_idx[(t, s)] = idx  # Ensure bidirectional mapping

    # Convert edge attributes list to a PyTorch tensor
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    # Initialize lists for node features and degrees
    x_list = []
    degrees = []
    
    # Iterate over each node to extract features and compute degrees
    for i, n in enumerate(nodes):
        props = n['properties']
        x_coord = props.get('x', 0.0)  # X-coordinate
        y_coord = props.get('y', 0.0)  # Y-coordinate

        # Ensure coordinates are floats; set to 0.0 if conversion fails
        try:
            x_coord = float(x_coord)
        except:
            x_coord = 0.0
        try:
            y_coord = float(y_coord)
        except:
            y_coord = 0.0

        # Create a feature vector for the node
        # Here, we use [x, y, 1.0, 1.0] as node features
        # The last two features can be placeholders or represent additional information
        x_list.append([x_coord, y_coord, 1.0, 1.0])
        
        # Compute the degree of the node in the NetworkX graph
        degrees.append(G.degree(i))

    # Convert node features list to a PyTorch tensor
    x = torch.tensor(x_list, dtype=torch.float)
    
    # Normalize the coordinates (columns 0 and 1) using simple standardization
    for col in [0, 1]:
        if x[:, col].std() > 0:
            x[:, col] = (x[:, col] - x[:, col].mean()) / (x[:, col].std() + 1e-8)
        else:
            x[:, col] = x[:, col] - x[:, col].mean()

    # Create a PyG Data object with node features and degrees
    data = Data(
        x=x,  # Node features, shape [N, 4]
        degree=torch.tensor(degrees, dtype=torch.long)  # Node degrees, shape [N]
    )
    
    # If there are edges, add edge_index and edge_attr to the Data object
    if edge_tuples:
        # edge_index is a [2, E] tensor where E is the number of edges
        data.edge_index = torch.tensor(edge_tuples, dtype=torch.long).t().contiguous()
        data.edge_attr = edge_attr  # Edge attributes, shape [E, 3]
    else:
        # If no edges, initialize empty tensors for edge_index and edge_attr
        data.edge_index = torch.empty((2, 0), dtype=torch.long)
        data.edge_attr = torch.empty((0, 3), dtype=torch.float)

    return G, data, edge_to_idx  # Return the NetworkX graph, PyG Data, and edge mapping


def convert_nx_to_pyg(subgraph, data, edge_to_idx):
    """
    Convert a NetworkX subgraph into a PyTorch Geometric (PyG) Data object.

    Args:
        subgraph (networkx.Graph): The subgraph to convert.
        data (torch_geometric.data.Data): The original graph data containing node features and edge attributes.
        edge_to_idx (dict): Mapping from edge tuples (u, v) to their index in edge_attr.

    Returns:
        torch_geometric.data.Data: The subgraph in PyG Data format, including node features,
        edge indices, degrees, node IDs, and edge attributes.
    """
    # List of nodes and edges in the subgraph
    nodes = list(subgraph.nodes())
    edges = list(subgraph.edges())
    
    # Convert node indices to a PyTorch tensor
    nodes_tensor = torch.tensor(nodes, dtype=torch.long)
    
    # Extract node features for the subgraph from the original data
    x = data.x[nodes_tensor]  # [num_sub_nodes, feature_dim]

    # Create a mapping from original node index to subgraph node index
    mapping = {node: i for i, node in enumerate(nodes)}
    
    # Initialize lists to store edge indices and edge attributes for the subgraph
    edge_index_list = []
    edge_attrs_list = []
    
    # Iterate over each edge in the subgraph
    for e in edges:
        u, v = e
        if (u in mapping) and (v in mapping):
            # Map original node indices to subgraph node indices
            edge_index_list.append([mapping[u], mapping[v]])

    # Convert edge indices list to a PyTorch tensor
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()  # [2, num_sub_edges]
    
    # Compute degrees for each node in the subgraph
    degrees = torch.tensor([subgraph.degree(n) for n in nodes], dtype=torch.long)  # [num_sub_nodes]
    
    # Assign unique node IDs within the subgraph
    node_ids = torch.arange(len(nodes), dtype=torch.long)  # [num_sub_nodes]
    
    # Create a PyG Data object for the subgraph
    sub_data = Data(
        x=x,  # Node features, shape [num_sub_nodes, feature_dim]
        edge_index=edge_index,  # Edge indices, shape [2, num_sub_edges]
        degree=degrees,  # Node degrees, shape [num_sub_nodes]
        node_ids=node_ids  # Node IDs, shape [num_sub_nodes]
    )
    
    # Iterate over each edge to assign edge attributes
    for e in edges:
        if e in edge_to_idx:
            idx = edge_to_idx[e]
        elif (e[1], e[0]) in edge_to_idx:
            idx = edge_to_idx[(e[1], e[0])]
        else:
            idx = -1  # Edge not found in the original mapping
        
        if idx != -1:
            # Append the corresponding edge attributes
            edge_attrs_list.append(data.edge_attr[idx].tolist())
        else:
            # If edge attributes are missing, append zeros
            edge_attrs_list.append([0.0, 0.0, 0.0])

    # Convert edge attributes list to a PyTorch tensor
    if edge_attrs_list:
        sub_data.edge_attr = torch.tensor(edge_attrs_list, dtype=torch.float)  # [num_sub_edges, 3]
    else:
        # If no edge attributes, initialize an empty tensor
        sub_data.edge_attr = torch.empty((0, 3), dtype=torch.float)

    return sub_data  # Return the subgraph as a PyG Data object


def split_graph_into_subgraphs_louvain(G, data, num_communities, edge_to_idx):
    """
    Split a NetworkX graph into subgraphs using the Louvain community detection algorithm.

    Args:
        G (networkx.Graph): The original graph to split.
        data (torch_geometric.data.Data): The original graph data containing node features and edge attributes.
        num_communities (int): Desired number of communities (subgraphs) to split into.
        edge_to_idx (dict): Mapping from edge tuples (u, v) to their index in edge_attr.

    Returns:
        list of torch_geometric.data.Data: A list of subgraphs in PyG Data format.
    
    Notes:
        - If the initial number of detected communities exceeds `num_communities`, smaller communities
          are iteratively merged until the desired number is reached.
        - Each subgraph is converted into PyG Data format using `convert_nx_to_pyg`.
    """
    # Perform Louvain community detection to partition the graph
    partition = community_louvain.best_partition(G)
    
    # Organize nodes into communities based on the partition
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, set()).add(node)
    
    current_num = len(communities)  # Current number of detected communities

    # Merge smaller communities if there are more communities than desired
    if current_num > num_communities:
        # Sort communities by size (ascending)
        sorted_communities = sorted(communities.values(), key=lambda x: len(x))
        while len(sorted_communities) > num_communities:
            # Remove the two smallest communities and merge them
            c1 = sorted_communities.pop(0)
            if not sorted_communities:
                break
            c2 = sorted_communities.pop(0)
            merged = c1.union(c2)
            sorted_communities.append(merged)
            # Re-sort the communities after merging
            sorted_communities = sorted(sorted_communities, key=lambda x: len(x))
        # Update communities with the merged communities
        communities = {i: c for i, c in enumerate(sorted_communities)}
    elif current_num < num_communities:
        # If fewer communities are detected than desired, issue a warning
        print(f"Warning: Detected {current_num} communities, which is fewer than the desired {num_communities}.")

    # Initialize a list to store the resulting subgraphs
    subgraphs = []
    
    # Convert each community into a subgraph and then into PyG Data format
    comm_items = list(communities.items())
    for comm_id, nodes in tqdm(comm_items, desc="Converting subgraphs", unit="community"):
        # Extract the subgraph corresponding to the current community
        subG = G.subgraph(nodes).copy()
        # Convert the NetworkX subgraph to PyG Data format
        sub_data = convert_nx_to_pyg(subG, data, edge_to_idx)
        # Append the subgraph to the list
        subgraphs.append(sub_data)

    return subgraphs  # Return the list of subgraphs


def load_multiple_graphs(graphs_dir, num_communities=10):
    """
    Load multiple JSON graph files from a directory, split each graph into subgraphs
    using the Louvain algorithm, and aggregate all subgraphs into a single list.

    Args:
        graphs_dir (str): Directory containing JSON graph files.
        num_communities (int, optional): Number of communities to split each graph into. Defaults to 10.

    Returns:
        list of torch_geometric.data.Data: A list containing all subgraphs from all graphs.
    
    Raises:
        ValueError: If no JSON files are found in the specified directory.
    """
    # List all JSON files in the specified directory
    all_files = [f for f in os.listdir(graphs_dir) if f.endswith('.json')]
    
    # Raise an error if no JSON files are found
    if not all_files:
        raise ValueError(f"No JSON files found in {graphs_dir}.")

    # Initialize a list to store all subgraphs from all graphs
    all_subgraphs = []
    
    # Iterate over each JSON file and process it
    for file in all_files:
        full_path = os.path.join(graphs_dir, file)  # Full path to the JSON file
        G, data, edge_to_idx = load_data(full_path)  # Load the graph data
        # Split the graph into subgraphs based on the desired number of communities
        subgraphs = split_graph_into_subgraphs_louvain(G, data, num_communities, edge_to_idx)
        # Add the resulting subgraphs to the aggregate list
        all_subgraphs.extend(subgraphs)

    # Print the total number of loaded graphs and subgraphs
    print(f"Loaded {len(all_files)} JSON graphs, resulting in {len(all_subgraphs)} subgraphs in total.")
    
    return all_subgraphs  # Return the aggregated list of subgraphs


class JEPACommunityDataset(Dataset):
    """
    Custom Dataset class for JEPA (Joint Embedding Predictive Architecture) training.

    Each item in the dataset consists of a pair of subgraphs:
        - context_subgraph: The context graph used to predict the target.
        - target_subgraph: The target graph to be predicted.
    
    The dataset is constructed such that for each subgraph, a fixed number of different target subgraphs
    are paired with it. This facilitates training the JEPA model in a self-supervised manner.

    Args:
        subgraphs (list of torch_geometric.data.Data): List of subgraphs to include in the dataset.
        ratio (int, optional): Number of target subgraphs to pair with each context subgraph.
                               Defaults to 9.
    """
    def __init__(self, subgraphs, ratio=9):
        """
        Initialize the JEPACommunityDataset.

        Args:
            subgraphs (list of torch_geometric.data.Data): List of subgraphs.
            ratio (int, optional): Number of target subgraphs per context subgraph. Defaults to 9.
        """
        self.subgraphs = subgraphs  # List of all subgraphs
        self.ratio = ratio          # Number of target pairs per context
        self.pairs = []             # List to store index pairs (context, target)
        self._build_pairs()         # Build the list of pairs

    def _build_pairs(self):
        """
        Build pairs of context and target subgraphs.

        For each subgraph, randomly select `ratio` number of different subgraphs as targets.
        If there are fewer than `ratio` subgraphs available, pair with all other subgraphs.
        """
        import random
        n = len(self.subgraphs)  # Total number of subgraphs
        for i in range(n):
            all_others = list(range(n))  # List of all subgraph indices
            all_others.remove(i)          # Exclude the current context subgraph
            if len(all_others) <= self.ratio:
                chosen = all_others  # If not enough subgraphs, select all
            else:
                chosen = random.sample(all_others, self.ratio)  # Randomly sample target indices
            for j in chosen:
                self.pairs.append((i, j))  # Append the (context, target) pair

    def __len__(self):
        """
        Return the total number of (context, target) pairs in the dataset.

        Returns:
            int: Number of pairs.
        """
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Retrieve a (context, target) pair by index.

        Args:
            idx (int): Index of the pair to retrieve.

        Returns:
            tuple:
                - context_subgraph (torch_geometric.data.Data): The context subgraph.
                - target_subgraph (torch_geometric.data.Data): The target subgraph.
        """
        i, j = self.pairs[idx]  # Get the indices of context and target subgraphs
        context_subgraph = self.subgraphs[i].clone()   # Clone to avoid in-place modifications
        target_subgraph = self.subgraphs[j].clone()
        return context_subgraph, target_subgraph  # Return the pair


def count_parameters(model):
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model whose parameters to count.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
