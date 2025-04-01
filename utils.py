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
    Convert a NetworkX subgraph into a PyTorch Geometric (PyG) Data object,
    ensuring node IDs are re-labeled from 0..(num_sub_nodes-1) to avoid out-of-range indices.

    Args:
        subgraph (networkx.Graph): The subgraph to convert (already extracted from the original G).
        data (torch_geometric.data.Data): The full graph data containing node features and edge attributes.
        edge_to_idx (dict): Mapping from (u, v) in the original graph to the edge_attr index.

    Returns:
        torch_geometric.data.Data: The subgraph in PyG Data format, with fields:
            - x (Tensor): Node features, shape [num_sub_nodes, feature_dim].
            - edge_index (Tensor): Edge indices, shape [2, num_sub_edges].
            - edge_attr (Tensor): Edge attributes, shape [num_sub_edges, 3].
            - degree (Tensor): Node degrees, shape [num_sub_nodes].
            - node_ids (Tensor): Node IDs in the subgraph, shape [num_sub_nodes].
    
    Process overview:
        1) Re-index subgraph nodes to 0..(num_sub_nodes-1) and store original node IDs.
        2) Build PyG Data:
            - x: use original_label to index into 'data.x'
            - edge_index: directly from the re-labeled subG
            - edge_attr: from edge_to_idx[(orig_u, orig_v)] or [(orig_v, orig_u)]
    """
    import torch
    import networkx as nx
    from torch_geometric.data import Data

    # -- 1. Re-index subgraph nodes to 0..(N-1) and store original node IDs
    subG = nx.convert_node_labels_to_integers(
        subgraph, 
        first_label=0, 
        ordering='default',        # or 'sorted'
        label_attribute='original_label'
    )
    # Now, subG's node labels are 0..(num_sub_nodes-1)
    # subG.nodes[u]['original_label'] stores the original node ID

    # -- 2. Get nodes and edges
    nodes = sorted(list(subG.nodes()))           # [0, 1, 2, ..., N-1]
    edges = list(subG.edges())                   # [(0,1), (1,2), ...]

    num_sub_nodes = len(nodes)
    num_sub_edges = len(edges)

    # -- 3. Construct node features --
    #    3.1 For each new node u, get original_label and index into data.x
    original_ids = []
    for u in nodes:
        orig_label = subG.nodes[u]['original_label']  # Original graph node ID
        original_ids.append(orig_label)
    original_ids_tensor = torch.tensor(original_ids, dtype=torch.long)  # [num_sub_nodes]

    # Extract corresponding features from the full graph's node features
    x = data.x[original_ids_tensor]  # shape [num_sub_nodes, feature_dim]

    # Compute degrees in the subgraph
    degrees = torch.tensor([subG.degree(u) for u in nodes], dtype=torch.long)  # [num_sub_nodes]

    # node_ids are local subgraph node IDs [0..N-1]
    node_ids = torch.arange(num_sub_nodes, dtype=torch.long)  # [num_sub_nodes]

    # -- 4. Construct edge_index and edge_attr --
    edge_index_list = []
    edge_attrs_list = []

    for (u, v) in edges:
        # u, v are subgraph node IDs [0..N-1], get original node IDs
        orig_u = subG.nodes[u]['original_label']
        orig_v = subG.nodes[v]['original_label']

        # Add to edge_index list as subgraph node IDs
        edge_index_list.append([u, v])

        # Lookup edge attributes using original node IDs
        if (orig_u, orig_v) in edge_to_idx:
            idx = edge_to_idx[(orig_u, orig_v)]
        elif (orig_v, orig_u) in edge_to_idx:
            idx = edge_to_idx[(orig_v, orig_u)]
        else:
            idx = -1  # Edge not found in original graph

        if idx != -1:
            edge_attrs_list.append(data.edge_attr[idx].tolist())  # [dist, speed, jamFactor]
        else:
            edge_attrs_list.append([0.0, 0.0, 0.0])               # Default values

    if len(edge_index_list) > 0:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()  # [2, E]
        edge_attr = torch.tensor(edge_attrs_list, dtype=torch.float)                # [E, 3]
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 3), dtype=torch.float)

    # -- 5. Assemble PyG Data object --
    sub_data = Data(
        x         = x,             # [num_sub_nodes, feature_dim]
        edge_index= edge_index,    # [2, num_sub_edges]
        edge_attr = edge_attr,     # [num_sub_edges, 3]
        degree    = degrees,       # [num_sub_nodes]
        node_ids  = node_ids       # [num_sub_nodes], local subgraph node ID
    )

    # -- 6. Final out-of-range check (safety) --
    #    If scatter still throws out-of-range, something is wrong elsewhere.
    if edge_index.numel() > 0:
        max_idx = edge_index.max().item()
        if max_idx >= num_sub_nodes:
            raise ValueError(
                f"convert_nx_to_pyg: Found edge_index with node {max_idx} "
                f">= num_sub_nodes={num_sub_nodes}. Possibly data mismatch."
            )
    
    return sub_data

def split_graph_into_subgraphs_louvain(G, data, num_communities, edge_to_idx):
    """
    Split a NetworkX graph into subgraphs using the Louvain community detection algorithm,
    then convert each community subgraph into PyG Data.

    This version explicitly removes any edges whose target node (or source node)
    is not in the set of community nodes, ensuring we do not keep cross-community edges.

    Args:
        G (networkx.Graph): The original graph.
        data (torch_geometric.data.Data): The PyG Data for the entire graph,
                                          containing 'x', 'edge_index', 'edge_attr', etc.
        num_communities (int): Desired number of communities.
        edge_to_idx (dict): Mapping from (u, v) or (v, u) in the original graph
                            to the index in data.edge_attr.

    Returns:
        list of torch_geometric.data.Data:
            A list of subgraphs in PyG format, one for each community.
    """

    # 1) Perform Louvain community detection
    partition = community_louvain.best_partition(G)
    # partition[node] = comm_id

    # 2) Group nodes by community ID
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = set()
        communities[comm_id].add(node)

    current_num = len(communities)

    # 3) If we have more communities than desired, merge the smallest ones
    if current_num > num_communities:
        sorted_communities = sorted(communities.values(), key=lambda x: len(x))
        while len(sorted_communities) > num_communities:
            c1 = sorted_communities.pop(0)  # Remove the smallest community
            if not sorted_communities:
                break
            c2 = sorted_communities.pop(0)  # Remove the next smallest community
            merged = c1.union(c2)           # Merge them
            sorted_communities.append(merged)
            sorted_communities = sorted(sorted_communities, key=lambda x: len(x))  # Re-sort
        communities = {i: c for i, c in enumerate(sorted_communities)}

    elif current_num < num_communities:
        print(f"Warning: Detected {current_num} communities, fewer than desired {num_communities}.")

    # 4) Build subgraphs, convert to PyG Data
    subgraphs = []
    comm_items = list(communities.items())

    for comm_id, nodes in tqdm(comm_items, desc="Converting subgraphs", unit="community"):
        # 4.1 Construct subG with only these nodes
        subG = G.subgraph(nodes).copy()

        # 4.2 Explicitly remove edges that reference nodes outside this community
        #     (should already be handled by subgraph, but ensure)
        filtered_edges = [(u, v) for u, v in subG.edges() if u in nodes and v in nodes]
        subG.clear_edges()  # Remove all edges from subG
        subG.add_edges_from(filtered_edges)

        # 4.3 Convert to PyG Data
        sub_data = convert_nx_to_pyg(subG, data, edge_to_idx)
        subgraphs.append(sub_data)

    return subgraphs

def load_multiple_graphs(graphs_dir, num_communities=10, save_path=None):
    """
    Load multiple JSON graph files from a directory, split each graph into subgraphs
    using the Louvain algorithm, and aggregate all subgraphs into a single list.

    Args:
        graphs_dir (str): Directory containing JSON graph files.
        num_communities (int, optional): Number of communities to split each graph into. Defaults to 10.
        save_path (str, optional): Path to save the loaded subgraphs. If provided, subgraphs are saved to this path.

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
    # Save subgraphs if a save_path is provided
    if save_path is not None:
        torch.save(all_subgraphs, save_path)  # Use default settings for compatibility
        print(f"All subgraphs saved to {save_path}.")
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
            ratio (int, optional): Number of target subgraphs per context. Defaults to 9.
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

class MemoryEfficientJEPADataLoader:
    """
    Memory-efficient data loader that processes smaller batches based on node count,
    helping to prevent out-of-memory errors when dealing with variable-sized graphs.
    """
    def __init__(self, dataset, batch_size, max_nodes_per_batch=2000, shuffle=True):
        """
        Initialize the memory-efficient data loader.
        
        Args:
            dataset (JEPACommunityDataset): Dataset containing context-target graph pairs.
            batch_size (int): Maximum number of graph pairs in a batch.
            max_nodes_per_batch (int): Maximum total number of nodes allowed in a batch.
            shuffle (bool): Whether to shuffle the dataset.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_nodes_per_batch = max_nodes_per_batch
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if shuffle:
            import random
            random.shuffle(self.indices)
        
    def __iter__(self):
        """
        Iterator that yields batches of context-target graph pairs.
        Limits batch size based on total node count to avoid memory issues.
        
        Yields:
            tuple: (context_batch, target_batch) where each is a Batch object 
                  containing multiple graphs.
        """
        from torch_geometric.data import Batch
        current_batch_contexts = []
        current_batch_targets = []
        current_num_nodes = 0
        
        for idx in self.indices:
            context, target = self.dataset[idx]
            context_nodes = context.x.size(0)
            target_nodes = target.x.size(0)
            
            # If adding this pair would exceed the node limit, yield the current batch
            if current_num_nodes > 0 and current_num_nodes + context_nodes + target_nodes > self.max_nodes_per_batch:
                yield (Batch.from_data_list(current_batch_contexts), 
                       Batch.from_data_list(current_batch_targets))
                # Reset batch
                current_batch_contexts = []
                current_batch_targets = []
                current_num_nodes = 0
            
            # Add to current batch
            current_batch_contexts.append(context)
            current_batch_targets.append(target)
            current_num_nodes += context_nodes + target_nodes
            
            # If we've reached batch_size, yield the batch
            if len(current_batch_contexts) >= self.batch_size:
                yield (Batch.from_data_list(current_batch_contexts), 
                       Batch.from_data_list(current_batch_targets))
                # Reset batch
                current_batch_contexts = []
                current_batch_targets = []
                current_num_nodes = 0
        
        # Don't forget to yield the last batch if it's not empty
        if current_batch_contexts:
            yield (Batch.from_data_list(current_batch_contexts), 
                   Batch.from_data_list(current_batch_targets))
    
    def __len__(self):
        """
        Estimate the number of batches.
        
        Returns:
            int: Estimated number of batches.
        """
        # This is an estimate
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

def count_parameters(model):
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model whose parameters to count.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
