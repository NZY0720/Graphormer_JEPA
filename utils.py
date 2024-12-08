# utils.py

import json
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch.utils.data import Dataset
import networkx as nx
import community as community_louvain
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F  # Ensure this import exists


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        focal_loss = focal_weight * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_data(json_path):
    """
    Loads and processes graph data from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing graph data.

    Returns:
        tuple: 
            - G (networkx.Graph): The constructed NetworkX graph.
            - data (torch_geometric.data.Data): PyG Data object containing node and edge features.
            - positive_count (int): Number of positive samples (has_charging_station=1).
            - negative_count (int): Number of negative samples (has_charging_station=0).
    """
    print("Loading and processing data...")
    with open(json_path, 'r') as f:
        graph_json = json.load(f)
    nodes = graph_json['nodes']
    edges = graph_json['edges']

    G = nx.Graph()
    node_osmids = [n['properties']['osmid'] for n in nodes]
    osmid_to_index = {osmid: i for i, osmid in enumerate(node_osmids)}

    for i, n in enumerate(nodes):
        G.add_node(i)

    for e in tqdm(edges, desc="Processing Edges"):
        s = osmid_to_index[e['source']]
        t = osmid_to_index[e['target']]
        G.add_edge(s, t)

    x_list = []
    degrees = []
    positive_count = 0
    negative_count = 0
    for i in tqdm(range(len(nodes)), desc="Processing Nodes"):
        n = nodes[i]
        props = n['properties']
        x_coord = props['x']
        y_coord = props['y']
        station_flag = props['has_charging_station']
        x_list.append([x_coord, y_coord, station_flag, 1.0])
        degrees.append(G.degree(i))
        if station_flag == 1:
            positive_count += 1
        else:
            negative_count += 1

    x = torch.tensor(x_list, dtype=torch.float)
    # Normalize features (optional)
    x[:, 0] = (x[:, 0] - x[:, 0].mean()) / (x[:, 0].std() + 1e-8)
    x[:, 1] = (x[:, 1] - x[:, 1].mean()) / (x[:, 1].std() + 1e-8)
    # Keep the third feature (label) and the fourth feature (fixed at 1.0)

    degree = torch.tensor(degrees, dtype=torch.long)
    data = Data(x=x, degree=degree)
    data.edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

    # Calculate edge features (node distances)
    pos = x[:, :2]
    row, col = data.edge_index
    distances = torch.sqrt(torch.sum((pos[row] - pos[col]) ** 2, dim=1))
    data.edge_attr = distances.unsqueeze(1)

    return G, data, positive_count, negative_count


class GraphPairDataset(Dataset):
    """
    Custom Dataset for graph pairs, consisting of context and target subgraphs.

    Args:
        subgraphs (list): List of PyG Data objects representing subgraphs.
        num_samples (int, optional): Maximum number of samples. Defaults to 1000.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns a tuple of (context_subgraph, target_subgraph).
    """
    def __init__(self, subgraphs, num_samples=1000):
        self.subgraphs = subgraphs
        self.num_samples = num_samples

    def __len__(self):
        return min(self.num_samples, len(self.subgraphs))

    def __getitem__(self, idx):
        subgraph = self.subgraphs[idx]
        target = subgraph.clone()
        target.node_ids = subgraph.node_ids.clone()
        return subgraph, target


def convert_nx_to_pyg(subgraph, x_full):
    """
    Converts a NetworkX subgraph to a PyTorch Geometric Data object.

    Args:
        subgraph (networkx.Graph): The subgraph to convert.
        x_full (torch.Tensor): Full node feature tensor.

    Returns:
        torch_geometric.data.Data: PyG Data object representing the subgraph.
    """
    nodes = list(subgraph.nodes())
    edges = list(subgraph.edges())
    x = x_full[nodes]
    mapping = {node: i for i, node in enumerate(nodes)}
    edge_index = torch.tensor([[mapping[e[0]], mapping[e[1]]] for e in edges], dtype=torch.long).t().contiguous()
    degrees = torch.tensor([subgraph.degree(n) for n in nodes], dtype=torch.long)
    node_ids = torch.arange(len(nodes), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, degree=degrees, node_ids=node_ids)

    pos = x[:, :2]
    row, col = edge_index
    distances = torch.sqrt(torch.sum((pos[row] - pos[col]) ** 2, dim=1))
    data.edge_attr = distances.unsqueeze(1)

    return data


def split_graph_into_subgraphs_louvain(G, data, num_communities):
    """
    Splits a large graph into multiple subgraphs using the Louvain community detection method.

    Args:
        G (networkx.Graph): The original large graph.
        data (torch_geometric.data.Data): PyG Data object containing node features.
        num_communities (int): Desired number of communities/subgraphs.

    Returns:
        list: List of PyG Data objects representing the subgraphs.
    """
    print("Splitting the large graph into multiple subgraphs using the Louvain method...")
    partition = community_louvain.best_partition(G)
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, set()).add(node)

    if len(communities) > num_communities:
        sorted_communities = sorted(communities.values(), key=lambda x: len(x))
        while len(sorted_communities) > num_communities:
            smallest = sorted_communities.pop(0)
            second_smallest = sorted_communities.pop(0)
            merged = smallest.union(second_smallest)
            sorted_communities.append(merged)
            sorted_communities = sorted(sorted_communities, key=lambda x: len(x))
        communities = {i: comm for i, comm in enumerate(sorted_communities)}
    elif len(communities) < num_communities:
        print(f"Warning: Detected number of communities ({len(communities)}) is less than the target number ({num_communities}).")

    subgraphs = []
    with tqdm(total=len(communities), desc="Splitting Progress", unit="subgraph") as pbar:
        for comm_id, nodes in communities.items():
            subG = G.subgraph(nodes).copy()
            sub_data = convert_nx_to_pyg(subG, data.x)
            subgraphs.append(sub_data)
            pbar.update(1)

    print(f"Number of subgraphs after splitting: {len(subgraphs)}")
    return subgraphs


def evaluate_and_save(model, dataloader, loss_fn, save_path, device, prob=0.5):
    """
    Evaluates the model and saves the results to a CSV file.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch_geometric.loader.DataLoader): DataLoader for evaluation data.
        loss_fn (torch.nn.Module): Loss function used for evaluation.
        save_path (str): Path to save the evaluation results CSV.
        device (torch.device): Device to perform computations on.
        prob (float, optional): Probability threshold for classification. Defaults to 0.5.
    """
    model.eval()
    total_loss = 0.0
    results = []
    node_counter = 0

    with torch.no_grad():
        for context_batch, target_batch in tqdm(dataloader, desc="Evaluating"):
            context_batch = context_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            predicted_scores, target_scores, spatial_penalty = model(context_batch, target_batch)
            focal_loss = loss_fn(predicted_scores, target_scores.float())
            loss = focal_loss + spatial_penalty
            total_loss += loss.item()

            pred_logits = predicted_scores.cpu().numpy()
            true_labels = target_scores.cpu().numpy()

            pred_probs = 1 / (1 + np.exp(-pred_logits))
            pred_labels = (pred_probs >= prob).astype(int)

            B, N = predicted_scores.shape
            for b in range(B):
                for n in range(N):
                    results.append({
                        'node_id': node_counter,
                        'has_charging_station_true': int(true_labels[b, n]),
                        'has_charging_station_pred': int(pred_labels[b, n])
                    })
                    node_counter += 1

    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation Loss: {avg_loss:.4f}")
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"Evaluation results saved to {save_path}")


def evaluate_model(model, dataloader, loss_fn, device, prob=0.5):
    """
    Evaluates the model on a given dataloader and computes evaluation metrics.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch_geometric.loader.DataLoader): DataLoader for evaluation data.
        loss_fn (torch.nn.Module): Loss function used for evaluation.
        device (torch.device): Device to perform computations on.
        prob (float, optional): Probability threshold for classification. Defaults to 0.5.

    Returns:
        tuple: 
            - avg_loss (float): Average loss over the dataset.
            - metrics (dict): Dictionary containing accuracy, precision, recall, and F1 score.
    """
    model.eval()
    total_loss = 0.0
    all_true = []
    all_pred = []
    with torch.no_grad():
        for context_batch, target_batch in tqdm(dataloader, desc="Evaluating"):
            context_batch = context_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            predicted_scores, target_scores, spatial_penalty = model(context_batch, target_batch)
            focal_loss = loss_fn(predicted_scores, target_scores.float())
            loss = focal_loss + spatial_penalty
            total_loss += loss.item()

            pred_probs = torch.sigmoid(predicted_scores)
            pred_labels = (pred_probs >= prob).int()
            all_true.append(target_scores.cpu())
            all_pred.append(pred_labels.cpu())

    avg_loss = total_loss / len(dataloader)
    all_true = torch.cat(all_true, dim=0).numpy().flatten()
    all_pred = torch.cat(all_pred, dim=0).numpy().flatten()

    accuracy = accuracy_score(all_true, all_pred)
    precision = precision_score(all_true, all_pred, zero_division=0)
    recall = recall_score(all_true, all_pred, zero_division=0)
    f1 = f1_score(all_true, all_pred, zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return avg_loss, metrics


def custom_collate(batch):
    """
    Custom collate function for DataLoader to handle graph pairs.

    Args:
        batch (list): List of tuples containing context and target subgraphs.

    Returns:
        tuple: Batch of context subgraphs and batch of target subgraphs.
    """
    context_list, target_list = zip(*batch)
    context_batch = Batch.from_data_list(context_list)
    target_batch = Batch.from_data_list(target_list)
    return context_batch, target_batch


def search_best_threshold(model, dataloader, loss_fn, device):
    """
    Searches for the best threshold on the validation set to maximize the F1 score.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch_geometric.loader.DataLoader): DataLoader for validation data.
        loss_fn (torch.nn.Module): Loss function used for evaluation.
        device (torch.device): Device to perform computations on.

    Returns:
        float: The best threshold value found.
    """
    thresholds = np.linspace(0.1, 0.9, 17)  # 0.1, 0.15, ..., 0.9
    best_f1 = -1
    best_t = 0.5
    for t in thresholds:
        _, metrics = evaluate_model(model, dataloader, loss_fn, device=device, prob=t)
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_t = t
    print(f"Best threshold: {best_t}, F1: {best_f1}")
    return best_t
