import json
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
import networkx as nx
import community as community_louvain
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
import random

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
    print("Loading and processing data...")
    with open(json_path, 'r') as f:
        graph_json = json.load(f)
    nodes = graph_json['nodes']
    edges = graph_json['edges']

    G = nx.Graph()
    node_osmids = [n['properties']['osmid'] for n in nodes]
    osmid_to_index = {osmid: i for i, osmid in enumerate(node_osmids)}

    G.add_nodes_from(range(len(nodes)))

    edge_tuples = []
    edge_attrs = []
    for e in tqdm(edges, desc="Processing Edges"):
        s = osmid_to_index.get(e['source'])
        t = osmid_to_index.get(e['target'])
        if s is None or t is None:
            continue
        G.add_edge(s, t)
        edge_tuples.append((s, t))
        dist = e.get('dist', 0.0)
        speed = e.get('speed', 0.0)
        jamFactor = e.get('jamFactor', 0.0)
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
        edge_attrs.append([dist, speed, jamFactor])

    edge_to_idx = {}
    for idx, (s, t) in enumerate(edge_tuples):
        edge_to_idx[(s, t)] = idx
        edge_to_idx[(t, s)] = idx

    if not edge_attrs:
        raise ValueError("No valid edges found.")

    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    x_list = []
    degrees = []
    labels = [n['properties']['has_charging_station'] for n in nodes]
    positive_count = labels.count(1)
    negative_count = labels.count(0)
    for i, n in tqdm(enumerate(nodes), total=len(nodes), desc="Processing Nodes"):
        props = n['properties']
        x_coord = props.get('x', 0.0)
        y_coord = props.get('y', 0.0)
        station_flag = props.get('has_charging_station', 0)
        try:
            x_coord = float(x_coord)
        except:
            x_coord = 0.0
        try:
            y_coord = float(y_coord)
        except:
            y_coord = 0.0
        try:
            station_flag = int(station_flag)
        except:
            station_flag = 0
        x_list.append([x_coord, y_coord, station_flag, 1.0])
        degrees.append(G.degree(i))

    x = torch.tensor(x_list, dtype=torch.float)
    if x[:, 0].std() > 0:
        x[:, 0] = (x[:, 0] - x[:, 0].mean()) / (x[:, 0].std() + 1e-8)
    else:
        x[:, 0] = x[:, 0] - x[:, 0].mean()
    if x[:, 1].std() > 0:
        x[:, 1] = (x[:, 1] - x[:, 1].mean()) / (x[:, 1].std() + 1e-8)
    else:
        x[:, 1] = x[:, 1] - x[:, 1].mean()

    degree = torch.tensor(degrees, dtype=torch.long)
    data = Data(x=x, degree=degree)
    if not edge_tuples:
        raise ValueError("No edges")
    data.edge_index = torch.tensor(edge_tuples, dtype=torch.long).t().contiguous()
    data.edge_attr = edge_attr
    data.osmid = node_osmids

    return G, data, positive_count, negative_count, edge_to_idx

class GraphPairDataset(Dataset):
    def __init__(self, subgraphs, num_samples=1000):
        self.subgraphs = subgraphs
        self.num_samples = num_samples

    def __len__(self):
        return min(self.num_samples, len(self.subgraphs))

    def __getitem__(self, idx):
        subgraph = self.subgraphs[idx]
        target = subgraph.clone()
        target.node_ids = subgraph.node_ids.clone()
        target.osmid = subgraph.osmid.copy()
        return subgraph, target

class MaskedGraphDataset(Dataset):
    def __init__(self, subgraphs, mask_ratio=0.15, num_samples=1000):
        self.subgraphs = subgraphs
        self.mask_ratio = mask_ratio
        self.num_samples = num_samples

    def __len__(self):
        return min(self.num_samples, len(self.subgraphs))

    def __getitem__(self, idx):
        subgraph = self.subgraphs[idx].clone()
        num_nodes = subgraph.num_nodes
        num_mask = max(1, int(self.mask_ratio * num_nodes))
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask_indices = torch.randperm(num_nodes)[:num_mask]
        mask[mask_indices] = True
        original_coords = subgraph.x[mask, :2].clone()
        subgraph.x[mask, 0] = 0.0
        subgraph.x[mask, 1] = 0.0
        subgraph.original_coords = original_coords
        subgraph.mask = mask
        return subgraph

def convert_nx_to_pyg(subgraph, data, edge_to_idx):
    nodes = list(subgraph.nodes())
    edges = list(subgraph.edges())
    nodes_tensor = torch.tensor(nodes, dtype=torch.long)
    x = data.x[nodes_tensor]
    mapping = {node: i for i, node in enumerate(nodes)}
    edge_index = torch.tensor([[mapping[e[0]], mapping[e[1]]] for e in edges], dtype=torch.long).t().contiguous()
    degrees = torch.tensor([subgraph.degree(n) for n in nodes], dtype=torch.long)
    node_ids = torch.arange(len(nodes), dtype=torch.long)
    sub_data = Data(x=x, edge_index=edge_index, degree=degrees, node_ids=node_ids)
    edge_attrs = []
    for e in edges:
        if e in edge_to_idx:
            idx = edge_to_idx[e]
        elif (e[1], e[0]) in edge_to_idx:
            idx = edge_to_idx[(e[1], e[0])]
        else:
            idx = -1
        if idx != -1:
            edge_attrs.append(data.edge_attr[idx].tolist())
        else:
            edge_attrs.append([0.0, 0.0, 0.0])
    sub_data.edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    sub_data.osmid = [data.osmid[n] for n in nodes]
    return sub_data

def split_graph_into_subgraphs_louvain(G, data, num_communities, edge_to_idx):
    print("Splitting graph into subgraphs with Louvain...")
    partition = community_louvain.best_partition(G)
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, set()).add(node)
    current_num = len(communities)
    if current_num > num_communities:
        sorted_communities = sorted(communities.values(), key=lambda x: len(x))
        while len(sorted_communities) > num_communities:
            smallest = sorted_communities.pop(0)
            if not sorted_communities:
                break
            second_smallest = sorted_communities.pop(0)
            merged = smallest.union(second_smallest)
            sorted_communities.append(merged)
            sorted_communities = sorted(sorted_communities, key=lambda x: len(x))
        communities = {i: comm for i, comm in enumerate(sorted_communities)}
    elif current_num < num_communities:
        print(f"Warning: got {current_num} communities, less than {num_communities}.")

    subgraphs = []
    with tqdm(total=len(communities), desc="Splitting Progress", unit="subgraph") as pbar:
        for comm_id, nodes in communities.items():
            subG = G.subgraph(nodes).copy()
            sub_data = convert_nx_to_pyg(subG, data, edge_to_idx)
            subgraphs.append(sub_data)
            pbar.update(1)

    print(f"Number of subgraphs: {len(subgraphs)}")
    return subgraphs

def evaluate_and_save(model, dataloader, loss_fn, save_path, device, prob=0.5):
    model.eval()
    total_loss = 0.0
    results = []
    with torch.no_grad():
        for context_batch, target_batch in tqdm(dataloader, desc="Evaluating"):
            context_batch = context_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            predicted_scores, target_scores, spatial_penalty = model(context_batch, target_batch, pretrain=False)
            focal_loss = loss_fn(predicted_scores, target_scores.float())
            loss = focal_loss + spatial_penalty
            total_loss += loss.item()

            pred_logits = predicted_scores.cpu().numpy()
            true_labels = target_scores.cpu().numpy()

            pred_probs = 1 / (1 + np.exp(-pred_logits))
            pred_labels = (pred_probs >= prob).astype(int)

            osmid_batch = target_batch.osmid
            B, N = predicted_scores.shape
            osmid_flat = [osmid for sublist in osmid_batch for osmid in sublist]

            true_labels_flat = true_labels.flatten()
            pred_labels_flat = pred_labels.flatten()
            for osmid, true, pred in zip(osmid_flat, true_labels_flat, pred_labels_flat):
                results.append({
                    'osmid': osmid,
                    'has_charging_station_true': int(true),
                    'has_charging_station_pred': int(pred)
                })
    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation Loss: {avg_loss:.4f}")
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")

def evaluate_model(model, dataloader, loss_fn, device, prob=0.5):
    model.eval()
    total_loss = 0.0
    all_true = []
    all_pred = []
    with torch.no_grad():
        for context_batch, target_batch in tqdm(dataloader, desc="Evaluating"):
            context_batch = context_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            predicted_scores, target_scores, spatial_penalty = model(context_batch, target_batch, pretrain=False)
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

def search_best_threshold(model, dataloader, loss_fn, device):
    thresholds = np.linspace(0.1, 0.9, 17)
    best_f1 = -1
    best_t = 0.5
    for t in thresholds:
        _, metrics = evaluate_model(model, dataloader, loss_fn, device=device, prob=t)
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_t = t
    print(f"Best threshold: {best_t}, F1: {best_f1}")
    return best_t

class CombinedJEPAData(Dataset):
    def __init__(self, subgraphs, mask_ratio=0.15, num_samples=None):
        self.subgraphs = subgraphs
        self.mask_ratio = mask_ratio
        self.num_samples = len(subgraphs) if num_samples is None else min(num_samples, len(subgraphs))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        original_subgraph = self.subgraphs[idx].clone()
        subgraph = self.subgraphs[idx].clone()
        num_nodes = subgraph.num_nodes
        num_mask = max(1, int(self.mask_ratio * num_nodes))
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask_indices = torch.randperm(num_nodes)[:num_mask]
        mask[mask_indices] = True
        original_coords = subgraph.x[mask, :2].clone()
        subgraph.x[mask, 0] = 0.0
        subgraph.x[mask, 1] = 0.0
        subgraph.original_coords = original_coords
        subgraph.mask = mask

        return subgraph, original_subgraph
