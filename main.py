# main.py

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, random_split
from tqdm import tqdm
import networkx as nx
import community as community_louvain
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from graph_model import GraphormerJEPA


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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 参数设置
input_dim = 4
hidden_dim = 64
output_dim = 1
lr = 1e-5  # 降低学习率
epochs = 100  # 减少epoch数以快速测试
batch_size = 1
max_degree = 128
max_nodes = 50000
num_heads = 4
num_layers = 4
num_communities = 10
prob = 0.5
alpha = 0.001  # 降低空间损失的权重


def load_data(json_path):
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
    # 对特征进行标准化（可选）
    x[:, 0] = (x[:, 0] - x[:, 0].mean()) / (x[:, 0].std() + 1e-8)
    x[:, 1] = (x[:, 1] - x[:, 1].mean()) / (x[:, 1].std() + 1e-8)
    # 保持第三个特征（标签）和第四个特征（固定为1.0）

    degree = torch.tensor(degrees, dtype=torch.long)
    data = Data(x=x, degree=degree)
    data.edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

    # 计算边特征（节点间距）
    pos = x[:, :2]
    row, col = data.edge_index
    distances = torch.sqrt(torch.sum((pos[row] - pos[col]) ** 2, dim=1))
    data.edge_attr = distances.unsqueeze(1)

    return G, data, positive_count, negative_count


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
        return subgraph, target


def convert_nx_to_pyg(subgraph, x_full):
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
    print("正在使用 Louvain 方法拆分大图为多个子图...")
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
        print(f"警告: 检测到的社区数量 ({len(communities)}) 少于目标数量 ({num_communities})。")

    subgraphs = []
    with tqdm(total=len(communities), desc="拆分进度", unit="子图") as pbar:
        for comm_id, nodes in communities.items():
            subG = G.subgraph(nodes).copy()
            sub_data = convert_nx_to_pyg(subG, data.x)
            subgraphs.append(sub_data)
            pbar.update(1)

    print(f"拆分后的子图数量: {len(subgraphs)}")
    return subgraphs


def evaluate_and_save(model, dataloader, loss_fn, save_path, prob=0.5):
    model.eval()
    total_loss = 0.0
    results = []
    node_counter = 0

    with torch.no_grad():
        for context_batch, target_batch in tqdm(dataloader, desc="评估"):
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
    print(f"评估损失: {avg_loss:.4f}")
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"评估结果已保存至 {save_path}")


def evaluate_model(model, dataloader, loss_fn, prob=0.5):
    model.eval()
    total_loss = 0.0
    all_true = []
    all_pred = []
    with torch.no_grad():
        for context_batch, target_batch in tqdm(dataloader, desc="评估"):
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
    context_list, target_list = zip(*batch)
    context_batch = Batch.from_data_list(context_list)
    target_batch = Batch.from_data_list(target_list)
    return context_batch, target_batch


def search_best_threshold(model, dataloader, loss_fn):
    """在验证集或测试集上搜索最佳阈值，提高F1等指标。"""
    thresholds = np.linspace(0.1, 0.9, 17)  # 更细的阈值扫描
    best_f1 = -1
    best_t = 0.5
    for t in thresholds:
        _, metrics = evaluate_model(model, dataloader, loss_fn, prob=t)
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_t = t
    print(f"Best threshold: {best_t}, F1: {best_f1}")
    return best_t


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 数据加载
    G, data, positive_count, negative_count = load_data("updated_san_francisco_graph.json")

    # 计算正类权重
    if positive_count > 0:
        pos_weight_value = (negative_count / positive_count) * 0.1
    else:
        pos_weight_value = 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float).to(device)
    print(f"Positive Count: {positive_count}, Negative Count: {negative_count}")
    print(f"pos_weight: {pos_weight_value}")

    # 拆分子图
    subgraphs_nx = split_graph_into_subgraphs_louvain(G, data, num_communities=num_communities)
    subgraphs = subgraphs_nx
    dataset = GraphPairDataset(subgraphs, num_samples=len(subgraphs))

    # 划分数据集
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

    # 数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate)

    # 模型、优化器和损失函数
    model = GraphormerJEPA(input_dim, hidden_dim, output_dim, max_degree=max_degree, max_nodes=max_nodes, alpha=alpha).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = FocalLoss(gamma=2.0, pos_weight=pos_weight)

    total_params = count_parameters(model)
    print(f"模型的总参数量: {total_params}")

    best_loss = float('inf')
    best_model_path = "model_checkpoint.pt"

    # 使用ReduceLROnPlateau动态调整学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    # 添加梯度裁剪参数
    max_grad_norm = 1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        with tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs}") as pbar:
            for context_batch, target_batch in pbar:
                context_batch = context_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)

                optimizer.zero_grad()

                # 更新 autocast 用法
                with torch.amp.autocast('cuda'):
                    predicted_scores, target_scores, spatial_penalty = model(context_batch, target_batch)
                    focal_loss = loss_fn(predicted_scores, target_scores.float())
                    loss = focal_loss + spatial_penalty

                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"train_loss": loss.item()})

                # 调试信息（可选）
                # print(f"Focal Loss: {focal_loss.item()}, Spatial Loss: {spatial_penalty.item()}, Total Loss: {loss.item()}")

        train_avg_loss = total_loss / len(train_dataloader)
        val_avg_loss, val_metrics = evaluate_model(model, val_dataloader, loss_fn, prob=prob)
        print(f"Epoch {epoch}: Train Loss={train_avg_loss:.4f}, Val Loss={val_avg_loss:.4f}")
        print(f"Validation Metrics: {val_metrics}")

        scheduler.step(val_avg_loss)

        if val_avg_loss < best_loss:
            best_loss = val_avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"模型在第{epoch}轮保存，验证损失={val_avg_loss:.4f}")

    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))

    # 在测试集上评估
    test_avg_loss, test_metrics = evaluate_model(model, test_dataloader, loss_fn, prob=prob)
    print(f"测试集损失: {test_avg_loss:.4f}")
    print(f"Test Metrics: {test_metrics}")

    # 搜索最佳阈值
    best_thresh = search_best_threshold(model, val_dataloader, loss_fn)
    # 使用最佳阈值重新评估测试集
    test_avg_loss, test_metrics = evaluate_model(model, test_dataloader, loss_fn, prob=best_thresh)
    print(f"使用最佳阈值({best_thresh})的测试集结果: {test_metrics}")

    # 保存评估结果
    evaluate_and_save(model, test_dataloader, loss_fn, save_path="test_evaluation_results.csv", prob=best_thresh)
    print("训练和评估完成。")
