# data_generation.py

import torch
from torch_geometric.data import Data
import numpy as np
import os

def generate_synthetic_data(num_nodes=100, num_edges=500, num_time_steps=1000, seed=42):
    """
    生成包含时间序列特征的合成图数据。

    Args:
        num_nodes (int): 图中的节点数量。
        num_edges (int): 图中的边数量。
        num_time_steps (int): 时间步数。
        seed (int): 随机种子，用于可重复性。

    Returns:
        Data: 包含节点和边特征的 PyG 图数据对象。
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 定义节点类型比例
    proportion_charging = 0.2
    num_charging_stations = int(num_nodes * proportion_charging)
    num_road_nodes = num_nodes - num_charging_stations

    # 生成节点特征
    time = np.linspace(0, 20 * np.pi, num_time_steps)
    node_features = []

    for i in range(num_nodes):
        if i < num_road_nodes:
            # 道路节点：不同频率和相位的正弦波
            freq = np.random.uniform(0.5, 1.5)
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.uniform(0.5, 1.5)
            signal = amplitude * np.sin(freq * time + phase)
        else:
            # 充电站节点：不同频率和相位的余弦波
            freq = np.random.uniform(0.5, 1.5)
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.uniform(0.5, 1.5)
            signal = amplitude * np.cos(freq * time + phase)
        # 添加噪声
        noise = np.random.normal(0, 0.1, num_time_steps)
        signal += noise
        node_features.append(signal)

    node_features = np.array(node_features)  # 形状：[num_nodes, num_time_steps]
    node_features = node_features[:, :, np.newaxis]  # 形状：[num_nodes, num_time_steps, 1]
    node_features = torch.tensor(node_features, dtype=torch.float)

    # 生成边索引
    edge_index = []
    while len(edge_index) < num_edges:
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        if src != dst:
            edge_index.append([src, dst])
    edge_index = np.array(edge_index).T  # 形状：[2, num_edges]
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # 生成边特征
    edge_features = []
    for i in range(num_edges):
        # 不同频率和相位的正弦和余弦波的组合
        freq = np.random.uniform(0.3, 1.2)
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.3, 1.2)
        signal = amplitude * np.sin(freq * time + phase) + amplitude * np.cos(freq * time + phase)
        # 添加噪声
        noise = np.random.normal(0, 0.1, num_time_steps)
        signal += noise
        edge_features.append(signal)

    edge_features = np.array(edge_features)  # 形状：[num_edges, num_time_steps]
    edge_features = edge_features[:, :, np.newaxis]  # 形状：[num_edges, num_time_steps, 1]
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    # 创建 PyG 的 Data 对象
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

    # 添加节点类型掩码
    data.is_charging_station = torch.zeros(num_nodes, dtype=torch.bool)
    data.is_charging_station[num_road_nodes:] = True

    return data

def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, t=3):
    """
    将数据集分割为训练集、验证集和测试集。

    Args:
        data (Data): 完整的图数据对象。
        train_ratio (float): 训练集比例。
        val_ratio (float): 验证集比例。
        test_ratio (float): 测试集比例。
        t (int): 预测窗口大小。

    Returns:
        tuple: (train_data, train_target, val_data, val_target, test_data, test_target)
    """
    total_time_steps = data.x.size(1)
    train_time_steps = int(total_time_steps * train_ratio)
    val_time_steps = int(total_time_steps * val_ratio)
    test_time_steps = total_time_steps - train_time_steps - val_time_steps

    # 保证时间步不重叠
    # 训练集：[0, train_time_steps - t)
    # 验证集：[train_time_steps, train_time_steps + val_time_steps - t)
    # 测试集：[train_time_steps + val_time_steps, total_time_steps - t)
    train_input = data.x[:, :train_time_steps - t, :]  # [num_nodes, train_time_steps - t, 1]
    train_target = data.x[:, train_time_steps - t:train_time_steps, :]  # [num_nodes, t, 1]

    val_input = data.x[:, train_time_steps:train_time_steps + val_time_steps - t, :]  # [num_nodes, val_time_steps - t, 1]
    val_target = data.x[:, train_time_steps + val_time_steps - t:train_time_steps + val_time_steps, :]  # [num_nodes, t, 1]

    test_input = data.x[:, train_time_steps + val_time_steps:test_time_steps + train_time_steps + val_time_steps - t, :]  # [num_nodes, test_time_steps - t, 1]
    test_target = data.x[:, train_time_steps + val_time_steps - t:test_time_steps + train_time_steps + val_time_steps, :]  # [num_nodes, t, 1]

    # 创建分割后的 Data 对象
    train_data = Data(x=train_input, edge_index=data.edge_index, edge_attr=data.edge_attr)
    train_data.is_charging_station = data.is_charging_station

    val_data = Data(x=val_input, edge_index=data.edge_index, edge_attr=data.edge_attr)
    val_data.is_charging_station = data.is_charging_station

    test_data = Data(x=test_input, edge_index=data.edge_index, edge_attr=data.edge_attr)
    test_data.is_charging_station = data.is_charging_station

    return train_data, train_target, val_data, val_target, test_data, test_target

def main():
    # 参数设置
    num_nodes = 100
    num_edges = 500
    num_time_steps = 1000
    seed = 42
    t = 3  # 预测窗口大小

    # 生成数据
    data = generate_synthetic_data(num_nodes, num_edges, num_time_steps, seed)
    print(f"生成合成数据：{data.num_nodes} 个节点，{data.num_edges//2} 条边，{data.x.size(1)} 个时间步。")

    # 分割数据
    train_data, train_target, val_data, val_target, test_data, test_target = split_data(
        data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, t=t)
    print(f"数据集分割：训练集 {train_data.x.size(1)} 步，验证集 {val_data.x.size(1)} 步，测试集 {test_data.x.size(1)} 步。")

    # 保存分割后的数据
    os.makedirs("saved_models", exist_ok=True)
    torch.save({'data': train_data, 'target': train_target}, os.path.join("saved_models", "train_data.pt"))
    torch.save({'data': val_data, 'target': val_target}, os.path.join("saved_models", "val_data.pt"))
    torch.save({'data': test_data, 'target': test_target}, os.path.join("saved_models", "test_data.pt"))
    print("分割后的数据已保存到 saved_models 目录下。")

if __name__ == "__main__":
    main()
