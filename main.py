import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from tqdm import tqdm
import os
import numpy as np

# 导入更新后的 GraphTransformer 模块
from graph_transformer import GraphTransformer

# 1. 修改后的示例数据生成函数
def generate_synthetic_data(num_nodes, num_edges, num_time_steps):
    """
    生成包含时间序列特征的合成图数据。

    Args:
        num_nodes (int): 节点数量
        num_edges (int): 边数量
        num_time_steps (int): 时间步数

    Returns:
        Data: PyG 的图数据对象
    """
    # 假设 20% 的节点是充电站节点，其余是道路节点
    num_charging_stations = int(num_nodes * 0.2)
    num_road_nodes = num_nodes - num_charging_stations

    # 创建节点特征
    # 道路节点的特征：道路流量时间序列数据
    road_node_features = np.random.rand(num_road_nodes, num_time_steps, 1)
    # 充电站节点的特征：负荷时间序列数据
    charging_node_features = np.random.rand(num_charging_stations, num_time_steps, 1)

    # 合并节点特征
    node_features = np.vstack([road_node_features, charging_node_features])

    # 创建边索引
    edge_index = np.random.randint(0, num_nodes, size=(2, num_edges))

    # 创建边特征：道路流量时间序列数据
    edge_features = np.random.rand(num_edges, num_time_steps, 1)

    # 转换为 PyTorch 张量
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    # 创建 PyG 的 Data 对象
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

    # 添加充电站节点的掩码，用于区分节点类型
    data.is_charging_station = torch.zeros(num_nodes, dtype=torch.bool)
    data.is_charging_station[num_road_nodes:] = True

    return data

# 2. JEPA 模型定义，更新以处理时间序列数据
class GraphJEPAModel(nn.Module):
    def __init__(self, graph_in_channels, graph_hidden_channels, graph_out_channels,
                 transformer_embed_dim, transformer_num_heads, transformer_num_layers, dropout=0.1):
        super(GraphJEPAModel, self).__init__()
        
        # 图神经网络部分，需要处理时间序列数据
        self.gcn1 = GCNConv(graph_in_channels, graph_hidden_channels)
        self.gcn2 = GCNConv(graph_hidden_channels, graph_out_channels)
        
        # Graph Transformer 部分，更新以处理时间序列数据
        self.transformer = GraphTransformer(
            in_channels=graph_out_channels,
            embed_dim=transformer_embed_dim,
            num_heads=transformer_num_heads,
            num_layers=transformer_num_layers,
            dropout=dropout
        )
        
        # Predictor 部分
        self.predictor = nn.Sequential(
            nn.Linear(transformer_embed_dim, transformer_embed_dim // 2),
            nn.ReLU(),
            nn.Linear(transformer_embed_dim // 2, graph_out_channels)
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, graph_data):
        x, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr
        # x: [num_nodes, num_time_steps, in_channels]
        # edge_attr: [num_edges, num_time_steps, edge_in_channels]

        num_nodes = x.size(0)
        num_time_steps = x.size(1)
        in_channels = x.size(2)

        # 将 x 和 edge_attr 展平，以适应 GCN 的输入
        x = x.reshape(num_nodes * num_time_steps, in_channels)
        edge_index_time = self.expand_edge_index(edge_index, num_time_steps, num_nodes)

        # GCN 前向传播
        x = self.gcn1(x, edge_index_time)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index_time)
        x = self.activation(x)
        x = self.dropout(x)

        # 恢复 x 的形状
        x = x.reshape(num_nodes, num_time_steps, -1)

        # Graph Transformer 前向传播
        x = self.transformer(x, edge_index)

        # Predictor 前向传播
        x_pred = self.predictor(x)

        return x, x_pred

    def expand_edge_index(self, edge_index, num_time_steps, num_nodes):
        """
        扩展 edge_index，使其适应时间序列数据。
        """
        edge_index_list = []
        for t in range(num_time_steps):
            # 偏移节点索引
            offset = t * num_nodes
            edge_index_t = edge_index + offset
            edge_index_list.append(edge_index_t)
        edge_index_time = torch.cat(edge_index_list, dim=1)
        return edge_index_time

# 3. 参数量展示
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 4. 训练过程展示和模型保存
def train_model():
    # 检查 CUDA 是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载图数据
    data_path = os.path.join("saved_models", "graph_data.pt")
    if os.path.exists(data_path):
        # 如果存在预先生成的图数据，则加载
        graph_data = torch.load(data_path).to(device)
        print(f"Loaded graph data: {graph_data.num_nodes} nodes, {graph_data.num_edges//2} edges.")
    else:
        # 否则，生成示例数据
        print("Graph data file not found. Generating synthetic data for testing...")
        num_nodes = 100  # 示例节点数量
        num_edges = 500  # 示例边数量
        num_time_steps = 24  # 示例时间步数
        graph_data = generate_synthetic_data(num_nodes, num_edges, num_time_steps).to(device)
        print(f"Synthetic graph data generated: {graph_data.num_nodes} nodes, {graph_data.num_edges//2} edges.")

    # 初始化模型，更新参数以匹配新的 GraphTransformer
    in_channels = graph_data.x.size(2)  # 输入特征维度
    edge_in_channels = graph_data.edge_attr.size(2)
    model = GraphJEPAModel(
        graph_in_channels=in_channels,
        graph_hidden_channels=64,
        graph_out_channels=128,
        transformer_embed_dim=256,
        transformer_num_heads=8,
        transformer_num_layers=6,
        dropout=0.1
    ).to(device)
    
    print(f"模型参数总数: {count_parameters(model)}")
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 创建保存模型的目录
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    
    # 模拟训练数据标签（示例），预测下一个时间步的数据
    target = graph_data.x[:, 1:, :]  # 真实值，移除第一个时间步
    prediction_input = graph_data.x[:, :-1, :]  # 模型输入，移除最后一个时间步

    # 将 graph_data.x 替换为 prediction_input
    graph_data.x = prediction_input

    # 训练循环
    num_epochs = 50
    for epoch in tqdm(range(1, num_epochs + 1), desc="训练进度"):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        x, x_pred = model(graph_data)
        
        # 计算损失
        loss = criterion(x_pred, target)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 打印损失
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")
    
    # 保存模型
    model_path = os.path.join(model_dir, "graph_jepa_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"训练好的模型已保存到 {model_path}")
    
    # 可选：加载模型进行推理
    def load_model(model_path, device):
        model = GraphJEPAModel(
            graph_in_channels=in_channels,
            graph_hidden_channels=64,
            graph_out_channels=128,
            transformer_embed_dim=256,
            transformer_num_heads=8,
            transformer_num_layers=6,
            dropout=0.1
        ).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    loaded_model = load_model(model_path, device)
    with torch.no_grad():
        loaded_x, loaded_x_pred = loaded_model(graph_data)
        print("加载模型后的节点嵌入 (x) 形状:", loaded_x.shape)
        print("加载模型后的预测嵌入 (x_pred) 形状:", loaded_x_pred.shape)

if __name__ == "__main__":
    train_model()
