import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from tqdm import tqdm
import os

# 导入 GraphTransformer 模块
from graph_transformer import GraphTransformer

# 1. 数据准备已经在 data_generation.py 中完成，下面加载并使用

# 2. JEPA 模型定义
class GraphJEPAModel(nn.Module):
    def __init__(self, graph_in_channels, graph_hidden_channels, graph_out_channels,
                 transformer_num_heads, transformer_num_layers, dropout=0.0):
        super(GraphJEPAModel, self).__init__()
        
        # 图神经网络部分
        self.gcn1 = GCNConv(graph_in_channels, graph_hidden_channels)
        self.gcn2 = GCNConv(graph_hidden_channels, graph_out_channels)
        
        # Graph Transformer 部分
        self.transformer = GraphTransformer(
            embed_dim=graph_out_channels,
            num_heads=transformer_num_heads,
            num_layers=transformer_num_layers,
            dropout=dropout
        )
        
        # Predictor 部分
        self.predictor = nn.Sequential(
            nn.Linear(graph_out_channels, graph_out_channels // 2),
            nn.ReLU(),
            nn.Linear(graph_out_channels // 2, graph_out_channels)
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, graph_data):
        """
        graph_data: PyG的Data对象，包含x和edge_index
        """
        x, edge_index = graph_data.x, graph_data.edge_index
        # GCN 前向传播
        x = self.gcn1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        # Transformer 前向传播
        x = self.transformer(x, edge_index)

        # Predictor 前向传播
        x_pred = self.predictor(x)

        return x, x_pred

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
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Graph data file not found at {data_path}. Please run data_generation.py first.")
    
    graph_data = torch.load(data_path).to(device)
    print(f"Loaded graph data: {graph_data.num_nodes} nodes, {graph_data.num_edges//2} edges.")
    
    # 初始化模型
    model = GraphJEPAModel(
        graph_in_channels=2,  # 经度和纬度
        graph_hidden_channels=64,
        graph_out_channels=128,
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
    
    # 模拟训练数据标签（示例）
    target = torch.randn_like(graph_data.x).to(device)  # 假设目标与输入维度相同
    
    # 训练循环
    num_epochs = 200
    for epoch in tqdm(range(1, num_epochs + 1), desc="训练进度"):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        x, x_pred = model(graph_data)
        
        # 计算损失
        loss = criterion(x_pred, x)
        
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
            graph_in_channels=2,
            graph_hidden_channels=64,
            graph_out_channels=128,
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
