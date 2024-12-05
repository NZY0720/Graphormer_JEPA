# main.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 导入 GraphTransformer 模块
from graph_transformer import GraphTransformer

class GraphJEPAModel(nn.Module):
    def __init__(self, graph_in_channels, graph_hidden_channels, graph_out_channels,
                 transformer_embed_dim, transformer_num_heads, transformer_num_layers, dropout=0.1):
        """
        初始化 GraphJEPA 模型。

        Args:
            graph_in_channels (int): 输入特征的维度。
            graph_hidden_channels (int): GCN 隐藏层的维度。
            graph_out_channels (int): GCN 输出层的维度。
            transformer_embed_dim (int): Transformer 的嵌入维度。
            transformer_num_heads (int): 多头注意力机制的头数。
            transformer_num_layers (int): Transformer 编码器的层数。
            dropout (float): Dropout 概率。
        """
        super(GraphJEPAModel, self).__init__()
        
        # 图卷积网络部分
        self.gcn1 = GCNConv(graph_in_channels, graph_hidden_channels)
        self.gcn2 = GCNConv(graph_hidden_channels, graph_out_channels)
        
        # Graph Transformer 部分
        self.transformer = GraphTransformer(
            in_channels=graph_out_channels,
            embed_dim=transformer_embed_dim,
            num_heads=transformer_num_heads,
            num_layers=transformer_num_layers,
            dropout=dropout
        )
        
        # Predictor 部分，增加层数以提升表达能力
        self.predictor = nn.Sequential(
            nn.Linear(transformer_embed_dim, transformer_embed_dim),
            nn.ReLU(),
            nn.Linear(transformer_embed_dim, transformer_embed_dim // 2),
            nn.ReLU(),
            nn.Linear(transformer_embed_dim // 2, 1)  # 输出 1 个通道
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, graph_data):
        """
        前向传播。

        Args:
            graph_data (Data): 图数据对象，包含节点和边特征。

        Returns:
            tuple: (x, x_pred)
                - x (torch.Tensor): GCN 和 Transformer 提取的特征，形状为 [num_nodes, num_time_steps, embed_dim]。
                - x_pred (torch.Tensor): 预测值，形状为 [num_nodes, num_time_steps, 1]。
        """
        x, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr
        # x: [num_nodes, num_time_steps, in_channels]
        # edge_attr: [num_edges, num_time_steps, edge_in_channels]

        num_nodes = x.size(0)
        num_time_steps = x.size(1)
        in_channels = x.size(2)

        # 将 x 和 edge_attr 展平，以适应 GCN 的输入
        x = x.reshape(num_nodes * num_time_steps, in_channels)
        edge_index_time = self.expand_edge_index(edge_index, num_time_steps, num_nodes).to(x.device)

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
        x_pred = self.predictor(x)  # [num_nodes, num_time_steps, 1]

        return x, x_pred

    def expand_edge_index(self, edge_index, num_time_steps, num_nodes):
        """
        扩展 edge_index，使其适应时间序列数据。

        Args:
            edge_index (torch.Tensor): 原始边索引，形状为 [2, num_edges]。
            num_time_steps (int): 时间步数。
            num_nodes (int): 节点数量。

        Returns:
            torch.Tensor: 扩展后的边索引，形状为 [2, num_edges * num_time_steps]。
        """
        if edge_index is None:
            raise ValueError("edge_index is None. 请提供有效的 edge_index。")
        
        edge_index_list = []
        for t in range(num_time_steps):
            # 偏移节点索引
            offset = t * num_nodes
            edge_index_t = edge_index + offset
            edge_index_list.append(edge_index_t)
        edge_index_time = torch.cat(edge_index_list, dim=1)
        return edge_index_time

def count_parameters(model):
    """
    计算模型的可训练参数数量。

    Args:
        model (nn.Module): 需要计算参数的模型。

    Returns:
        int: 可训练参数总数。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model():
    """
    训练 GraphJEPA 模型，并保存最佳模型。
    """
    # 设置滚动预测参数
    n = 24  # 输入窗口大小
    t = 3   # 预测窗口大小
    validation_time_steps = 100  # 验证集的时间步数
    test_time_steps = 100  # 测试集的时间步数

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载训练、验证和测试数据
    train_data_path = os.path.join("saved_models", "train_data.pt")
    val_data_path = os.path.join("saved_models", "val_data.pt")
    test_data_path = os.path.join("saved_models", "test_data.pt")
    
    if not os.path.exists(train_data_path) or not os.path.exists(val_data_path) or not os.path.exists(test_data_path):
        raise FileNotFoundError("训练、验证或测试数据文件不存在。请先运行 data_generation.py 生成数据。")
    
    # 加载数据
    train_data_dict = torch.load(train_data_path, map_location=device)
    train_data = train_data_dict['data'].to(device)
    train_target = train_data_dict['target'].to(device)
    
    val_data_dict = torch.load(val_data_path, map_location=device)
    val_data = val_data_dict['data'].to(device)
    val_target = val_data_dict['target'].to(device)
    
    test_data_dict = torch.load(test_data_path, map_location=device)
    test_data = test_data_dict['data'].to(device)
    test_target = test_data_dict['target'].to(device)

    print(f"加载训练数据：{train_data.x.size(1)} 个时间步")
    print(f"加载验证数据：{val_data.x.size(1)} 个时间步")
    print(f"加载测试数据：{test_data.x.size(1)} 个时间步")

    # 初始化模型
    in_channels = train_data.x.size(2)  # 输入特征维度
    hidden_channels = 64
    out_channels = 128  # GCN 输出通道数
    transformer_embed_dim = 256
    transformer_num_heads = 8
    transformer_num_layers = 6
    dropout = 0.1

    model = GraphJEPAModel(
        graph_in_channels=in_channels,
        graph_hidden_channels=hidden_channels,
        graph_out_channels=out_channels,
        transformer_embed_dim=transformer_embed_dim,
        transformer_num_heads=transformer_num_heads,
        transformer_num_layers=transformer_num_layers,
        dropout=dropout
    ).to(device)
    
    print(f"模型参数总数: {count_parameters(model)}")
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    # 训练循环参数
    num_epochs = 200  # 增加训练轮数
    patience = 10  # 早停耐心值
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = os.path.join("saved_models", "best_graph_jepa_model.pth")

    # 初始化损失和指标记录列表
    train_losses = []
    val_losses = []
    train_mae = []
    train_rmse = []
    val_mae = []
    val_rmse = []

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training Progress"):
        # 训练阶段
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        _, x_pred = model(train_data)
        
        # 计算损失
        pred = x_pred[:, -t:, :]  # [num_nodes, t, 1]
        loss = criterion(pred, train_target)
        
        # 反向传播和优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        
        # 计算训练 MAE 和 RMSE
        train_mae_val = mean_absolute_error(train_target.detach().cpu().numpy().flatten(), pred.detach().cpu().numpy().flatten())
        train_rmse_val = mean_squared_error(train_target.detach().cpu().numpy().flatten(), pred.detach().cpu().numpy().flatten(), squared=False)
        train_mae.append(train_mae_val)
        train_rmse.append(train_rmse_val)
        train_losses.append(loss.item())
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            _, val_pred = model(val_data)
            val_pred_final = val_pred[:, -validation_time_steps:, :]  # [num_nodes, validation_time_steps, 1]
            val_loss = criterion(val_pred_final, val_target)
            val_losses.append(val_loss.item())
            
            # 计算验证 MAE 和 RMSE
            val_mae_val = mean_absolute_error(val_target.detach().cpu().numpy().flatten(), val_pred_final.detach().cpu().numpy().flatten())
            val_rmse_val = mean_squared_error(val_target.detach().cpu().numpy().flatten(), val_pred_final.detach().cpu().numpy().flatten(), squared=False)
            val_mae.append(val_mae_val)
            val_rmse.append(val_rmse_val)
        
        # 更新学习率调度器
        scheduler.step(val_loss)
        
        # 检查是否有改善
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            epochs_no_improve = 0
            # 保存最优模型
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch}: 验证损失下降至 {val_loss.item():.6f}，保存模型。")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch}: 验证损失未下降（当前损失: {val_loss.item():.6f}），早停计数: {epochs_no_improve}/{patience}")
        
        # 检查是否达到早停条件
        if epochs_no_improve >= patience:
            print(f"早停触发：在 {patience} 个连续的epoch中验证损失未下降。")
            break
        
        # 打印损失和评价指标
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{num_epochs}, 训练损失: {loss.item():.6f}, 训练 MAE: {train_mae_val:.6f}, 训练 RMSE: {train_rmse_val:.6f}, 验证损失: {val_loss.item():.6f}, 验证 MAE: {val_mae_val:.6f}, 验证 RMSE: {val_rmse_val:.6f}")

    # 绘制损失曲线
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_curve.png')
    plt.show()
    
    # 绘制 MAE 曲线
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(train_mae)+1), train_mae, label='Training MAE')
    plt.plot(range(1, len(val_mae)+1), val_mae, label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Training and Validation MAE')
    plt.savefig('mae_curve.png')
    plt.show()
    
    # 绘制 RMSE 曲线
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(train_rmse)+1), train_rmse, label='Training RMSE')
    plt.plot(range(1, len(val_rmse)+1), val_rmse, label='Validation RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('Training and Validation RMSE')
    plt.savefig('rmse_curve.png')
    plt.show()
    
    # 加载最优模型
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"已加载最优模型，验证损失: {best_val_loss:.6f}")
    else:
        print("未找到最优模型，使用当前模型。")
    
    # 保存最终模型
    final_model_path = os.path.join("saved_models", "final_graph_jepa_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存到 {final_model_path}")

def load_model(model_path, device, in_channels, hidden_channels, out_channels,
              transformer_embed_dim, transformer_num_heads, transformer_num_layers, dropout):
    """
    加载已保存的模型。

    Args:
        model_path (str): 模型文件路径。
        device (torch.device): 设备。
        in_channels (int): 输入特征维度。
        hidden_channels (int): GCN 隐藏层维度。
        out_channels (int): GCN 输出层维度。
        transformer_embed_dim (int): Transformer 嵌入维度。
        transformer_num_heads (int): Transformer 多头注意力头数。
        transformer_num_layers (int): Transformer 编码器层数。
        dropout (float): Dropout 概率。

    Returns:
        GraphJEPAModel: 加载的模型。
    """
    model = GraphJEPAModel(
        graph_in_channels=in_channels,
        graph_hidden_channels=hidden_channels,
        graph_out_channels=out_channels,
        transformer_embed_dim=transformer_embed_dim,
        transformer_num_heads=transformer_num_heads,
        transformer_num_layers=transformer_num_layers,
        dropout=dropout
    ).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    except Exception as e:
        print(f"加载模型时出错: {e}")
        raise e
    model.eval()
    return model

if __name__ == "__main__":
    train_model()
