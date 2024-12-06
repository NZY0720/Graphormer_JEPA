# test.py

import torch
import torch.nn as nn
from torch_geometric.data import Data
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 导入 GraphJEPAModel 和 GraphTransformer
from graph_transformer import GraphTransformer
from main import GraphJEPAModel  # 确保 main.py 和 test.py 在同一目录下

def load_trained_model(model_path, device, in_channels, hidden_channels, out_channels,
                      transformer_embed_dim, transformer_num_heads, transformer_num_layers, dropout):
    """
    加载已训练好的模型。

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

def perform_rolling_prediction(model, data, device, t=3, n=24, total_steps=75):
    """
    使用模型进行滚动预测。

    Args:
        model (GraphJEPAModel): 训练好的模型。
        data (Data): 测试集数据对象。
        device (torch.device): 设备。
        t (int): 每次预测的时间步数。
        n (int): 输入窗口大小。
        total_steps (int): 需要预测的总时间步数。

    Returns:
        torch.Tensor: 所有预测结果，形状为 [num_nodes, total_steps, 1]。
    """
    model.eval()
    num_nodes = data.x.size(0)
    test_input = data.x[:, :n, :]  # 初始输入窗口 [num_nodes, n, 1]
    
    # 初始化预测结果列表
    all_predictions = []
    
    # 初始化当前输入为初始窗口
    current_input = test_input.clone()

    with torch.no_grad():
        for step in range(0, total_steps, t):
            remaining_steps = total_steps - step
            current_t = min(t, remaining_steps)  # 处理最后一次预测可能不足 t 步的情况

            # 确保输入窗口大小为 n
            if current_input.size(1) < n:
                padding = torch.zeros((num_nodes, n - current_input.size(1), 1)).to(device)
                current_input = torch.cat([padding, current_input], dim=1)

            # 创建 Data 对象
            current_data = Data(x=current_input, edge_index=data.edge_index, edge_attr=data.edge_attr).to(device)

            # 前向传播
            _, pred = model(current_data)  # pred shape: [num_nodes, n, 1]
            
            # 提取最后 current_t 步预测
            pred_final = pred[:, -current_t:, :]  # [num_nodes, current_t, 1]

            # 添加预测结果
            all_predictions.append(pred_final)

            # 更新当前输入窗口
            if current_t < t:
                # 如果剩余步数不足 t 步，仅添加可预测的步数
                current_input = torch.cat([current_input[:, t - current_t:, :], pred_final], dim=1)
            else:
                current_input = torch.cat([current_input[:, t:, :], pred_final], dim=1)

    # 合并所有预测
    all_predictions = torch.cat(all_predictions, dim=1)  # [num_nodes, total_steps, 1]
    return all_predictions

def evaluate_predictions(predictions, true_values):
    """
    计算 MAE 和 RMSE。

    Args:
        predictions (torch.Tensor): 预测值，形状为 [num_nodes, total_steps, 1]。
        true_values (torch.Tensor): 真实值，形状为 [num_nodes, total_steps, 1]。

    Returns:
        tuple: (mae, rmse)
    """
    # 确保 predictions 和 true_values 具有相同的时间步数
    min_steps = min(predictions.size(1), true_values.size(1))
    predictions = predictions[:, :min_steps, :].detach().cpu().numpy().flatten()
    true_values = true_values[:, :min_steps, :].detach().cpu().numpy().flatten()
    mae = mean_absolute_error(true_values, predictions)
    rmse = mean_squared_error(true_values, predictions, squared=False)
    return mae, rmse

def visualize_and_save_results(predictions, true_values, data, node_indices=[0]):
    """
    可视化并保存预测结果。

    Args:
        predictions (torch.Tensor): 预测值，形状为 [num_nodes, total_steps, 1]。
        true_values (torch.Tensor): 真实值，形状为 [num_nodes, total_steps, 1]。
        data (Data): 测试集数据对象。
        node_indices (list): 需要可视化的节点索引。
    """
    for node_index in node_indices:
        # 确定节点类型
        if data.is_charging_station[node_index]:
            node_type = "Charging Station Node"
        else:
            node_type = "Road Node"

        # 获取预测值和真实值
        prediction = predictions[node_index].cpu().numpy().flatten()  # [total_steps]
        true_val = true_values[node_index].cpu().numpy().flatten()     # [total_steps]

        # 创建时间步列表
        time_steps = np.arange(len(prediction))

        # 检查长度是否匹配
        if len(time_steps) != len(true_val):
            print(f"节点 {node_index} 的预测步数 ({len(prediction)}) 与真实步数 ({len(true_val)}) 不匹配。跳过可视化。")
            continue

        # 绘制真实值和预测值对比图
        plt.figure(figsize=(12, 6))
        plt.title(f"{node_type} {node_index} Prediction vs True Values")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.plot(time_steps, true_val, label="True Values", linestyle='--')
        plt.plot(time_steps, prediction, label="Predicted Values", linestyle='-')
        plt.legend()
        plt.savefig(f"node_{node_index}_prediction_vs_true.png")
        plt.show()

        # 将预测结果和真实值保存到 CSV 文件
        results_df = pd.DataFrame({
            "Time Step": time_steps,
            "True Value": true_val,
            "Predicted Value": prediction
        })
        results_df.to_csv(f"node_{node_index}_prediction_vs_true.csv", index=False)
        print(f"预测结果已保存到 node_{node_index}_prediction_vs_true.csv")

def main():
    """
    测试模型在测试集上的表现。
    """
    # 参数设置
    t = 3  # 每次预测的时间步数
    n = 24  # 输入窗口大小
    total_steps = 75  # 需要预测的总时间步数

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 模型参数（需与训练时一致）
    in_channels = 1  # 输入特征维度
    hidden_channels = 64
    out_channels = 128
    transformer_embed_dim = 256
    transformer_num_heads = 8
    transformer_num_layers = 6
    dropout = 0.1

    # 加载测试数据
    test_data_path = os.path.join("saved_models", "test_data.pt")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError("测试数据文件不存在。请先运行 data_generation.py 生成数据。")
    
    test_data_dict = torch.load(test_data_path, map_location=device)
    test_data = test_data_dict['data'].to(device)
    test_target = test_data_dict['target'].to(device)
    print(f"加载测试数据：输入 {test_data.x.size(1)} 个时间步，目标 {test_target.size(1)} 个时间步")

    # 加载最佳模型
    best_model_path = os.path.join("saved_models", "best_graph_jepa_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError("最佳模型文件不存在。请先运行 main.py 进行训练。")
    
    model = load_trained_model(
        model_path=best_model_path,
        device=device,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        transformer_embed_dim=transformer_embed_dim,
        transformer_num_heads=transformer_num_heads,
        transformer_num_layers=transformer_num_layers,
        dropout=dropout
    )
    print("已加载最佳模型。")

    # 进行滚动预测
    predictions = perform_rolling_prediction(model, test_data, device, t=t, n=n, total_steps=total_steps)
    print("滚动预测已完成。")

    # 计算评价指标
    # 由于 test_target 只有 3 步，这里仅计算与最后 3 步的评价指标
    mae, rmse = evaluate_predictions(predictions[:, -t:, :], test_target)
    print(f"测试集最后 {t} 个时间步 MAE: {mae:.6f}, RMSE: {rmse:.6f}")

    # 可视化并保存结果
    # 提取对应的真实值
    true_val_full = test_data.x[:, n:n+total_steps, :]  # [num_nodes, 75, 1]
    visualize_and_save_results(predictions, true_val_full, test_data, node_indices=[0, 10, 20])

if __name__ == "__main__":
    main()
