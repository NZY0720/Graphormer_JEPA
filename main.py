# main.py

import torch
from torch_geometric.loader import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import os

from utils import load_multiple_graphs, JEPACommunityDataset, count_parameters
from graph_model import GraphormerJEPA

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ===== 1. 参数设置 =====
    graphs_dir = "/workspace/GraphJEPA/pre-train_1/graphs"  # 多个 JSON 图文件路径
    num_communities = 20
    # JEPA 数据集中，上下文子图 : 目标子图 = 1 : 9
    ratio = 9

    input_dim = 4
    hidden_dim = 32
    lr = 1e-3
    epochs = 10
    batch_size = 1
    num_workers = 4
    max_degree = 128
    max_nodes = 50000
    num_heads = 2
    num_layers = 2
    dropout = 0.1
    delta = 1.0  # Huber Loss 的阈值

    # ===== 2. 加载并合并子图 =====
    subgraphs = load_multiple_graphs(graphs_dir, num_communities=num_communities)

    dataset = JEPACommunityDataset(subgraphs, ratio=ratio)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # ===== 3. 初始化模型 =====
    model = GraphormerJEPA(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        max_degree=max_degree,
        max_nodes=max_nodes,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        delta=delta
    ).to(device)

    # 打印模型总参数量
    total_params = count_parameters(model)
    print(f"Total number of trainable parameters: {total_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_loss = float('inf')
    best_model_path = "jepa_best_model.pt"
    loss_history = []

    # ===== 4. JEPA 预训练循环 =====
    print("Starting JEPA pre-training with ratio=1:9 (context:target).")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0

        for context_batch, target_batch in dataloader:
            context_batch = context_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss_jepa = model(context_batch, target_batch)
            else:
                loss_jepa = model(context_batch, target_batch)

            if scaler is not None:
                scaler.scale(loss_jepa).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_jepa.backward()
                optimizer.step()

            total_loss += loss_jepa.item()
            count += 1

        avg_loss = total_loss / count
        loss_history.append(avg_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs}, JEPA Loss = {avg_loss:.4f}")

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)

    # ===== 5. 加载最优模型, 保存曲线 =====
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Best model loaded with final loss {best_loss:.4f}")

    df = pd.DataFrame({
        "epoch": range(1, epochs + 1),
        "JEPA_loss": loss_history
    })
    df.to_csv("pretrain_results/jepa_loss_history.csv", index=False)
    print("Saved JEPA loss history to jepa_loss_history.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_history, label="JEPA Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pretrain_results/jepa_loss_curve.png")
    plt.close()
    print("JEPA pre-training completed. Loss curve saved.")

if __name__ == "__main__":
    main()
