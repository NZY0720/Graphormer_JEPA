import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from utils import (
    count_parameters,
    load_data,
    split_graph_into_subgraphs_louvain,
    CombinedJEPAData
)
from graph_model import GraphormerJEPA

from torch.amp import autocast, GradScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameter settings
input_dim = 4  
hidden_dim = 128
output_dim = 1
lr_pretrain = 1e-3
pretrain_epochs = 200
pretrain_batch_size = 1
mask_ratio = 0.15
max_degree = 128
max_nodes = 50000
num_heads = 4
num_layers = 4
num_communities = 10
alpha = 0.001

def main():
    G, data, positive_count, negative_count, edge_to_idx = load_data("sf_graph_final.json")
    subgraphs = split_graph_into_subgraphs_louvain(G, data, num_communities=num_communities, edge_to_idx=edge_to_idx)

    # 使用合并数据集类
    combined_dataset = CombinedJEPAData(subgraphs, mask_ratio=mask_ratio, num_samples=len(subgraphs))
    combined_dataloader = DataLoader(combined_dataset, batch_size=pretrain_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = GraphormerJEPA(input_dim, hidden_dim, output_dim, max_degree=max_degree, max_nodes=max_nodes, alpha=alpha).to(device)
    optimizer_pretrain = optim.Adam(model.parameters(), lr=lr_pretrain)

    scaler = GradScaler(device='cuda' if torch.cuda.is_available() else 'cpu')
    total_params = count_parameters(model)
    print(f"Total number of model parameters: {total_params}")

    best_pretrain_loss = float('inf')
    best_pretrain_model_path = "pretrain_model_checkpoint.pt"

    pretrain_losses = []

    print("Starting JEPA-style Pre-training...")

    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for context_batch, target_batch in combined_dataloader:
            context_batch = context_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            optimizer_pretrain.zero_grad()

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                # JEPA式预训练
                loss_pretrain = model(context_batch=context_batch, target_batch=target_batch, pretrain=True)

            scaler.scale(loss_pretrain).backward()
            scaler.step(optimizer_pretrain)
            scaler.update()

            total_loss += loss_pretrain.item()
            total_samples += 1

        pretrain_avg_loss = total_loss / total_samples
        pretrain_losses.append(pretrain_avg_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Pretrain Epoch {epoch}: Avg D Loss={pretrain_avg_loss:.4f}")

        if pretrain_avg_loss < best_pretrain_loss:
            best_pretrain_loss = pretrain_avg_loss
            torch.save(model.state_dict(), best_pretrain_model_path)
            if epoch % 10 == 0 or epoch == 1:
                print(f"Best model saved at epoch {epoch} with D={pretrain_avg_loss:.4f}")

    model.load_state_dict(torch.load(best_pretrain_model_path))

    pretrain_loss_history = pd.DataFrame({
        'epoch': range(1, pretrain_epochs + 1),
        'pretrain_loss(D)': pretrain_losses
    })
    pretrain_loss_history.to_csv("pretrain_loss_history.csv", index=False)
    print("Pre-training losses saved to pretrain_loss_history.csv")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, pretrain_epochs + 1), pretrain_losses, label='Pretrain Energy (D)')
    plt.xlabel('Epoch')
    plt.ylabel('D Loss')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("pretrain_loss_curve.png")
    plt.close()
    print("JEPA-style pre-training completed.")

if __name__ == "__main__":
    main()
