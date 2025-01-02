# main.py

import torch
from torch_geometric.loader import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import os

from utils import load_multiple_graphs, JEPACommunityDataset, count_parameters
from graph_model import GraphormerJEPA

def main():
    # Set device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ===== 1. Parameter Setup =====
    graphs_dir = "/workspace/GraphJEPA/pre-train/graphs"  # Directory containing multiple JSON graph files
    num_communities = 10  # Number of communities to split each graph into
    ratio = 9  # Context to target subgraph ratio

    # Model and training hyperparameters
    input_dim = 4 # Modify accordingly to the dataset
    hidden_dim = 128 # 2^n
    lr = 1e-3 # For fast astringency
    epochs = 5 # A small number of epochs will do
    batch_size = 2 # Now supporting multi-batches
    num_workers = 4 
    max_degree = 128
    max_nodes = 50000
    num_heads = 4 # 2^n
    num_layers = 4 # 2^n
    dropout = 0.1
    delta = 1.0  # Huber Loss threshold

    # ===== 2. Load and Merge Subgraphs =====
    if os.path.exists("all_subgraphs.pt"):
        # If a cached file exists, load subgraphs directly
        print("Loading subgraphs from cached file...")
        subgraphs = torch.load("all_subgraphs.pt")
    else:
        # Otherwise, load using the provided utility function
        subgraphs = load_multiple_graphs(
            graphs_dir,
            num_communities=num_communities,
            save_path="all_subgraphs.pt"
        )

    # Create the JEPA dataset with the specified ratio
    dataset = JEPACommunityDataset(subgraphs, ratio=ratio)

    # Initialize the DataLoader with the desired batch size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Enable pin_memory for faster GPU transfers
    )

    # ===== 3. Initialize Model =====
    model = GraphormerJEPA(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        max_degree=max_degree,
        max_nodes=max_nodes,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        delta=delta
    ).to(device)  # Move the model to the specified device

    # Print the total number of trainable parameters in the model
    total_params = count_parameters(model)
    print(f"Total number of trainable parameters: {total_params}")

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Initialize GradScaler for mixed precision training if GPU is available
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # Variables to track the best loss and loss history
    best_loss = float('inf')
    best_model_path = "jepa_best_model.pt"
    loss_history = []

    # ===== 4. JEPA Pretraining Loop =====
    print("Starting JEPA pre-training with ratio=1:9 (context:target).")
    for epoch in range(1, epochs + 1):
        model.train()  # Set model to training mode
        total_loss = 0.0  # Initialize total loss for the epoch
        count = 0  # Counter for the number of batches

        # Iterate over batches from the DataLoader
        for context_batch, target_batch in dataloader:
            # Move context and target batches to the specified device
            context_batch = context_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            optimizer.zero_grad()  # Reset optimizer gradients

            if scaler is not None:
                # Use automatic mixed precision for faster training on GPU
                with torch.cuda.amp.autocast():
                    loss_jepa = model(context_batch, target_batch, pretrain=True)
            else:
                # Standard precision training
                loss_jepa = model(context_batch, target_batch, pretrain=True)

            if scaler is not None:
                # Scale the loss and perform backpropagation
                loss_jepa = loss_jepa.mean()
                scaler.scale(loss_jepa).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Backpropagation and optimizer step without scaling
                loss_jepa.backward()
                optimizer.step()

            # Accumulate the loss and increment the batch counter
            total_loss += loss_jepa.item()
            count += 1

        # Compute the average loss for the epoch
        avg_loss = total_loss / count
        loss_history.append(avg_loss)

        print(f"Epoch {epoch}/{epochs}, JEPA Loss = {avg_loss:.4f}")

        # Save the model if the current loss is the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)

    # ===== 5. Load Best Model and Save Loss Curve =====
    if os.path.exists(best_model_path):
        # Load the best model's state dictionary
        model.load_state_dict(torch.load(best_model_path))
        print(f"Best model loaded with final loss {best_loss:.4f}")

    # Ensure the 'pretrain_results' directory exists
    os.makedirs("pretrain_results", exist_ok=True)

    # Save loss history to a CSV file for later analysis
    df = pd.DataFrame({
        "epoch": range(1, epochs + 1),
        "JEPA_loss": loss_history
    })
    df.to_csv("pretrain_results/jepa_loss_history.csv", index=False)
    print("Saved JEPA loss history to pretrain_results/jepa_loss_history.csv")

    # Plot and save the loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_history, label="JEPA Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pretrain_results/jepa_loss_curve.png")
    plt.close()
    print("JEPA pre-training completed. Loss curve saved to pretrain_results/jepa_loss_curve.png.")

if __name__ == "__main__":
    main()
