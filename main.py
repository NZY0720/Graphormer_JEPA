# main.py
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.serialization
import os
import gc  # For garbage collection
from torch_geometric.data.data import Data, DataEdgeAttr  # Add PyG classes for safe serialization
from utils import load_multiple_graphs, JEPACommunityDataset, count_parameters, MemoryEfficientJEPADataLoader
from graph_model import GraphormerJEPA

# Add PyG classes to safe globals for serialization
torch.serialization.add_safe_globals([Data, DataEdgeAttr])

def main():
    # Set device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ===== 1. Parameter Setup =====
    graphs_dir = "/workspace/RPFM/pre-train/graphs"  # Directory containing multiple JSON graph files
    num_communities = 20  # Number of communities to split each graph into
    ratio = 9  # Context to target subgraph ratio

    # Model and training hyperparameters
    input_dim = 4  # Modify accordingly to the dataset
    hidden_dim = 64  # 2^n
    lr = 1e-3  # For fast astringency
    epochs = 5  # A small number of epochs will do
    batch_size = 4  # Batch size for gradient accumulation
    num_workers = 4
    max_degree = 128
    max_nodes = 50000
    num_heads = 4  # 2^n
    num_layers = 2  # 2^n
    dropout = 0.1
    delta = 1.0  # Huber Loss threshold
    
    # Memory optimization parameters
    accumulation_steps = 4  # Accumulate gradients over multiple steps
    max_nodes_per_batch = 2000  # Maximum nodes per batch for memory efficiency
    effective_batch_size = batch_size // accumulation_steps  # Use smaller batches with accumulation

    # ===== 2. Load and Merge Subgraphs =====
    if os.path.exists("all_subgraphs.pt"):
        # If a cached file exists, load subgraphs directly
        print("Loading subgraphs from cached file...")
        try:
            # Load without weights_only parameter for older PyTorch versions
            subgraphs = torch.load("all_subgraphs.pt")
            print("Successfully loaded subgraphs")
        except Exception as e:
            print(f"Error loading subgraphs: {e}")
            print("Falling back to regenerating subgraphs...")
            # If loading fails, regenerate the subgraphs
            subgraphs = load_multiple_graphs(
                graphs_dir,
                num_communities=num_communities,
                save_path="all_subgraphs.pt"
            )
    else:
        # Otherwise, load using the provided utility function
        subgraphs = load_multiple_graphs(
            graphs_dir,
            num_communities=num_communities,
            save_path="all_subgraphs.pt"
        )

    # Create the JEPA dataset with the specified ratio
    dataset = JEPACommunityDataset(subgraphs, ratio=ratio)

    # Initialize the memory-efficient dataloader
    dataloader = MemoryEfficientJEPADataLoader(
        dataset,
        batch_size=effective_batch_size,
        max_nodes_per_batch=max_nodes_per_batch,
        shuffle=False
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

    # ===== 4. JEPA Pretraining Loop with Memory Optimization =====
    print(f"Starting JEPA pre-training with ratio=1:{ratio} (context:target).")
    print(f"Using accumulation steps: {accumulation_steps}, max nodes per batch: {max_nodes_per_batch}")
    
    for epoch in range(1, epochs + 1):
        model.train()  # Set model to training mode
        total_loss = 0.0  # Initialize total loss for the epoch
        count = 0  # Counter for the number of batches
        
        # Reset gradients at the beginning of each epoch
        optimizer.zero_grad()

        # Iterate over batches from the DataLoader
        for i, (context_batch, target_batch) in enumerate(dataloader):
            # Move context and target batches to the specified device
            context_batch = context_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            
            # Use automatic mixed precision
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_jepa = model(context_batch, target_batch, pretrain=True)
            
            # Scale the loss for gradient accumulation
            if scaler is not None:
                loss_jepa = loss_jepa.mean() / accumulation_steps
                scaler.scale(loss_jepa).backward()
            else:
                loss_jepa = loss_jepa / accumulation_steps
                loss_jepa.backward()
            
            # Update weights only after accumulation_steps or at the end of dataloader
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()  # Reset gradients after update
            
            # Record the loss
            total_loss += loss_jepa.item() * accumulation_steps
            count += 1

            # Free memory occasionally
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()

        # Compute the average loss for the epoch
        avg_loss = total_loss / count
        loss_history.append(avg_loss)

        print(f"Epoch {epoch}/{epochs}, JEPA Loss = {avg_loss:.4f}")

        # Save the model if the current loss is the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            
        # Force garbage collection after each epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
