# main.py

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import pandas as pd
import matplotlib.pyplot as plt  # For plotting loss curves

from utils import (
    FocalLoss,
    count_parameters,
    load_data,
    GraphPairDataset,
    split_graph_into_subgraphs_louvain,
    evaluate_model,
    evaluate_and_save,
    custom_collate,
    search_best_threshold
)
from graph_model import GraphormerJEPA

# Add the missing import for random_split
from torch.utils.data import random_split


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameter settings
input_dim = 4
hidden_dim = 64
output_dim = 1
lr = 1e-5  
epochs = 500  
batch_size = 1
max_degree = 128
max_nodes = 50000
num_heads = 4
num_layers = 4
num_communities = 10
prob = 0.5
alpha = 0.001  # Reduced weight for spatial loss


if __name__ == "__main__":
    import torch.multiprocessing as mp
    from tqdm import tqdm
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Load data
    G, data, positive_count, negative_count = load_data("updated_san_francisco_graph.json")

    # Calculate positive class weight
    if positive_count > 0:
        pos_weight_value = (negative_count / positive_count) * 0.1
    else:
        pos_weight_value = 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float).to(device)
    print(f"Positive Count: {positive_count}, Negative Count: {negative_count}")
    print(f"pos_weight: {pos_weight_value}")

    # Split into subgraphs
    subgraphs_nx = split_graph_into_subgraphs_louvain(G, data, num_communities=num_communities)
    subgraphs = subgraphs_nx
    dataset = GraphPairDataset(subgraphs, num_samples=len(subgraphs))

    # Split dataset using random_split
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate)

    # Model, optimizer, and loss function
    model = GraphormerJEPA(input_dim, hidden_dim, output_dim, max_degree=max_degree, max_nodes=max_nodes, alpha=alpha).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = FocalLoss(gamma=2.0, pos_weight=pos_weight)

    total_params = count_parameters(model)
    print(f"Total number of model parameters: {total_params}")

    best_loss = float('inf')
    best_model_path = "model_checkpoint.pt"

    # Use ReduceLROnPlateau to adjust learning rate dynamically
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    # Add gradient clipping parameter
    max_grad_norm = 1.0

    # Initialize loss history lists
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        with tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs}") as pbar:
            for context_batch, target_batch in pbar:
                context_batch = context_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)

                optimizer.zero_grad()

                # Update autocast usage
                with torch.amp.autocast('cuda'):
                    predicted_scores, target_scores, spatial_penalty = model(context_batch, target_batch)
                    focal_loss = loss_fn(predicted_scores, target_scores.float())
                    loss = focal_loss + spatial_penalty

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"train_loss": loss.item()})

        # Calculate average training loss
        train_avg_loss = total_loss / len(train_dataloader)
        train_losses.append(train_avg_loss)  # Record training loss

        # Evaluate on validation set
        val_avg_loss, val_metrics = evaluate_model(model, val_dataloader, loss_fn, device=device, prob=prob)
        val_losses.append(val_avg_loss)  # Record validation loss

        print(f"Epoch {epoch}: Train Loss={train_avg_loss:.4f}, Val Loss={val_avg_loss:.4f}")
        print(f"Validation Metrics: {val_metrics}")

        # Update learning rate scheduler
        scheduler.step(val_avg_loss)

        # Save the best model
        if val_avg_loss < best_loss:
            best_loss = val_avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Model saved at epoch {epoch} with validation loss={val_avg_loss:.4f}")

    # Load the best model
    model.load_state_dict(torch.load(best_model_path))

    # Evaluate on test set
    test_avg_loss, test_metrics = evaluate_model(model, test_dataloader, loss_fn, device=device, prob=prob)
    print(f"Test Loss: {test_avg_loss:.4f}")
    print(f"Test Metrics: {test_metrics}")

    # Search for the best threshold
    best_thresh = search_best_threshold(model, val_dataloader, loss_fn, device=device)
    # Re-evaluate the test set using the best threshold
    test_avg_loss, test_metrics = evaluate_model(model, test_dataloader, loss_fn, device=device, prob=best_thresh)
    print(f"Test Results with Best Threshold ({best_thresh}): {test_metrics}")

    # Save evaluation results
    evaluate_and_save(model, test_dataloader, loss_fn, save_path="test_evaluation_results.csv", device=device, prob=best_thresh)
    print("Training and evaluation completed.")

    # Save training and validation losses to CSV
    loss_history = pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    loss_history.to_csv("loss_history.csv", index=False)
    print("Training and validation losses saved to loss_history.csv")

    # Plot and save loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(False)
    plt.savefig("loss_curve.png")
    plt.close()
    print("Loss curve saved to loss_curve.png")
