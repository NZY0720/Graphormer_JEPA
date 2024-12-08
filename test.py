# test.py

import torch
from torch_geometric.loader import DataLoader
import pandas as pd
from tqdm import tqdm
import argparse

from utils import (
    FocalLoss,
    load_data,
    GraphPairDataset,
    split_graph_into_subgraphs_louvain,
    evaluate_model,
    evaluate_and_save,
    custom_collate
)
from graph_model import GraphormerJEPA

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    G, data, positive_count, negative_count = load_data(args.data_path)
    print(f"Positive Count: {positive_count}, Negative Count: {negative_count}")

    # Split into subgraphs
    print("Splitting graph into subgraphs...")
    subgraphs_nx = split_graph_into_subgraphs_louvain(G, data, num_communities=args.num_communities)
    dataset = GraphPairDataset(subgraphs_nx, num_samples=len(subgraphs_nx))

    # Create DataLoader for the test set
    test_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )

    # Initialize model
    print("Initializing model...")
    model = GraphormerJEPA(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        max_degree=args.max_degree,
        max_nodes=args.max_nodes,
        alpha=args.alpha
    ).to(device)

    # Load trained model weights
    print(f"Loading model weights from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # Define loss function
    # Calculate positive class weight
    if positive_count > 0:
        pos_weight_value = (negative_count / positive_count) * 0.1
    else:
        pos_weight_value = 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float).to(device)
    loss_fn = FocalLoss(gamma=2.0, pos_weight=pos_weight)

    # Perform evaluation
    print("Evaluating model on test set...")
    test_avg_loss, test_metrics = evaluate_model(
        model,
        test_dataloader,
        loss_fn,
        device=device,
        prob=args.prob_threshold
    )
    print(f"Test Loss: {test_avg_loss:.4f}")
    print(f"Test Metrics: {test_metrics}")

    # Save evaluation results
    print(f"Saving evaluation results to {args.save_path}...")
    evaluate_and_save(
        model,
        test_dataloader,
        loss_fn,
        save_path=args.save_path,
        device=device,
        prob=args.prob_threshold
    )
    print("Evaluation completed and results saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the trained GraphormerJEPA model")

    # Paths for model and data
    parser.add_argument('--data_path', type=str, default="updated_san_francisco_graph.json",
                        help="Path to the JSON file containing graph data")
    parser.add_argument('--model_path', type=str, default="model_checkpoint.pt",
                        help="Path to the trained model weights file")
    parser.add_argument('--save_path', type=str, default="test_evaluation_results.csv",
                        help="Path to save the evaluation results CSV file")

    # Model parameters
    parser.add_argument('--input_dim', type=int, default=4,
                        help="Input feature dimension")
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help="Hidden layer dimension")
    parser.add_argument('--output_dim', type=int, default=1,
                        help="Output dimension")
    parser.add_argument('--max_degree', type=int, default=128,
                        help="Maximum node degree")
    parser.add_argument('--max_nodes', type=int, default=50000,
                        help="Maximum number of nodes")
    parser.add_argument('--alpha', type=float, default=0.001,
                        help="Weight for spatial loss")

    # DataLoader parameters
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of worker threads for DataLoader")

    # Other parameters
    parser.add_argument('--num_communities', type=int, default=10,
                        help="Number of communities for graph splitting")
    parser.add_argument('--prob_threshold', type=float, default=0.5,
                        help="Probability threshold for classification")

    args = parser.parse_args()
    main(args)
