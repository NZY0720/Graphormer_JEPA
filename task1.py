# task1.py

import torch 
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

# Import the updated GraphormerJEPA model with edge attribute support
from graph_model import GraphormerJEPA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


########################################
# 1. LoRALayer
########################################
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank):
        """
        Initialize the LoRA (Low-Rank Adaptation) layer for efficient parameter tuning.

        Args:
            in_features (int): Input feature dimension.
            out_features (int): Output feature dimension.
            rank (int): Rank of the low-rank matrices.
        """
        super(LoRALayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Low-rank matrices W_A and W_B for adaptation
        self.W_A = nn.Parameter(torch.randn(in_features, rank))
        self.W_B = nn.Parameter(torch.randn(rank, out_features))

    def forward(self, x):
        """
        Forward pass for the LoRA layer.

        Args:
            x (Tensor): Input tensor, shape [*, in_features], where * denotes any number of leading dimensions.

        Returns:
            Tensor: Output tensor after applying low-rank adaptation, shape [*, out_features].
        """
        device_x = x.device  # Get the device of the input tensor
        W_A = self.W_A.to(device_x)  # Move W_A to the same device as input
        W_B = self.W_B.to(device_x)  # Move W_B to the same device as input

        # Compute the low-rank adaptation
        low_rank_out = torch.matmul(x, W_A)   # [*, rank]
        low_rank_out = torch.matmul(low_rank_out, W_B)  # [*, out_features]
        return low_rank_out  # Return the adapted output


########################################
# 2. EVCSLoadDataset
########################################
class EVCSLoadDataset(Dataset):
    """
    Custom Dataset class for loading Electric Vehicle Charging Station (EVCS) time series data and graph structures.

    Each sample consists of:
        - Input sequence: Historical time steps [T_in, N, 1].
        - Target: Future value to predict [N].
        - Adjacency matrix: Graph structure [N, N].
        - Time index: Index of the target time step.
    """
    def __init__(self, power_array, adj_array, T_in=24, step=1, mode='train',
                 train_ratio=0.8, val_ratio=0.1):
        """
        Initialize the EVCSLoadDataset.

        Args:
            power_array (np.ndarray): Power data array, shape [N, T], where N is the number of nodes and T is the number of time steps.
            adj_array (np.ndarray): Adjacency matrix array, shape [N, N].
            T_in (int): Number of historical time steps to use as input.
            step (int): Number of steps to predict into the future.
            mode (str): Dataset split mode - 'train', 'val', or 'test'.
            train_ratio (float): Proportion of data to use for training.
            val_ratio (float): Proportion of data to use for validation.
        """
        self.power = power_array  # Power data [N, T]
        self.adj = adj_array      # Adjacency matrix [N, N]
        self.N, self.T = self.power.shape  # Number of nodes, number of time steps
        self.T_in = T_in
        self.step = step

        # Determine the end indices for training and validation splits
        train_end = int(self.T * train_ratio)
        val_end = int(self.T * (train_ratio + val_ratio))

        if mode == 'train':
            start_idx = 0
            end_idx = train_end
        elif mode == 'val':
            start_idx = train_end
            end_idx = val_end
        else:  # 'test'
            start_idx = val_end
            end_idx = self.T

        # Calculate the number of possible samples in the selected split
        possible_length = end_idx - self.T_in - self.step + 1
        if possible_length <= 0:
            raise ValueError(f"No samples for mode={mode}, possible_length={possible_length}")

        # Store the valid starting time indices for samples
        self.time_index = range(start_idx, end_idx - self.T_in - self.step + 1)
        if len(self.time_index) == 0:
            raise ValueError(f"No samples for mode={mode} after indexing.")

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.time_index)

    def __getitem__(self, idx):
        """
        Retrieve a sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Tensor, Tensor, Tensor, int]: (Input sequence, Target, Adjacency matrix, Time index)
        """
        start_t = self.time_index[idx]  # Starting time step for input sequence

        # Extract input sequence: [T_in, N]
        x_seq = self.power[:, start_t : start_t + self.T_in].T.astype(np.float32)  # [T_in, N]
        x_seq = x_seq[:, :, None]  # Add feature dimension: [T_in, N, 1]

        # Extract target: [N]
        y = self.power[:, start_t + self.T_in + self.step - 1].astype(np.float32)  # [N]

        # Extract adjacency matrix: [N, N]
        adj = self.adj.astype(np.float32)  # [N, N]

        # Time index for the target
        idx_time = start_t + self.T_in + self.step - 1

        return torch.tensor(x_seq), torch.tensor(y), torch.tensor(adj), idx_time  # Return the sample


########################################
# 3. LSTMAdapter
########################################
class LSTMAdapter(nn.Module):
    """
    LSTM Adapter to encode temporal sequences into node embeddings.

    Converts input sequences of shape [T_in, N, l_f] into node embeddings [N, l].

    Args:
        N (int): Number of nodes.
        l_f (int): Input feature dimension (default: 1).
        l (int): Output embedding dimension.
        hidden_dim (int): Hidden dimension size of LSTM.
        lstm_layers (int): Number of LSTM layers.
    """
    def __init__(self, N=8, l_f=1, l=4, hidden_dim=64, lstm_layers=1):
        """
        Initialize the LSTMAdapter.

        Args:
            N (int): Number of nodes.
            l_f (int): Input feature dimension.
            l (int): Output embedding dimension.
            hidden_dim (int): LSTM hidden dimension.
            lstm_layers (int): Number of LSTM layers.
        """
        super(LSTMAdapter, self).__init__()
        self.N = N
        self.l_f = l_f
        self.l = l
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        # Define LSTM for encoding temporal sequences
        self.lstm = nn.LSTM(
            input_size=l_f, hidden_size=hidden_dim,
            num_layers=lstm_layers, batch_first=True
        )
        # Linear layer to map LSTM output to desired embedding dimension
        self.map_linear = nn.Linear(hidden_dim, l)

    def forward(self, X, W):
        """
        Forward pass for the LSTMAdapter.

        Args:
            X (Tensor): Input sequences, shape [T_in, N, l_f].
            W (Tensor): Adjacency matrix, shape [N, N].

        Returns:
            Tuple[Tensor, Tensor]: (Node embeddings Z, Adjacency matrix W)
                - Z (Tensor): Node embeddings, shape [N, l].
                - W (Tensor): Adjacency matrix, unchanged.
        """
        X = X.to(device)  # Move input to device
        W = W.to(device)  # Move adjacency matrix to device

        # Transpose X to shape [N, T_in, l_f]
        X = X.transpose(1, 0)  # [N, T_in, l_f]

        node_reps = []  # List to store node embeddings
        for i in range(self.N):
            node_x = X[i].unsqueeze(0)  # [1, T_in, l_f]
            _, (h, _) = self.lstm(node_x)  # h: [num_layers, 1, hidden_dim]
            h_final = h[-1, 0, :]  # [hidden_dim], take the last layer's hidden state
            z_i = self.map_linear(h_final)  # [l], map to embedding dimension
            node_reps.append(z_i)  # Append to list

        Z = torch.stack(node_reps, dim=0)  # [N, l]
        return Z, W  # Return node embeddings and adjacency matrix


########################################
# 4. create_dummy_batch
########################################
def create_dummy_batch(Z, W):
    """
    Package node embeddings and adjacency matrix into a PyG-like Data object.

    Args:
        Z (Tensor): Node embeddings, shape [N, l].
        W (Tensor): Adjacency matrix, shape [N, N].

    Returns:
        Data: PyG-like Data object containing node features, degrees, node IDs, and edge information.
    """
    class DummyData:
        pass

    context_batch = DummyData()
    context_batch.x = Z.unsqueeze(0)  # Add batch dimension: [1, N, l]
    context_batch.degree = torch.zeros((1, Z.size(0)), dtype=torch.long, device=device)  # Degrees [1, N]
    context_batch.node_ids = torch.arange(Z.size(0), dtype=torch.long, device=device).unsqueeze(0)  # Node IDs [1, N]

    # If edge_index and edge_attr are not used, initialize them as empty tensors
    # However, since edge_attr is considered, we need to extract edge indices from W
    edge_index = (W > 0).nonzero(as_tuple=False).t()  # [2, E], where E is number of edges
    if edge_index.size(1) > 0:
        context_batch.edge_index = edge_index.contiguous()  # [2, E]
        # Extract edge attributes based on edge indices
        # Assuming W contains raw edge attributes, here we mock edge_attr as ones
        # Replace this with actual edge_attr if available
        context_batch.edge_attr = W[edge_index[0], edge_index[1]].unsqueeze(-1)  # [E, 1]
    else:
        context_batch.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        context_batch.edge_attr = torch.empty((0, 1), dtype=torch.float, device=device)

    context_batch.batch = torch.zeros(Z.size(0), dtype=torch.long, device=device)  # Batch labels [N]
    context_batch.num_graphs = 1  # Number of graphs in the batch

    return context_batch  # Return the packaged Data object


########################################
# 5. Error Computation Functions
########################################
def compute_mae(predicted, true):
    """
    Compute Mean Absolute Error (MAE) for each sample.

    Args:
        predicted (np.ndarray): Predicted values, shape [test_samples, N].
        true (np.ndarray): True values, shape [test_samples, N].

    Returns:
        np.ndarray: MAE for each sample, shape [test_samples].
    """
    return np.mean(np.abs(predicted - true), axis=1)

def compute_rmse(predicted, true):
    """
    Compute Root Mean Square Error (RMSE) for each sample.

    Args:
        predicted (np.ndarray): Predicted values, shape [test_samples, N].
        true (np.ndarray): True values, shape [test_samples, N].

    Returns:
        np.ndarray: RMSE for each sample, shape [test_samples].
    """
    return np.sqrt(np.mean((predicted - true) ** 2, axis=1))


########################################
# 6. Main Function (Downstream Node Prediction)
########################################
def main():
    """
    Main function to perform the following:
        1. Load and preprocess data.
        2. Build datasets and data loaders.
        3. Initialize the GraphormerJEPA model and load pre-trained weights.
        4. Replace linear layers with LoRA layers.
        5. Define LSTMAdapter and optimizer.
        6. Train the model.
        7. Validate the model.
        8. Test the model and save results.
    """
    # ========== 1. Set Parameters ==========
    input_dim = 4         # Input feature dimension (consistent with graph_model.py)
    hidden_dim = 128      # Hidden layer dimension
    max_degree = 128      # Maximum node degree for embedding
    max_nodes = 50000     # Maximum number of nodes for positional embedding
    lr_finetune = 1e-3    # Learning rate for fine-tuning
    epochs = 30           # Number of training epochs
    T_in = 24             # Length of input sequence
    step = 5              # Prediction step size

    power_path = "power.csv"  # Path to power data CSV
    adj_path = "adj.csv"      # Path to adjacency matrix CSV

    # ========== 2. Load and Normalize Data ==========
    # Read power data from CSV and transpose to shape [N, T]
    power_data = pd.read_csv(power_path, header=None).values.T  # [N, T]
    adj_data = pd.read_csv(adj_path, header=None).values.astype(np.float32)  # [N, N]

    # Initialize MinMaxScaler for normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    power_data_reshaped = power_data.reshape(-1, 1)  # Reshape to [N*T, 1] for fitting
    scaler.fit(power_data_reshaped)  # Fit scaler on power data
    power_data_scaled = scaler.transform(power_data_reshaped).reshape(power_data.shape)  # [N, T]

    # ========== 3. Build Datasets and DataLoaders ==========
    # Create training, validation, and testing datasets
    train_dataset = EVCSLoadDataset(power_data_scaled, adj_data, T_in=T_in, step=step, mode='train')
    val_dataset = EVCSLoadDataset(power_data_scaled, adj_data, T_in=T_in, step=step, mode='val')
    test_dataset = EVCSLoadDataset(power_data_scaled, adj_data, T_in=T_in, step=step, mode='test')

    # Create DataLoaders with batch_size=1; shuffle training data
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ========== 4. Initialize GraphormerJEPA Model ==========
    # Instantiate the GraphormerJEPA model with edge attribute support
    model = GraphormerJEPA(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        max_degree=max_degree,
        max_nodes=max_nodes,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
        delta=1.0
    ).to(device)  # Move model to device (GPU/CPU)

    # Load pre-trained weights if available
    if os.path.exists("jepa_best_model.pt"):
        model.load_state_dict(torch.load("jepa_best_model.pt", map_location=device), strict=False)
        print("Loaded pre-trained JEPA weights from jepa_best_model.pt")
    else:
        print("Warning: jepa_best_model.pt not found. Using random initialization.")

    # ========== 5. Replace Linear Layers with LoRA Layers ==========
    # Collect all linear layers in the model for replacement
    layers_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            rank = 8  # Rank for LoRA
            layers_to_replace.append((name, in_features, out_features, rank))

    # Replace each linear layer with a LoRA layer
    for name, in_features, out_features, rank in layers_to_replace:
        lora_layer = LoRALayer(in_features, out_features, rank)  # Instantiate LoRA layer

        # Split the layer name to navigate through nested modules
        components = name.split('.')
        parent = model
        for comp in components[:-1]:
            parent = getattr(parent, comp)  # Navigate to the parent module
        setattr(parent, components[-1], lora_layer)  # Replace the linear layer with LoRA layer
        print(f"Replaced {name} with LoRALayer")

    # ========== 6. Define LSTMAdapter and Optimizer ==========
    # Initialize LSTMAdapter to encode temporal sequences into node embeddings
    lstm_adapter = LSTMAdapter(N=adj_data.shape[0], l_f=1, l=input_dim, hidden_dim=64, lstm_layers=1).to(device)
    for param in lstm_adapter.parameters():
        param.requires_grad = True  # Ensure LSTMAdapter parameters are trainable

    # Freeze all model parameters except the prediction head
    for name, param in model.named_parameters():
        if 'prediction_head' in name:
            param.requires_grad = True  # Unfreeze prediction head
        else:
            param.requires_grad = False  # Freeze all other parameters

    # Define optimizer to update only trainable parameters (LoRA layers, prediction head, LSTMAdapter)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, list(model.parameters()) + list(lstm_adapter.parameters())),
        lr=lr_finetune
    )
    scaler_grad = GradScaler()  # Initialize GradScaler for mixed precision
    loss_fn = nn.MSELoss()      # Define Mean Squared Error loss

    best_loss = float('inf')    # Initialize best loss for model checkpointing

    # ========== 7. Training Loop ==========
    for epoch in range(1, epochs + 1):
        model.train()          # Set model to training mode
        lstm_adapter.train()  # Set LSTMAdapter to training mode
        total_loss = 0.0      # Accumulate training loss
        total_samples = 0     # Count number of training samples

        # Iterate over the training DataLoader
        for X_seq, Y, W, idx_time in tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch"):
            # Data shapes:
            # X_seq: [1, T_in, N, 1] -> squeeze -> [T_in, N, 1]
            # Y: [1, N] -> squeeze -> [N]
            # W: [1, N, N] -> squeeze -> [N, N]
            X_seq = X_seq.squeeze(0).to(device)  # [T_in, N, 1]
            Y = Y.squeeze(0).to(device)          # [N]
            W = W.squeeze(0).to(device)          # [N, N]

            optimizer.zero_grad(set_to_none=True)  # Reset gradients

            # Encode the input sequence using LSTMAdapter to get node embeddings
            Z, W_adapt = lstm_adapter(X_seq, W)  # Z: [N, input_dim], W_adapt: [N, N]

            # Package node embeddings and adjacency matrix into a PyG-like Data object
            context_batch = create_dummy_batch(Z, W_adapt)  # PyG Data object

            # Forward pass through the model in downstream prediction mode
            predicted_scores = model(context_batch, context_batch, pretrain=False)  # [1, N]

            # Compute loss between predictions and true values
            loss = loss_fn(predicted_scores[0], Y)  # Scalar loss

            # Backpropagation with mixed precision
            scaler_grad.scale(loss).backward()  # Backward pass
            scaler_grad.step(optimizer)         # Update parameters
            scaler_grad.update()                # Update scaler

            total_loss += loss.item()  # Accumulate loss
            total_samples += 1         # Increment sample count

        # Calculate average training loss for the epoch
        train_loss = total_loss / total_samples

        # ========== 8. Validation ==========
        model.eval()          # Set model to evaluation mode
        lstm_adapter.eval()  # Set LSTMAdapter to evaluation mode
        val_loss = 0.0       # Accumulate validation loss
        val_samples = 0      # Count number of validation samples

        with torch.no_grad(), autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            # Iterate over the validation DataLoader
            for X_seq, Y, W, idx_time in val_loader:
                X_seq = X_seq.squeeze(0).to(device)  # [T_in, N, 1]
                Y = Y.squeeze(0).to(device)          # [N]
                W = W.squeeze(0).to(device)          # [N, N]

                # Encode the input sequence
                Z, W_adapt = lstm_adapter(X_seq, W)  # Z: [N, input_dim], W_adapt: [N, N]
                context_batch = create_dummy_batch(Z, W_adapt)  # PyG Data object

                # Forward pass in downstream prediction mode
                predicted_scores = model(context_batch, context_batch, pretrain=False)  # [1, N]
                loss_v = loss_fn(predicted_scores[0], Y)  # Scalar loss

                val_loss += loss_v.item()  # Accumulate validation loss
                val_samples += 1         # Increment validation sample count

        # Calculate average validation loss for the epoch
        val_loss /= val_samples
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # ========== 9. Save Best Model ==========
        # Check if current epoch has the best loss
        current_total_loss = train_loss + val_loss
        if current_total_loss < best_loss:
            best_loss = current_total_loss  # Update best loss
            torch.save(model.state_dict(), "finetuned_model_with_lora_1.pt")  # Save model weights
            torch.save(lstm_adapter.state_dict(), "lstm_adapter_1.pt")        # Save LSTMAdapter weights
            print(f"Saved best model at epoch {epoch} with Val Loss={val_loss:.4f}")

    # ========== 10. Testing Phase + Denormalization ==========
    # Load the best model and adapter weights if available
    if os.path.exists("finetuned_model_with_lora_1.pt"):
        model.load_state_dict(torch.load("finetuned_model_with_lora_1.pt", map_location=device), strict=False)
    if os.path.exists("lstm_adapter_1.pt"):
        lstm_adapter.load_state_dict(torch.load("lstm_adapter_1.pt", map_location=device))

    model.eval()          # Set model to evaluation mode
    lstm_adapter.eval()  # Set LSTMAdapter to evaluation mode

    pred_list = []  # List to store predictions
    true_list = []  # List to store true values
    idx_list = []   # List to store time indices

    with torch.no_grad(), autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
        # Iterate over the test DataLoader
        for X_seq, Y, W, idx_time in test_loader:
            X_seq = X_seq.squeeze(0).to(device)  # [T_in, N, 1]
            Y = Y.squeeze(0).to(device)          # [N]
            W = W.squeeze(0).to(device)          # [N, N]

            # Encode the input sequence
            Z, W_adapt = lstm_adapter(X_seq, W)  # Z: [N, input_dim], W_adapt: [N, N]
            context_batch = create_dummy_batch(Z, W_adapt)  # PyG Data object

            # Forward pass in downstream prediction mode
            predicted_scores = model(context_batch, context_batch, pretrain=False)  # [1, N]
            pred_np = predicted_scores[0].cpu().numpy()  # Convert to NumPy array [N]
            true_np = Y.cpu().numpy()                    # Convert to NumPy array [N]

            pred_list.append(pred_np)  # Append prediction
            true_list.append(true_np)  # Append true value
            idx_list.append(idx_time.item())  # Append time index

    # Convert lists to NumPy arrays
    pred_array = np.array(pred_list)  # [test_samples, N]
    true_array = np.array(true_list)  # [test_samples, N]
    idx_array = np.array(idx_list)    # [test_samples]

    # ========== 11. Denormalize Predictions and True Values ==========
    # Reshape arrays for denormalization
    pred_array_reshaped = pred_array.reshape(-1, 1)  # [test_samples * N, 1]
    true_array_reshaped = true_array.reshape(-1, 1)  # [test_samples * N, 1]

    # Apply inverse transformation to get original scale
    pred_array_denorm = scaler.inverse_transform(pred_array_reshaped).reshape(pred_array.shape)  # [test_samples, N]
    true_array_denorm = scaler.inverse_transform(true_array_reshaped).reshape(true_array.shape)  # [test_samples, N]

    # ========== 12. Compute Errors ==========
    mae = compute_mae(pred_array_denorm, true_array_denorm)  # [test_samples]
    rmse = compute_rmse(pred_array_denorm, true_array_denorm)  # [test_samples]
    mae_total = np.mean(mae)  # Average MAE across all samples
    rmse_total = np.mean(rmse)  # Average RMSE across all samples
    print(f"Total MAE (denormalized): {mae_total:.4f}")
    print(f"Total RMSE (denormalized): {rmse_total:.4f}")

    # ========== 13. Save Predictions to CSV ==========
    data = {'Time_Index': idx_array}  # Initialize dictionary with time indices
    N = true_array_denorm.shape[1]    # Number of nodes
    for node_id in range(N):
        data[f'Node_{node_id}_Predicted'] = pred_array_denorm[:, node_id]  # Predicted values per node
        data[f'Node_{node_id}_True'] = true_array_denorm[:, node_id]        # True values per node
    results_df = pd.DataFrame(data)  # Create DataFrame from dictionary
    results_df.to_csv("test_predictions_1.csv", index=False)  # Save to CSV
    print("Test set predictions saved to 'test_predictions_1.csv'.")

    # ========== 14. Visualization ==========
    plt.figure(figsize=(20, 4 * N))  # Set figure size
    for node_id in range(N):
        plt.subplot(N, 1, node_id + 1)  # Create subplot for each node
        plt.plot(idx_array, true_array_denorm[:, node_id], label="True", color="blue")  # Plot true values
        plt.plot(idx_array, pred_array_denorm[:, node_id], label="Pred", color="red")   # Plot predicted values
        plt.legend()  # Add legend
        plt.xlabel('Time Step')  # X-axis label
        plt.ylabel('Value')       # Y-axis label
    plt.tight_layout()  # Adjust subplot parameters for a clean layout
    plt.savefig("task1_all_nodes_1.png", dpi=600)  # Save the figure as a high-resolution PNG
    plt.show()  # Display the figure


if __name__ == '__main__':
    main()
