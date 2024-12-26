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

from graph_model import GraphormerJEPA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


########################################
# 1. LoRALayer
########################################
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LoRALayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.W_A = nn.Parameter(torch.randn(in_features, rank))
        self.W_B = nn.Parameter(torch.randn(rank, out_features))

    def forward(self, x):
        """
        parameters:
            x (Tensor):  [*, in_features], * is any previous dimension

        return:
            Tensor:  [*, out_features]
        """
        device_x = x.device
        W_A = self.W_A.to(device_x)
        W_B = self.W_B.to(device_x)

        low_rank_out = torch.matmul(x, W_A)   
        low_rank_out = torch.matmul(low_rank_out, W_B)
        return low_rank_out


########################################
# 2. EVCSLoadDataset
########################################
class EVCSLoadDataset(Dataset):
    """
    read: power_array (N, T) + adj_array (N, N)
    T_in: historical time step
    step: prediction time step
    mode: 'train', 'val', 'test'
    """
    def __init__(self, power_array, adj_array, T_in=24, step=1, mode='train',
                 train_ratio=0.8, val_ratio=0.1):
        self.power = power_array  # (N, T)
        self.adj = adj_array      # (N, N)
        self.N, self.T = self.power.shape
        self.T_in = T_in
        self.step = step

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

        possible_length = end_idx - self.T_in - self.step + 1
        if possible_length <= 0:
            raise ValueError(f"No samples for mode={mode}, possible_length={possible_length}")

        self.time_index = range(start_idx, end_idx - self.T_in - self.step + 1)
        if len(self.time_index) == 0:
            raise ValueError(f"No samples for mode={mode} after indexing.")

    def __len__(self):
        return len(self.time_index)

    def __getitem__(self, idx):
        start_t = self.time_index[idx]
        # x_seq: shape (T_in, N)
        x_seq = self.power[:, start_t : start_t + self.T_in].T.astype(np.float32)
        # => (T_in, N, 1)
        x_seq = x_seq[:, :, None]

        # prediction => (N,)
        y = self.power[:, start_t + self.T_in + self.step - 1].astype(np.float32)

        # adjacency => (N, N)
        adj = self.adj.astype(np.float32)

        idx_time = start_t + self.T_in + self.step - 1
        return torch.tensor(x_seq), torch.tensor(y), torch.tensor(adj), idx_time


########################################
# 3. LSTMAdapter
########################################
class LSTMAdapter(nn.Module):
    """

        input:
            X (Tensor):  [T_in, N, l_f]
            W (Tensor): adj matrix [N, N]

        output:
            Tuple[Tensor, Tensor]: (Z, W)
                - Z (Tensor): node embedding, [N, l]
                - W (Tensor): [N, N]
    """
    def __init__(self, N=8, l_f=1, l=4, hidden_dim=64, lstm_layers=1):
        super(LSTMAdapter, self).__init__()
        self.N = N
        self.l_f = l_f
        self.l = l
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        self.lstm = nn.LSTM(
            input_size=l_f, hidden_size=hidden_dim,
            num_layers=lstm_layers, batch_first=True
        )
        self.map_linear = nn.Linear(hidden_dim, l)

    def forward(self, X, W):
        X = X.to(device)
        W = W.to(device)
        X = X.transpose(1, 0)

        node_reps = []
        for i in range(self.N):
            node_x = X[i].unsqueeze(0)  # (1, T_in, l_f)
            _, (h, _) = self.lstm(node_x)
            h_final = h[-1, 0, :]  # (hidden_dim,)
            z_i = self.map_linear(h_final)  # (l,)
            node_reps.append(z_i)

        Z = torch.stack(node_reps, dim=0)  # => (N, l)
        return Z, W


########################################
# 4. create_dummy_batch
########################################
def create_dummy_batch(Z, W):
    """
    Pack (Z,W) into a PyG data
    
    return:
        Data: PyG-like Data, including: x, degree, node_ids, edge_index, edge_attr
    """
    class DummyData:
        pass

    context_batch = DummyData()
    context_batch.x = Z.unsqueeze(0)  # => (1, N, l)
    context_batch.degree = torch.zeros((1, Z.size(0)), dtype=torch.long, device=device)
    context_batch.node_ids = torch.arange(Z.size(0), dtype=torch.long, device=device).unsqueeze(0)
    # empty
    context_batch.edge_attr = None
    context_batch.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    context_batch.batch = torch.zeros(Z.size(0), dtype=torch.long, device=device)
    context_batch.num_graphs = 1
    return context_batch


########################################
# 5. evaluation functions
########################################
def compute_mae(predicted, true):
    """
    predicted, true shape: (batch_size, N)
    """
    return np.mean(np.abs(predicted - true), axis=1)

def compute_rmse(predicted, true):
    """
    predicted, true shape: (batch_size, N)
    """
    return np.sqrt(np.mean((predicted - true) ** 2, axis=1))


########################################
# 6. main
########################################
def main():
    input_dim = 4         # algin with pretrain
    hidden_dim = 32
    max_degree = 128
    max_nodes = 50000
    lr_finetune = 1e-3
    epochs = 30
    T_in = 24
    step = 5

    power_path = "power.csv"
    adj_path = "adj.csv"

    # ============ 1. Load and Normalization ============
    power_data = pd.read_csv(power_path, header=None).values.T  # => (N, T)
    adj_data = pd.read_csv(adj_path, header=None).values.astype(np.float32)

    scaler = MinMaxScaler(feature_range=(0, 1))
    power_data_reshaped = power_data.reshape(-1, 1)
    scaler.fit(power_data_reshaped)
    power_data_scaled = scaler.transform(power_data_reshaped).reshape(power_data.shape)

    # ============ 2. Dataset creation and DataLoader ============
    train_dataset = EVCSLoadDataset(power_data_scaled, adj_data, T_in=T_in, step=step, mode='train')
    val_dataset = EVCSLoadDataset(power_data_scaled, adj_data, T_in=T_in, step=step, mode='val')
    test_dataset = EVCSLoadDataset(power_data_scaled, adj_data, T_in=T_in, step=step, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ============ 3. Initialization ============
    model = GraphormerJEPA(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        max_degree=max_degree,
        max_nodes=max_nodes,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
        delta=1.0
    ).to(device)

    # Load Pre-trained parameters
    if os.path.exists("jepa_best_model.pt"):
        model.load_state_dict(torch.load("jepa_best_model.pt", map_location=device), strict=False)
        print("Loaded pre-trained JEPA weights from jepa_best_model.pt")
    else:
        print("No pre-trained JEPA weights found. Training from random init...")

    # ============ 4. Replace linear layer with LoRA ============
    layers_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            rank = 8  
            layers_to_replace.append((name, in_features, out_features, rank))
    # Replace linear layer with LoRA
    for name, in_features, out_features, rank in layers_to_replace:
        lora_layer = LoRALayer(in_features, out_features, rank) 
        components = name.split('.')
        parent = model
        for comp in components[:-1]:
            parent = getattr(parent, comp)
        setattr(parent, components[-1], lora_layer)
        print(f"Replaced {name} with LoRALayer")

    # ============ 5. Define LSTMAdapter + Optimizer ============
    # LSTMAdapter:  (T_in, N, 1) -> (N, input_dim=4)
    lstm_adapter = LSTMAdapter(N=adj_data.shape[0], l_f=1, l=input_dim, hidden_dim=64, lstm_layers=1).to(device)
    for param in lstm_adapter.parameters():
        param.requires_grad = True

    for name, param in model.named_parameters():
        if 'prediction_head' in name:
            param.requires_grad = True  # Unfreeze prediction_head
        else:
            param.requires_grad = False  # Freeze other layers

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, list(model.parameters()) + list(lstm_adapter.parameters())),
        lr=lr_finetune
    )
    scaler_grad = GradScaler()
    loss_fn = nn.MSELoss()

    best_loss = float('inf')

    # ============ 6. Downstream Epochs ============
    for epoch in range(1, epochs + 1):
        model.train()
        lstm_adapter.train()
        total_loss = 0.0
        total_samples = 0

        for X_seq, Y, W, idx_time in tqdm(train_loader, desc=f"Epoch {epoch}"):
            # X_seq: [1, T_in, N, 1] -> squeeze -> [T_in, N, 1]
            # Y: [1, N] -> squeeze -> [N]
            # W: [1, N, N] -> squeeze -> [N, N]
            X_seq = X_seq.squeeze(0).to(device)  # => (T_in, N, 1)
            Y = Y.squeeze(0).to(device)          # => (N,)
            W = W.squeeze(0).to(device)          # => (N, N)

            optimizer.zero_grad(set_to_none=True)
            Z, W_adapt = lstm_adapter(X_seq, W)  # Z: [N, input_dim], W_adapt: [N, N]
            context_batch = create_dummy_batch(Z, W_adapt)

            # pretrain=False => (B, N)
            predicted_scores = model(context_batch, context_batch, pretrain=False)
            # predicted_scores shape => [1, N] and Y shape => [N]
            # B=1 => predicted_scores[0] => (N,)
            loss = loss_fn(predicted_scores[0], Y)

            scaler_grad.scale(loss).backward()
            scaler_grad.step(optimizer)
            scaler_grad.update()

            total_loss += loss.item()
            total_samples += 1

        train_loss = total_loss / total_samples

        # Varify
        model.eval()
        lstm_adapter.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad(), autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            for X_seq, Y, W, idx_time in val_loader:
                X_seq = X_seq.squeeze(0).to(device)
                Y = Y.squeeze(0).to(device)
                W = W.squeeze(0).to(device)

                Z, W_adapt = lstm_adapter(X_seq, W)
                context_batch = create_dummy_batch(Z, W_adapt)

                predicted_scores = model(context_batch, context_batch, pretrain=False)
                loss_v = loss_fn(predicted_scores[0], Y)
                val_loss += loss_v.item()
                val_samples += 1

        val_loss /= val_samples
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        if (train_loss + val_loss) < best_loss:
            best_loss = train_loss + val_loss
            torch.save(model.state_dict(), "finetuned_model_with_lora_1.pt")
            torch.save(lstm_adapter.state_dict(), "lstm_adapter_1.pt")
            print(f"Saved best model at epoch {epoch} with Val Loss={val_loss:.4f}")

    # ============ 7. Test =============
    if os.path.exists("finetuned_model_with_lora_1.pt"):
        model.load_state_dict(torch.load("finetuned_model_with_lora_1.pt", map_location=device), strict=False)
    if os.path.exists("lstm_adapter_1.pt"):
        lstm_adapter.load_state_dict(torch.load("lstm_adapter_1.pt", map_location=device))

    model.eval()
    lstm_adapter.eval()

    pred_list = []
    true_list = []
    idx_list = []

    with torch.no_grad(), autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
        for X_seq, Y, W, idx_time in test_loader:
            X_seq = X_seq.squeeze(0).to(device)
            Y = Y.squeeze(0).to(device)
            W = W.squeeze(0).to(device)

            Z, W_adapt = lstm_adapter(X_seq, W)
            context_batch = create_dummy_batch(Z, W_adapt)

            predicted_scores = model(context_batch, context_batch, pretrain=False)  # => [1, N]
            pred_np = predicted_scores[0].cpu().numpy()  # => (N,)
            true_np = Y.cpu().numpy()                    # => (N,)

            pred_list.append(pred_np)
            true_list.append(true_np)
            idx_list.append(idx_time.item())

    pred_array = np.array(pred_list)  # => (test_samples, N)
    true_array = np.array(true_list)  # => (test_samples, N)
    idx_array = np.array(idx_list)    # => (test_samples,)

    # denormalization
    pred_array_reshaped = pred_array.reshape(-1, 1)
    true_array_reshaped = true_array.reshape(-1, 1)

    pred_array_denorm = scaler.inverse_transform(pred_array_reshaped).reshape(pred_array.shape)
    true_array_denorm = scaler.inverse_transform(true_array_reshaped).reshape(true_array.shape)

    # calculate MAE, RMSE
    mae = compute_mae(pred_array_denorm, true_array_denorm)
    rmse = compute_rmse(pred_array_denorm, true_array_denorm)
    mae_total = np.mean(mae)
    rmse_total = np.mean(rmse)
    print(f"Total MAE (denormalized): {mae_total:.4f}")
    print(f"Total RMSE (denormalized): {rmse_total:.4f}")


    data = {'Time_Index': idx_array}
    N = true_array_denorm.shape[1]
    for node_id in range(N):
        data[f'Node_{node_id}_Predicted'] = pred_array_denorm[:, node_id]
        data[f'Node_{node_id}_True'] = true_array_denorm[:, node_id]
    results_df = pd.DataFrame(data)
    results_df.to_csv("test_predictions_1.csv", index=False)
    print("Results on test saved to 'test_predictions_1.csv'.")

    # visualization
    plt.figure(figsize=(20, 4 * N))
    for node_id in range(N):
        plt.subplot(N, 1, node_id + 1)
        plt.plot(idx_array, true_array_denorm[:, node_id], label="True", color="blue")
        plt.plot(idx_array, pred_array_denorm[:, node_id], label="Pred", color="red")
        plt.legend()
        plt.xlabel('Time Step')
        plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig("task1_all_nodes_1.png", dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
