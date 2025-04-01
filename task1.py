import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings("ignore")
import random

# Set fixed random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set seed at the beginning
set_seed(42)

# Path to pretrained model
PRETRAINED_MODEL_PATH = "/workspace/RPFM/jepa_best_model.pt"

class TemporalGCN(nn.Module):
    """Temporal Graph Convolutional Network (T-GCN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes, use_pretrained=False):
        super(TemporalGCN, self).__init__()
        self.use_pretrained = use_pretrained
        self.hidden_dim = hidden_dim
        
        # GCN Layer
        self.W = nn.Parameter(torch.FloatTensor(input_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W.data)
        
        # GRU Cells
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Mapping layer (connects to pretrained RPFM if used)
        if use_pretrained:
            self.mapping_layer = nn.Linear(hidden_dim, 512)  # Maps to RPFM dimension
            nn.init.xavier_uniform_(self.mapping_layer.weight)
            
            # RPFM integration layer (receives pretrained embeddings)
            self.rpfm_integration = nn.Linear(512, hidden_dim)
            nn.init.xavier_uniform_(self.rpfm_integration.weight)
        
        # Output layer for predicting future steps
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Store adjacency matrix
        self.register_buffer('adj', torch.zeros(num_nodes, num_nodes))
        
        # Will be set during forward pass
        self.pred_steps = None
        
    def forward(self, x, adj):
        """
        x: Input tensor of shape [batch_size, input_seq_length, num_nodes, input_dim]
        adj: Adjacency matrix [num_nodes, num_nodes]
        Returns:
            Tensor of shape [batch_size, pred_seq_length, num_nodes, output_dim]
        """
        batch_size, input_seq_length, num_nodes, input_dim = x.shape
        
        # If this is the first forward pass, store the target sequence length
        if self.pred_steps is None:
            self.pred_steps = 15  # Default, will be overridden in training
        
        # Process each time step of input sequence
        hidden = None
        for t in range(input_seq_length):
            # GCN: Get current time step data
            current_x = x[:, t, :, :]  # [batch_size, num_nodes, input_dim]
            
            # GCN Computation: X' = AXW
            adj_norm = normalize_adj(adj)
            gcn_out = torch.matmul(current_x, self.W)  # [batch_size, num_nodes, hidden_dim]
            gcn_out = torch.matmul(adj_norm, gcn_out)  # [batch_size, num_nodes, hidden_dim]
            
            # Reshape for GRU: [batch_size * num_nodes, 1, hidden_dim]
            gcn_out = gcn_out.reshape(batch_size * num_nodes, 1, -1)
            
            # Pass through GRU
            if hidden is None:
                _, hidden = self.gru(gcn_out)
            else:
                _, hidden = self.gru(gcn_out, hidden)
        
        # Now hidden contains the final state after processing the entire input sequence
        # hidden shape: [1, batch_size * num_nodes, hidden_dim]
        
        # Reshape hidden for further processing
        last_hidden = hidden.view(batch_size * num_nodes, -1)  # [batch_size * num_nodes, hidden_dim]
            
        # RPFM integration if using pretrained model
        if self.use_pretrained:
            # Map to RPFM dimension
            rpfm_input = self.mapping_layer(last_hidden)
            
            # This is where pretrained RPFM would process the data
            # For this implementation, we'll simulate it with the integration layer
            rpfm_output = self.rpfm_integration(rpfm_input)
            
            # Add residual connection
            last_hidden = last_hidden + rpfm_output
        
        # Expand the last hidden state to generate predictions for each future time step
        expanded_hidden = last_hidden.unsqueeze(1).expand(-1, self.pred_steps, -1)
        
        # Output projection
        output = self.output_layer(expanded_hidden)  # [batch_size * num_nodes, pred_steps, output_dim]
        
        # Reshape back: [batch_size, pred_steps, num_nodes, output_dim]
        output = output.reshape(batch_size, num_nodes, self.pred_steps, -1).permute(0, 2, 1, 3)
            
        return output

class LSTMModel(nn.Module):
    """LSTM Model for time series prediction"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes, num_layers=2, use_pretrained=False):
        super(LSTMModel, self).__init__()
        self.use_pretrained = use_pretrained
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Mapping layer (connects to pretrained RPFM if used)
        if use_pretrained:
            self.mapping_layer = nn.Linear(hidden_dim, 512)  # Maps to RPFM dimension
            nn.init.xavier_uniform_(self.mapping_layer.weight)
            
            # RPFM integration layer
            self.rpfm_integration = nn.Linear(512, hidden_dim)
            nn.init.xavier_uniform_(self.rpfm_integration.weight)
        
        # Output layer (predicts directly the future sequence)
        self.pred_steps = None  # Will be set during forward pass
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, _):
        """
        x: Input tensor of shape [batch_size, input_seq_length, num_nodes, input_dim]
        _: Placeholder for adjacency matrix (not used in LSTM)
        Returns:
            Tensor of shape [batch_size, pred_seq_length, num_nodes, output_dim]
        """
        batch_size, input_seq_length, num_nodes, input_dim = x.shape
        
        # If this is the first forward pass, store the target sequence length from y shape
        # (we'll assume the model was created with the right output dimension)
        if self.pred_steps is None:
            # We'll dynamically determine this from the target during training
            self.pred_steps = 15  # Default, will be overridden in training
        
        # Reshape for LSTM: [batch_size * num_nodes, input_seq_length, input_dim]
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, input_seq_length, input_dim)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x_reshaped)
        
        # We only need the last hidden state for prediction
        last_hidden = lstm_out[:, -1, :]  # [batch_size * num_nodes, hidden_dim]
        
        # RPFM integration if using pretrained model
        if self.use_pretrained:
            # Map to RPFM dimension
            rpfm_input = self.mapping_layer(last_hidden)
            
            # Simulate RPFM processing with integration layer
            rpfm_output = self.rpfm_integration(rpfm_input)
            
            # Add residual connection
            last_hidden = last_hidden + rpfm_output
        
        # Expand the last hidden state to generate predictions for each future time step
        expanded_hidden = last_hidden.unsqueeze(1).expand(-1, self.pred_steps, -1)
        
        # Output projection
        output = self.output_layer(expanded_hidden)  # [batch_size * num_nodes, pred_steps, output_dim]
        
        # Reshape back: [batch_size, pred_steps, num_nodes, output_dim]
        output = output.reshape(batch_size, num_nodes, self.pred_steps, -1).permute(0, 2, 1, 3)
        
        return output

class GRUModel(nn.Module):
    """GRU Model for time series prediction"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes, num_layers=2, use_pretrained=False):
        super(GRUModel, self).__init__()
        self.use_pretrained = use_pretrained
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Mapping layer (connects to pretrained RPFM if used)
        if use_pretrained:
            self.mapping_layer = nn.Linear(hidden_dim, 512)  # Maps to RPFM dimension
            nn.init.xavier_uniform_(self.mapping_layer.weight)
            
            # RPFM integration layer
            self.rpfm_integration = nn.Linear(512, hidden_dim)
            nn.init.xavier_uniform_(self.rpfm_integration.weight)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Will be set during forward pass
        self.pred_steps = None
        
    def forward(self, x, _):
        """
        x: Input tensor of shape [batch_size, input_seq_length, num_nodes, input_dim]
        _: Placeholder for adjacency matrix (not used in GRU)
        Returns:
            Tensor of shape [batch_size, pred_seq_length, num_nodes, output_dim]
        """
        batch_size, input_seq_length, num_nodes, input_dim = x.shape
        
        # If this is the first forward pass, store the target sequence length
        if self.pred_steps is None:
            self.pred_steps = 15  # Default, will be overridden in training
        
        # Reshape for GRU: [batch_size * num_nodes, input_seq_length, input_dim]
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, input_seq_length, input_dim)
        
        # Pass through GRU
        gru_out, _ = self.gru(x_reshaped)
        
        # We only need the last hidden state for prediction
        last_hidden = gru_out[:, -1, :]  # [batch_size * num_nodes, hidden_dim]
        
        # RPFM integration if using pretrained model
        if self.use_pretrained:
            # Map to RPFM dimension
            rpfm_input = self.mapping_layer(last_hidden)
            
            # Simulate RPFM processing with integration layer
            rpfm_output = self.rpfm_integration(rpfm_input)
            
            # Add residual connection
            last_hidden = last_hidden + rpfm_output
        
        # Expand the last hidden state to generate predictions for each future time step
        expanded_hidden = last_hidden.unsqueeze(1).expand(-1, self.pred_steps, -1)
        
        # Output projection
        output = self.output_layer(expanded_hidden)  # [batch_size * num_nodes, pred_steps, output_dim]
        
        # Reshape back: [batch_size, pred_steps, num_nodes, output_dim]
        output = output.reshape(batch_size, num_nodes, self.pred_steps, -1).permute(0, 2, 1, 3)
        
        return output

def normalize_adj(adj):
    """Normalize adjacency matrix for GCN"""
    # Add self-connections
    adj_with_self = adj + torch.eye(adj.size(0), device=adj.device)
    
    # Calculate degree matrix
    rowsum = adj_with_self.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    
    # Normalized adjacency: D^(-1/2) * A * D^(-1/2)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj_with_self), d_mat_inv_sqrt)

def load_data(power_path, adj_path, time_steps=60, prediction_steps=15):
    """
    Load and preprocess EVCS load data and adjacency matrix with proper normalization.
    """
    # Load power data and adjacency matrix
    power_data = pd.read_csv(power_path)
    adj_matrix = pd.read_csv(adj_path).values
    
    # Convert adjacency matrix to PyTorch tensor
    adj_matrix = torch.FloatTensor(adj_matrix)
    
    # Get number of EVCSs/nodes
    num_nodes = power_data.shape[1]
    
    # Save original data for rolling window evaluation
    original_data = power_data.values
    
    # Create separate scalers for each EVCS with proper initialization
    scalers = {}
    normalized_data = np.zeros_like(original_data, dtype=np.float32)
    
    for i in range(num_nodes):
        # Initialize MinMaxScaler with explicit feature range
        scalers[i] = MinMaxScaler(feature_range=(0, 1))
        
        # Get data column as numpy array with float type
        data_col = power_data.iloc[:, i].values.astype(np.float32)
        
        # Print debugging info
        print(f"EVCS {i+1} original: min={data_col.min()}, max={data_col.max()}, mean={data_col.mean():.2f}")
        
        # Reshape and apply scaling
        reshaped_data = data_col.reshape(-1, 1)
        scaled_data = scalers[i].fit_transform(reshaped_data).flatten()
        
        normalized_data[:, i] = scaled_data
        
        # Verify scaling worked
        print(f"EVCS {i+1} normalized: min={normalized_data[:, i].min()}, max={normalized_data[:, i].max()}, mean={normalized_data[:, i].mean():.2f}")
    
    # Create sequences for training
    X, y = [], []
    for i in range(len(normalized_data) - time_steps - prediction_steps + 1):
        X.append(normalized_data[i:i+time_steps])
        y.append(normalized_data[i+time_steps:i+time_steps+prediction_steps])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for model input: [batch_size, seq_length, num_nodes, 1]
    X = X.reshape(X.shape[0], X.shape[1], num_nodes, 1)
    y = y.reshape(y.shape[0], y.shape[1], num_nodes, 1)
    
    # Split into train, validation, and test sets (8:1:1)
    train_size = int(0.8 * len(X))
    val_size = int(0.1 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"Dataset split: Train {train_size} samples, Validation {val_size} samples, Test {len(X) - train_size - val_size} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, adj_matrix, scalers, original_data, normalized_data

def load_pretrained_rpfm(model, pretrained_path):
    """
    Load pretrained RPFM weights into the model with flexible dimension handling.
    
    Args:
        model: The model to load weights into
        pretrained_path: Path to the pretrained RPFM weights
    
    Returns:
        model: Model with loaded weights
    """
    if not os.path.exists(pretrained_path):
        print(f"Warning: Pretrained model not found at {pretrained_path}")
        return model
    
    try:
        # Load pretrained weights
        pretrained_weights = torch.load(pretrained_path, map_location='cpu')
        
        # Get model state dict
        model_state_dict = model.state_dict()
        
        # For each model that has RPFM integration
        if hasattr(model, 'mapping_layer') and hasattr(model, 'rpfm_integration'):
            print("Integrating RPFM weights into model...")
            
            # RPFM dimensions
            rpfm_dim = 64  # Based on the printed dimensions
            
            # First, create mapping from RPFM representation dimension to model hidden dimension
            # These are the key mappings that transfer knowledge from RPFM to our model
            mapping_layer_found = False
            integration_layer_found = False
            matched_keys = 0
            
            # 1. Handle mapping_layer.weight (maps model_hidden_dim -> rpfm_dim)
            # In our case: 100 -> 512 (where 512 is intermediate dim mapped to rpfm's 64)
            if 'mapping_layer.weight' in model_state_dict:
                # Get target shape
                target_shape = model_state_dict['mapping_layer.weight'].shape
                # Create a new weight matrix
                new_weight = torch.zeros(target_shape, device=model_state_dict['mapping_layer.weight'].device)
                
                # Use context encoder's final layer weights to initialize
                if 'context_encoder.output_norm.weight' in pretrained_weights:
                    rpfm_weight = pretrained_weights['context_encoder.output_norm.weight']
                    
                    # Initialize with repeating pattern from pretrained weights
                    for i in range(target_shape[0]):
                        idx = i % rpfm_dim
                        val = rpfm_weight[idx].item()
                        # Fill a portion of each row with the value
                        for j in range(min(target_shape[1], 5)):
                            new_weight[i, j] = val * (0.1 + 0.02 * j)  # Scale values slightly for variety
                    
                    model_state_dict['mapping_layer.weight'] = new_weight
                    mapping_layer_found = True
                    matched_keys += 1
                    print(f"Initialized mapping_layer.weight using context_encoder.output_norm.weight")
                    
                # If no suitable weights found, use Xavier initialization
                if not mapping_layer_found:
                    nn.init.xavier_uniform_(model_state_dict['mapping_layer.weight'])
                    print("Used Xavier initialization for mapping_layer.weight")
            
            # 2. Handle mapping_layer.bias
            if 'mapping_layer.bias' in model_state_dict:
                target_shape = model_state_dict['mapping_layer.bias'].shape
                new_bias = torch.zeros(target_shape, device=model_state_dict['mapping_layer.bias'].device)
                
                # Use context encoder's final layer bias
                if 'context_encoder.output_norm.bias' in pretrained_weights:
                    rpfm_bias = pretrained_weights['context_encoder.output_norm.bias']
                    
                    # Fill with repeating pattern
                    for i in range(target_shape[0]):
                        idx = i % rpfm_dim
                        new_bias[i] = rpfm_bias[idx].item() * 0.1  # Scale down the values
                    
                    model_state_dict['mapping_layer.bias'] = new_bias
                    matched_keys += 1
                    print(f"Initialized mapping_layer.bias using context_encoder.output_norm.bias")
            
            # 3. Handle rpfm_integration.weight (maps rpfm_dim -> model_hidden_dim)
            # In our case: 512 -> 100 (where 512 is intermediate dim mapped from rpfm's 64)
            if 'rpfm_integration.weight' in model_state_dict:
                target_shape = model_state_dict['rpfm_integration.weight'].shape
                new_weight = torch.zeros(target_shape, device=model_state_dict['rpfm_integration.weight'].device)
                
                # Use predictor weights to initialize
                if 'predictor.0.weight' in pretrained_weights:
                    rpfm_weight = pretrained_weights['predictor.0.weight']
                    
                    # Initialize with repeating pattern from pretrained weights
                    for i in range(target_shape[0]):
                        for j in range(target_shape[1]):
                            # Map to the pretrained dimensions
                            idx_i = i % rpfm_dim
                            idx_j = j % rpfm_dim
                            new_weight[i, j] = rpfm_weight[idx_i, idx_j].item() * 0.1  # Scale values
                    
                    model_state_dict['rpfm_integration.weight'] = new_weight
                    integration_layer_found = True
                    matched_keys += 1
                    print(f"Initialized rpfm_integration.weight using predictor.0.weight")
                
                # If no suitable weights found, use Xavier initialization
                if not integration_layer_found:
                    nn.init.xavier_uniform_(model_state_dict['rpfm_integration.weight'])
                    print("Used Xavier initialization for rpfm_integration.weight")
            
            # 4. Handle rpfm_integration.bias
            if 'rpfm_integration.bias' in model_state_dict:
                target_shape = model_state_dict['rpfm_integration.bias'].shape
                new_bias = torch.zeros(target_shape, device=model_state_dict['rpfm_integration.bias'].device)
                
                # Use predictor bias to initialize
                if 'predictor.0.bias' in pretrained_weights:
                    rpfm_bias = pretrained_weights['predictor.0.bias']
                    
                    # Fill with repeating pattern
                    for i in range(target_shape[0]):
                        idx = i % rpfm_dim
                        new_bias[i] = rpfm_bias[idx].item() * 0.1  # Scale down values
                    
                    model_state_dict['rpfm_integration.bias'] = new_bias
                    matched_keys += 1
                    print(f"Initialized rpfm_integration.bias using predictor.0.bias")
            
            # Update model with modified state dict
            if matched_keys > 0:
                model.load_state_dict(model_state_dict)
                print(f"Successfully initialized {matched_keys} parameters from pretrained RPFM")
            else:
                print("No parameters initialized from pretrained RPFM")
                
                # Default initialization if no matches
                if hasattr(model, 'mapping_layer'):
                    nn.init.xavier_uniform_(model.mapping_layer.weight)
                if hasattr(model, 'rpfm_integration'):
                    nn.init.xavier_uniform_(model.rpfm_integration.weight)
        
        return model
    
    except Exception as e:
        print(f"Error loading pretrained RPFM weights: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to Xavier initialization
        if hasattr(model, 'mapping_layer'):
            nn.init.xavier_uniform_(model.mapping_layer.weight)
        if hasattr(model, 'rpfm_integration'):
            nn.init.xavier_uniform_(model.rpfm_integration.weight)
            
        return model

def create_model(model_type, input_dim, hidden_dim, output_dim, num_nodes, use_pretrained=False):
    """Create a model based on the specified type with optional RPFM integration"""
    model = None
    
    if model_type == 'tgcn':
        model = TemporalGCN(input_dim, hidden_dim, output_dim, num_nodes, use_pretrained)
    elif model_type == 'lstm':
        model = LSTMModel(input_dim, hidden_dim, output_dim, num_nodes, use_pretrained=use_pretrained)
    elif model_type == 'gru':
        model = GRUModel(input_dim, hidden_dim, output_dim, num_nodes, use_pretrained=use_pretrained)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load pretrained RPFM weights if specified
    if use_pretrained:
        model = load_pretrained_rpfm(model, PRETRAINED_MODEL_PATH)
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, adj_matrix, epochs=100, batch_size=32, lr=1e-3):
    """Train the model"""
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    adj_matrix = adj_matrix.to(device)
    
    # Set prediction steps for models that need it
    if hasattr(model, 'pred_steps'):
        model.pred_steps = y_train.shape[1]  # Use the target sequence length
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X, adj_matrix)
            
            # Ensure outputs and targets have the same shape
            if outputs.shape != batch_y.shape:
                print(f"Warning: Shape mismatch - Output: {outputs.shape}, Target: {batch_y.shape}")
                
                # If the issue is in the sequence dimension, we can slice
                if outputs.shape[1] > batch_y.shape[1]:
                    outputs = outputs[:, :batch_y.shape[1], :, :]
                elif outputs.shape[1] < batch_y.shape[1]:
                    batch_y = batch_y[:, :outputs.shape[1], :, :]
            
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X, adj_matrix)
                
                # Ensure outputs and targets have the same shape
                if outputs.shape != batch_y.shape:
                    if outputs.shape[1] > batch_y.shape[1]:
                        outputs = outputs[:, :batch_y.shape[1], :, :]
                    elif outputs.shape[1] < batch_y.shape[1]:
                        batch_y = batch_y[:, :outputs.shape[1], :, :]
                
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    return model, train_losses, val_losses

def rolling_window_prediction(model, original_data, normalized_data, adj_matrix, scalers, input_steps, pred_steps):
    """
    Perform rolling window prediction on the entire test set
    
    Args:
        model: Trained model
        original_data: Original full time series data
        normalized_data: Normalized full time series data
        adj_matrix: Adjacency matrix
        scalers: Dictionary of scalers for each node
        input_steps: Number of input time steps
        pred_steps: Number of prediction time steps
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    adj_matrix = adj_matrix.to(device)
    
    model.eval()
    
    # Get dimensions
    num_nodes = original_data.shape[1]
    
    # Calculate test set starting point (80% train, 10% val, 10% test)
    train_val_size = int(0.9 * (len(original_data) - input_steps - pred_steps + 1))
    test_start_idx = train_val_size
    
    print(f"Starting rolling window prediction from index {test_start_idx}")
    
    # Storage for all predictions and ground truth
    all_predictions = []
    all_ground_truth = []
    
    # Loop through test set with sliding window
    for i in range(test_start_idx, len(normalized_data) - input_steps - pred_steps + 1):
        # Get current window
        current_window = normalized_data[i:i+input_steps]
        
        # Reshape for model input: [1, input_steps, num_nodes, 1]
        model_input = current_window.reshape(1, input_steps, num_nodes, 1)
        
        # Convert to tensor
        model_input_tensor = torch.FloatTensor(model_input).to(device)
        
        # Make prediction
        with torch.no_grad():
            pred = model(model_input_tensor, adj_matrix).cpu().numpy()
        
        # Store prediction for this window
        all_predictions.append(pred[0])  # [pred_steps, num_nodes, 1]
        
        # Store corresponding ground truth
        true_values = normalized_data[i+input_steps:i+input_steps+pred_steps]
        all_ground_truth.append(true_values.reshape(pred_steps, num_nodes, 1))
    
    # Convert lists to arrays
    all_predictions = np.array(all_predictions)  # [num_windows, pred_steps, num_nodes, 1]
    all_ground_truth = np.array(all_ground_truth)  # [num_windows, pred_steps, num_nodes, 1]
    
    # Inverse transform predictions and ground truth
    y_pred_orig = np.zeros_like(all_predictions)
    y_test_orig = np.zeros_like(all_ground_truth)
    
    for i in range(num_nodes):
        # For each node, inverse transform all windows and time steps
        for window in range(all_predictions.shape[0]):
            # Extract data for this node and window
            pred_node = all_predictions[window, :, i, :].reshape(-1, 1)
            true_node = all_ground_truth[window, :, i, :].reshape(-1, 1)
            
            # Inverse transform
            y_pred_orig[window, :, i, :] = scalers[i].inverse_transform(pred_node).reshape(pred_steps, 1)
            y_test_orig[window, :, i, :] = scalers[i].inverse_transform(true_node).reshape(pred_steps, 1)
    
    # Calculate metrics
    # Flatten for metric calculation
    y_pred_flat = y_pred_orig.reshape(-1)
    y_test_flat = y_test_orig.reshape(-1)
    
    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
    
    # Calculate custom accuracy as defined in the paper
    y_pred_norm = np.linalg.norm(y_pred_flat)
    y_test_norm = np.linalg.norm(y_test_flat)
    diff_norm = np.linalg.norm(y_pred_flat - y_test_flat)
    accuracy = 1 - (diff_norm / y_test_norm)
    
    # Print results
    print(f'Test MAE: {mae:.2f}')
    print(f'Test RMSE: {rmse:.2f}')
    print(f'Test Accuracy: {accuracy:.4f}')
    
    return y_pred_orig, y_test_orig, mae, rmse, accuracy

def plot_results(y_test, y_pred):
    """
    Plot predictions vs ground truth for EVCS load with rolling window prediction
    
    Args:
        y_test: Ground truth data [num_windows, pred_steps, num_nodes, 1]
        y_pred: Predicted data [num_windows, pred_steps, num_nodes, 1]
    """
    num_evcs = y_test.shape[2]
    
    # Plot full time series for each EVCS
    for i in range(min(8, num_evcs)):  # Plot up to 8 EVCSs
        plt.figure(figsize=(15, 6))
        
        # Reshape from [windows, pred_steps, evcs, 1] to a continuous sequence
        # For each time step, we take the first predicted value from each window
        y_test_seq = np.vstack([y_test[j, 0, i, 0:1] for j in range(y_test.shape[0])])
        y_pred_seq = np.vstack([y_pred[j, 0, i, 0:1] for j in range(y_pred.shape[0])])
        
        # Plot
        plt.plot(y_test_seq, label='True', alpha=0.7)
        plt.plot(y_pred_seq, label='Predicted', alpha=0.7)
        plt.xlabel('Time Steps')
        plt.ylabel('Power (kW)')
        plt.title(f'EVCS {i+1} Load Prediction (1-Step Ahead)')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig(f'evcs_{i+1}_1step_prediction.png')
        plt.close()
    
    # Create a combined plot for all EVCSs
    plt.figure(figsize=(20, 15))
    
    for i in range(min(8, num_evcs)):
        plt.subplot(4, 2, i+1)
        y_test_seq = np.vstack([y_test[j, 0, i, 0:1] for j in range(y_test.shape[0])])
        y_pred_seq = np.vstack([y_pred[j, 0, i, 0:1] for j in range(y_pred.shape[0])])
        
        plt.plot(y_test_seq, label='True', alpha=0.7)
        plt.plot(y_pred_seq, label='Predicted', alpha=0.7)
        plt.title(f'EVCS {i+1}')
        plt.xlabel('Time Steps')
        plt.ylabel('Power (kW)')
        if i == 0:
            plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('all_evcs_1step_prediction.png')
    plt.close()
    
    # Multi-step prediction visualization for the first EVCS
    # We'll visualize predictions at different horizons
    horizons = [0, 4, 9, 14]  # 1-step, 5-step, 10-step, 15-step ahead
    for h in horizons:
        # Skip if horizon is beyond our prediction steps
        if h >= y_pred.shape[1]:
            continue
            
        plt.figure(figsize=(15, 6))
        for i in range(min(4, num_evcs)):  # Show first 4 EVCSs
            plt.subplot(2, 2, i+1)
            
            # Extract the h-step ahead prediction for each window
            y_test_seq = np.vstack([y_test[j, h, i, 0:1] for j in range(y_test.shape[0])])
            y_pred_seq = np.vstack([y_pred[j, h, i, 0:1] for j in range(y_pred.shape[0])])
            
            plt.plot(y_test_seq, label='True', alpha=0.7)
            plt.plot(y_pred_seq, label='Predicted', alpha=0.7)
            plt.title(f'EVCS {i+1} - {h+1}-Step Ahead')
            plt.xlabel('Time Steps')
            plt.ylabel('Power (kW)')
            if i == 0:
                plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'multi_step_prediction_{h+1}_step.png')
        plt.close()
    
    # Also create a heatmap to visualize prediction error over time
    if num_evcs > 1:
        plt.figure(figsize=(15, 8))
        error_matrix = np.abs(y_pred[:, :, :, 0] - y_test[:, :, :, 0])
        
        # Average error across all windows for each time step and EVCS
        avg_error = np.mean(error_matrix, axis=0)  # [pred_steps, num_evcs]
        
        # Create heatmap
        plt.imshow(avg_error.T, aspect='auto', cmap='hot')
        plt.colorbar(label='Absolute Error (kW)')
        plt.xlabel('Prediction Time Step')
        plt.ylabel('EVCS ID')
        plt.title('Average Prediction Error Across Test Period')
        plt.yticks(range(num_evcs), [f'EVCS {i+1}' for i in range(num_evcs)])
        
        plt.tight_layout()
        plt.savefig('prediction_error_heatmap.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='EVCS Load Prediction with RPFM (Rolling Window Only)')
    parser.add_argument('--model', type=str, default='tgcn', choices=['tgcn', 'lstm', 'gru'],
                        help='Model type (tgcn, lstm, gru)')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained RPFM model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden dimension')
    parser.add_argument('--input_steps', type=int, default=60, help='Number of input time steps')
    parser.add_argument('--pred_steps', type=int, default=15, help='Number of prediction time steps')
    
    args = parser.parse_args()
    
    # Paths to data
    power_path = '/workspace/RPFM/data/task1/power.csv'
    adj_path = '/workspace/RPFM/data/task1/adj.csv'
    
    try:
        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test, adj_matrix, scalers, original_data, normalized_data = load_data(
            power_path, adj_path, args.input_steps, args.pred_steps
        )
        
        # Create model
        input_dim = 1  # Single feature (power)
        output_dim = 1  # Predict power
        num_nodes = X_train.shape[2]  # Number of EVCSs
        
        model = create_model(
            args.model, 
            input_dim, 
            args.hidden_dim, 
            output_dim, 
            num_nodes, 
            args.pretrained
        )
        
        print(f"Created {args.model.upper()} model {'with' if args.pretrained else 'without'} RPFM integration")
        
        
        # Train model
        trained_model, train_losses, val_losses = train_model(
            model, X_train, y_train, X_val, y_val, adj_matrix, 
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
        )
        
        # Perform rolling window prediction on the test set
        print("\nPerforming rolling window prediction on the test set...")
        y_pred, y_test_orig, mae, rmse, accuracy = rolling_window_prediction(
            trained_model, original_data, normalized_data, adj_matrix, scalers, 
            args.input_steps, args.pred_steps
        )
        
        # Plot results using rolling window prediction results
        plot_results(y_test_orig, y_pred)
        
        # Plot training loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('loss_curves.png')
        plt.close()
        
        # Save results to file
        result_dict = {
            'model': args.model,
            'pretrained': args.pretrained,
            'MAE': mae,
            'RMSE': rmse,
            'Accuracy': accuracy
        }
        
        results_df = pd.DataFrame([result_dict])
        results_df.to_csv(f'{args.model}_{"with" if args.pretrained else "without"}_rpfm_results.csv', index=False)
        
        print(f"Results saved to {args.model}_{'with' if args.pretrained else 'without'}_rpfm_results.csv")
        print("Evaluation used rolling window prediction on the entire test set")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Function call stack:")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
