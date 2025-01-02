# graph_model_1.py

import torch
import torch.nn as nn
import torch.nn.functional as F

###################################################
# 1. Graphormer Modules with Edge Attributes
###################################################

def safe_scatter_(out: torch.Tensor,
                  index: torch.Tensor,
                  src: torch.Tensor,
                  dim: int = 3) -> torch.Tensor:
    """
    A custom 'scatter_' that skips any out-of-range indices instead of throwing an error.

    Args:
        out (Tensor): The destination tensor, e.g., shape [B, num_heads, N, N].
        index (Tensor): The indices to scatter into `out`, e.g., shape [B, num_heads, N, E].
        src (Tensor): The source values, e.g., shape [B, num_heads, 1, E].
        dim (int): The dimension along which to scatter. Default is 3.

    Returns:
        Tensor: The updated 'out' tensor, after scattering valid entries.
    
    Process:
        1) Identify valid indices within the range.
        2) Scatter source values into the output tensor at valid positions.
        3) Skip any invalid indices to prevent errors.
    
    Note:
        - This function might be slow for large tensors due to masking and advanced indexing.
        - Any index < 0 or >= out.size(dim) is ignored.
    """

    # Validate that all tensors have the same number of dimensions
    if out.dim() != index.dim() or out.dim() != src.dim():
        raise ValueError("safe_scatter_: 'out', 'index', 'src' must have the same number of dimensions.")

    if dim < 0:
        dim = out.dim() + dim

    # Identify valid indices within the specified dimension
    valid_mask = (index >= 0) & (index < out.size(dim))
    if not valid_mask.any():
        # If no valid indices, return the original tensor
        return out

    # Extract coordinates of valid indices
    coords = valid_mask.nonzero(as_tuple=False)  # Shape: [K, out.dim()]

    # Gather valid index values and corresponding source values
    index_values = index[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]]
    src_values = src[coords[:, 0], coords[:, 1], 0, coords[:, 3]]

    # Scatter the source values into the output tensor
    out[coords[:, 0], coords[:, 1], coords[:, 2], index_values] = src_values

    return out

class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Initialize the multi-head attention mechanism with support for edge attributes.

        Args:
            embed_dim (int): Embedding dimension for the nodes.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability for attention weights.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout layer for attention probabilities
        self.attn_drop = nn.Dropout(dropout)

        # Linear layer to transform edge attributes into attention biases
        # Assuming edge_attr has dimension E_dim, here we set E_dim = 3 as per utils.py
        self.edge_bias_proj = nn.Linear(3, num_heads)  # Transform edge attributes to biases per head

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        """
        Forward pass for multi-head attention with edge attributes.
        Supports batched graph inputs.

        Args:
            x (Tensor): Node features, shape [total_nodes, D], where
                        total_nodes = sum of nodes in all graphs in the batch,
                        D = embedding dimension.
            edge_index (Tensor, optional): Edge indices, shape [2, E].
            edge_attr (Tensor, optional): Edge attributes, shape [E, E_dim] (e.g., [E, 3]).
            batch (Tensor, optional): Batch vector, shape [total_nodes],
                                      where batch[i] indicates the graph index to which node i belongs.

        Returns:
            Tensor: Output features after attention, shape [total_nodes, D].
        """

        # 1) Linear projections
        Q = self.q_proj(x)  # [total_nodes, D]
        K = self.k_proj(x)  # [total_nodes, D]
        V = self.v_proj(x)  # [total_nodes, D]

        # 2) Reshape for multi-head attention
        # [total_nodes, D] -> [total_nodes, num_heads, head_dim]
        Q = Q.view(-1, self.num_heads, self.head_dim)
        K = K.view(-1, self.num_heads, self.head_dim)
        V = V.view(-1, self.num_heads, self.head_dim)

        # Transpose to [num_heads, total_nodes, head_dim]
        Q = Q.transpose(0, 1)  # [num_heads, total_nodes, head_dim]
        K = K.transpose(0, 1)  # [num_heads, total_nodes, head_dim]
        V = V.transpose(0, 1)  # [num_heads, total_nodes, head_dim]

        # 3) Compute attention scores
        # [num_heads, total_nodes, head_dim] @ [num_heads, head_dim, total_nodes] -> [num_heads, total_nodes, total_nodes]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [num_heads, total_nodes, total_nodes]

        # 4) If we have edge attributes, add them to scores
        if edge_index is not None and edge_attr is not None:
            E = edge_index.shape[1]  # number of edges
            # Project edge_attr -> [E, num_heads]
            edge_bias = self.edge_bias_proj(edge_attr)  # [E, num_heads]
            # Transpose to [num_heads, E]
            edge_bias = edge_bias.transpose(0, 1)  # [num_heads, E]

            # Initialize a tensor to store edge biases
            edge_bias_full = torch.zeros(
                (self.num_heads, x.size(0), x.size(0)),
                device=x.device,
                dtype=edge_bias.dtype
            )  # [num_heads, total_nodes, total_nodes]

            # Scatter the edge biases into the full tensor
            for head in range(self.num_heads):
                # Scatter edge biases into the appropriate positions in the attention scores
                edge_bias_full[head].index_put_(
                    (edge_index[0], edge_index[1]),
                    edge_bias[head],
                    accumulate=True
                )

            # Add the edge biases to the attention scores
            scores = scores + edge_bias_full  # [num_heads, total_nodes, total_nodes]

        # 5) If batch information is provided, mask attention scores to prevent cross-graph attention
        if batch is not None:
            # batch: [total_nodes], each value is the graph index
            # Create mask where mask[i,j] = 1 if nodes i and j are in the same graph, else 0
            # Expand to [num_heads, total_nodes, total_nodes]
            batch = batch.unsqueeze(0).unsqueeze(0)  # [1, 1, total_nodes]
            batch_i = batch.expand(self.num_heads, -1, -1)  # [num_heads, 1, total_nodes]
            batch_j = batch.transpose(1, 2).expand(self.num_heads, -1, -1)  # [num_heads, total_nodes, 1]
            attention_mask = (batch_i == batch_j).float()  # [num_heads, total_nodes, total_nodes]

            # Set scores to -inf where attention_mask == 0 to mask them out
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        # 6) Softmax over the last dimension to obtain attention probabilities
        attn = F.softmax(scores, dim=-1)  # [num_heads, total_nodes, total_nodes]
        attn = self.attn_drop(attn)

        # 7) Weighted sum over V
        # [num_heads, total_nodes, total_nodes] @ [num_heads, total_nodes, head_dim] -> [num_heads, total_nodes, head_dim]
        out = torch.matmul(attn, V)  # [num_heads, total_nodes, head_dim]

        # 8) Transpose and reshape to [total_nodes, embed_dim]
        out = out.transpose(0, 1).contiguous().view(-1, self.embed_dim)  # [total_nodes, D]

        # 9) Final linear projection
        out = self.out_proj(out)  # [total_nodes, D]

        return out

class GraphormerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        """
        Initialize a single Graphormer layer, comprising multi-head attention and a feed-forward network.

        Args:
            hidden_dim (int): Hidden dimension size.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.attn = GraphormerMultiHeadAttention(hidden_dim, num_heads, dropout)  # Multi-head attention
        self.norm1 = nn.LayerNorm(hidden_dim)  # Layer normalization after attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),  # Feed-forward network expansion
            nn.ReLU(),                             # Activation function
            nn.Linear(4 * hidden_dim, hidden_dim)  # Feed-forward network contraction
        )
        self.norm2 = nn.LayerNorm(hidden_dim)  # Layer normalization after feed-forward
        self.dropout = nn.Dropout(dropout)      # Dropout layer

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        """
        Forward pass for a single Graphormer layer.

        Args:
            x (Tensor): Input node features, shape [total_nodes, D].
            edge_index (Tensor, optional): Edge indices, shape [2, E].
            edge_attr (Tensor, optional): Edge attributes, shape [E, 3].
            batch (Tensor, optional): Batch vector, shape [total_nodes].

        Returns:
            Tensor: Output node features after the layer, shape [total_nodes, D].
        """
        # Multi-head attention with residual connection
        h = self.norm1(x)  # Apply layer normalization
        h = self.attn(h, edge_index=edge_index, edge_attr=edge_attr, batch=batch)  # Apply multi-head attention
        x = x + self.dropout(h)  # Add residual connection and apply dropout

        # Feed-forward network with residual connection
        h = self.norm2(x)  # Apply layer normalization
        h = self.ffn(h)    # Apply feed-forward network
        x = x + self.dropout(h)  # Add residual connection and apply dropout

        return x  # [total_nodes, D]

class Graphormer(nn.Module):
    """
    Simplified Graphormer Encoder.

    Transforms input node features into hidden representations using multiple Graphormer layers.
    """
    def __init__(self, input_dim, hidden_dim, num_heads=4, num_layers=4,
                 dropout=0.1, max_degree=128, max_nodes=50000):
        """
        Initialize the Graphormer encoder.

        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Dimension of hidden node features.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Graphormer layers.
            dropout (float): Dropout probability.
            max_degree (int): Maximum node degree for embedding.
            max_nodes (int): Maximum number of nodes for positional embedding.
        """
        super().__init__()
        self.max_degree = max_degree
        self.max_nodes = max_nodes

        # Linear projection of input features to hidden dimensions
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Embeddings for node degrees
        self.degree_embedding = nn.Embedding(self.max_degree, hidden_dim)

        # Positional embeddings based on node IDs
        self.pos_embedding = nn.Embedding(self.max_nodes, hidden_dim)

        # Stack multiple Graphormer layers
        self.layers = nn.ModuleList([
            GraphormerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        # Final layer normalization
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, degree, node_ids, edge_index=None, edge_attr=None, batch=None):
        """
        Forward pass for the Graphormer encoder.

        Args:
            x (Tensor): Input node features, shape [total_nodes, input_dim].
            degree (Tensor): Node degrees, shape [total_nodes].
            node_ids (Tensor): Node IDs for positional embedding, shape [total_nodes].
            edge_index (Tensor, optional): Edge indices, shape [2, E].
            edge_attr (Tensor, optional): Edge attributes, shape [E, 3].
            batch (Tensor, optional): Batch vector, shape [total_nodes].

        Returns:
            Tensor: Hidden node representations, shape [total_nodes, hidden_dim].
        """
        # 1) Project input features to hidden dimensions
        h = self.input_proj(x)  # [total_nodes, hidden_dim]

        # 2) Add degree embeddings to node features
        deg_embed = self.degree_embedding(torch.clamp(degree, max=self.max_degree - 1))  # [total_nodes, hidden_dim]
        h = h + deg_embed  # [total_nodes, hidden_dim]

        # 3) Add positional embeddings based on node IDs
        node_ids_clamped = torch.clamp(node_ids, max=self.max_nodes - 1)  # Clamp node IDs to max_nodes
        pos_embed = self.pos_embedding(node_ids_clamped)  # [total_nodes, hidden_dim]
        h = h + pos_embed  # [total_nodes, hidden_dim]

        # 4) Pass through each Graphormer layer
        for layer in self.layers:
            h = layer(h, edge_index=edge_index, edge_attr=edge_attr, batch=batch)  # [total_nodes, hidden_dim]

        # 5) Apply final layer normalization
        h = self.output_norm(h)  # [total_nodes, hidden_dim]

        return h  # [total_nodes, hidden_dim]

###################################################
# 2. Huber-like Loss
###################################################

def huber_like_loss(pred, target, delta=1.0):
    """
    Compute the Huber-like loss between predictions and targets.

    Args:
        pred (Tensor): Predicted tensor, shape [B, D].
        target (Tensor): Target tensor, shape [B, D].
        delta (float): Threshold at which to change between quadratic and linear loss.

    Returns:
        Tensor: Scalar loss value.
    """
    diff = pred - target  # [B, D]
    abs_diff = torch.abs(diff)  # [B, D]
    mask = (abs_diff <= delta).float()  # [B, D], 1 where |diff| <= delta, else 0

    # Quadratic loss for small differences
    huber_part = 0.5 * (diff ** 2) * mask  # [B, D]

    # Linear loss for large differences
    linear_part = delta * (abs_diff - 0.5 * delta) * (1 - mask)  # [B, D]

    return (huber_part + linear_part).mean()  # Scalar

###################################################
# 3. GraphormerJEPA Model with Edge Attributes
###################################################

class GraphormerJEPA(nn.Module):
    """
    Joint Embedding Predictive Architecture (JEPA) using Graphormer.

    Supports both:
        - Self-supervised pretraining (pretrain=True): Returns scalar loss.
        - Downstream node-level supervised prediction (pretrain=False): Returns [total_nodes] predictions.
    """
    def __init__(self, input_dim, hidden_dim,
                 max_degree=128, max_nodes=50000,
                 num_heads=4, num_layers=4, dropout=0.1, delta=1.0):
        """
        Initialize the GraphormerJEPA model.

        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Dimension of hidden node features.
            max_degree (int): Maximum node degree for embedding.
            max_nodes (int): Maximum number of nodes for positional embedding.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Graphormer layers.
            dropout (float): Dropout probability.
            delta (float): Threshold for Huber-like loss.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.delta = delta

        # Separate encoders for context and target graphs
        self.context_encoder = Graphormer(
            input_dim, hidden_dim, num_heads=num_heads,
            num_layers=num_layers, dropout=dropout,
            max_degree=max_degree, max_nodes=max_nodes
        )
        self.target_encoder = Graphormer(
            input_dim, hidden_dim, num_heads=num_heads,
            num_layers=num_layers, dropout=dropout,
            max_degree=max_degree, max_nodes=max_nodes
        )

        # Predictor for aligning context and target representations in pretraining
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Prediction head for downstream node-level supervised tasks
        self.prediction_head = nn.Linear(hidden_dim, 1)  # Maps hidden features to scalar predictions

    def forward(self, context_batch, target_batch, pretrain=False):
        """
        Forward pass of the GraphormerJEPA model.

        Args:
            context_batch (torch_geometric.data.Batch): Batch of context subgraphs.
            target_batch (torch_geometric.data.Batch): Batch of target subgraphs.
            pretrain (bool): Flag indicating whether to perform pretraining.

        Returns:
            Tensor:
                - If pretrain=True: Scalar loss.
                - If pretrain=False: Node-level predictions, shape [total_nodes].
        """
        # Encode the context subgraphs
        context_h = self.context_encoder(
            x=context_batch.x,
            degree=context_batch.degree,
            node_ids=context_batch.node_ids,
            edge_index=context_batch.edge_index,
            edge_attr=context_batch.edge_attr,
            batch=context_batch.batch
        )  # [total_context_nodes, hidden_dim]

        if pretrain:
            # Encode the target subgraphs
            target_h = self.target_encoder(
                x=target_batch.x,
                degree=target_batch.degree,
                node_ids=target_batch.node_ids,
                edge_index=target_batch.edge_index,
                edge_attr=target_batch.edge_attr,
                batch=target_batch.batch
            )  # [total_target_nodes, hidden_dim]

            # Global average pooling to get graph-level representations
            # PyG's global_mean_pool can be used, but since we have batch vectors, average per graph
            from torch_geometric.nn import global_mean_pool

            context_rep = global_mean_pool(context_h, context_batch.batch)  # [B, hidden_dim]
            target_rep = global_mean_pool(target_h, target_batch.batch)    # [B, hidden_dim]

            # Predict target representations from context representations
            predicted_target = self.predictor(context_rep)  # [B, hidden_dim]

            # Compute Huber-like loss between predicted and actual target representations
            loss = huber_like_loss(predicted_target, target_rep, self.delta)

            return loss  # Scalar loss
        else:
            # Downstream node-level supervised prediction
            # Pass the encoded context through the prediction head
            predicted_scores = self.prediction_head(context_h).squeeze(-1)  # [total_context_nodes]

            return predicted_scores  # Node-level predictions [total_nodes, ]

