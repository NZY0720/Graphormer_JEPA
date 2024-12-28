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
        out (Tensor): The destination tensor, e.g. shape [B, num_heads, N, N].
        index (Tensor): The indices to scatter into `out`, e.g. shape [B, num_heads, N, E].
        src (Tensor): The source values, e.g. shape [B, num_heads, 1, E].
        dim (int): The dimension along which to scatter. Default is 3.

    Returns:
        Tensor: The updated 'out' tensor, after scattering valid entries.
    
    Process:
        1) We flatten all but the dimension we scatter along, so we can gather
           the valid positions in one pass.
        2) We find the subset of positions where 'index' is within the valid range.
        3) We place 'src' values into 'out' at those positions, skipping any invalid ones.
    
    Note:
        - This function might be slow for large tensors because it does additional
          masking and advanced indexing. Use at your own risk.
        - Any index < 0 or >= out.size(dim) is ignored.
    """

    # Check shapes roughly
    if out.dim() != index.dim() or out.dim() != src.dim():
        raise ValueError("safe_scatter_: 'out', 'index', 'src' must have the same number of dimensions.")

    if dim < 0:
        dim = out.dim() + dim

    # For simplicity, assume the shapes match except at 'dim'
    # E.g. out.shape == [B, H, N, N]
    #      index.shape == [B, H, N, E]
    #      src.shape   == [B, H, 1, E]  (or maybe [B, H, N, E], you can adapt as needed)

    # (1) Identify valid mask: index in [0, out.size(dim))
    valid_mask = (index >= 0) & (index < out.size(dim))
    if not valid_mask.any():
        # No valid entries, nothing to scatter
        return out

    # (2) Flatten 'index' so we can do advanced indexing
    # Let's get the coordinates of all valid positions
    coords = valid_mask.nonzero(as_tuple=False)  # shape [K, out.dim()], K = # of valid entries
    # coords[i] = [b, h, row, e] if dim=3

    # The actual index value at these coords:
    index_values = index[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]]
    # The corresponding src value:
    # If your src shape is [B, H, 1, E], then row dimension is always 0 for src
    # so we do something like:
    src_values = src[coords[:, 0], coords[:, 1], 0, coords[:, 3]]

    # (3) Scatter manually: out[b, h, row, index_val] = src_val
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

    def forward(self, x, edge_index=None, edge_attr=None):
        """
        Forward pass for multi-head attention with edge attributes.
        In this modified version, we remove any edges that reference
        out-of-range node indices (>= N) to avoid scatter index errors.

        Args:
            x (Tensor): Node features, shape [B, N, D], where
                        B = batch size,
                        N = number of nodes,
                        D = embedding dimension.
            edge_index (Tensor, optional): Edge indices, shape [2, E].
            edge_attr (Tensor, optional): Edge attributes, shape [E, E_dim] (e.g. [E, 3]).

        Returns:
            Tensor: Output features after attention, shape [B, N, D].
        """

        B, N, D = x.shape  # batch_size, num_nodes, embedding_dim
        # 1) Linear projections
        Q = self.q_proj(x)  # [B, N, D]
        K = self.k_proj(x)  # [B, N, D]
        V = self.v_proj(x)  # [B, N, D]

        # 2) Reshape for multi-head attention
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]

        # 3) Compute attention scores
        scores = (Q @ K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # [B, num_heads, N, N]

        # 4) If we have edge attributes, add them to scores
        if edge_index is not None and edge_attr is not None:
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # (A) Remove edges that reference out-of-range nodes
            #     Suppose edge_index shape = [2, E], with 0-row = src, 1-row = tgt
            #     We only keep edges where src < N and tgt < N
            src_nodes = edge_index[0]  # shape [E]
            tgt_nodes = edge_index[1]  # shape [E]
            valid_mask = (src_nodes < N) & (tgt_nodes < N)
            if valid_mask.sum().item() < len(src_nodes):
                # We are indeed filtering out some edges
                edge_index = edge_index[:, valid_mask]            # shape [2, E']
                edge_attr = edge_attr[valid_mask]                # shape [E', E_dim]

            # (B) Now proceed with attention bias
            E = edge_index.shape[1]  # number of valid edges
            # Project edge_attr -> [E, num_heads]
            edge_bias = self.edge_bias_proj(edge_attr)
            # Prepare a full matrix to store these biases
            edge_bias_full = torch.zeros(
                (B, self.num_heads, N, N), 
                device=x.device, 
                dtype=edge_bias.dtype
            )
            # Expand edge_bias to shape [B, num_heads, 1, E]
            edge_bias = edge_bias.unsqueeze(0).expand(B, -1, -1)  # [B, E, num_heads]
            edge_bias = edge_bias.permute(0, 2, 1)                # [B, num_heads, E]
            edge_bias = edge_bias.unsqueeze(2)                    # [B, num_heads, 1, E]

            # We'll scatter on dim=3, using tgt_nodes as index
            tgt_index = tgt_nodes.view(1, 1, 1, E).expand(B, self.num_heads, N, E)
            edge_bias_full = safe_scatter_(
                out=edge_bias_full,
                index=tgt_index,
                src=edge_bias,
                dim=3
            )


            # Add these biases to scores
            scores = scores + edge_bias_full
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # 5) Softmax over the last dim => attention probabilities
        attn = F.softmax(scores, dim=-1)  # [B, num_heads, N, N]
        attn = self.attn_drop(attn)

        # 6) Weighted sum over V
        out = attn @ V  # [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]

        # 7) Final linear projection
        out = self.out_proj(out)  # [B, N, D]
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

    def forward(self, x, edge_index=None, edge_attr=None):
        """
        Forward pass for a single Graphormer layer.

        Args:
            x (Tensor): Input node features, shape [B, N, D].
            edge_index (Tensor, optional): Edge indices, shape [2, E].
            edge_attr (Tensor, optional): Edge attributes, shape [E, 3].

        Returns:
            Tensor: Output node features after the layer, shape [B, N, D].
        """
        # Multi-head attention with residual connection
        h = self.norm1(x)  # Apply layer normalization
        h = self.attn(h, edge_index=edge_index, edge_attr=edge_attr)  # Apply multi-head attention
        x = x + self.dropout(h)  # Add residual connection and apply dropout

        # Feed-forward network with residual connection
        h = self.norm2(x)  # Apply layer normalization
        h = self.ffn(h)    # Apply feed-forward network
        x = x + self.dropout(h)  # Add residual connection and apply dropout

        return x  # [B, N, D]


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

    def forward(self, x, degree, node_ids, edge_index=None, edge_attr=None):
        """
        Forward pass for the Graphormer encoder.

        Args:
            x (Tensor): Input node features, shape [B, N, input_dim].
            degree (Tensor): Node degrees, shape [B, N].
            node_ids (Tensor): Node IDs for positional embedding, shape [B, N].
            edge_index (Tensor, optional): Edge indices, shape [2, E].
            edge_attr (Tensor, optional): Edge attributes, shape [E, 3].

        Returns:
            Tensor: Hidden node representations, shape [B, N, hidden_dim].
        """
        if x.dim() == 2:  # Handle case where batch size might be missing
            x = x.unsqueeze(0)            # [1, N, D]
            degree = degree.unsqueeze(0)  # [1, N]
            node_ids = node_ids.unsqueeze(0)  # [1, N]

        B, N, _ = x.shape  # Batch size, number of nodes, input dimension

        # 1) Project input features to hidden dimensions
        h = self.input_proj(x)  # [B, N, hidden_dim]

        # 2) Add degree embeddings to node features
        deg_embed = self.degree_embedding(torch.clamp(degree, max=self.max_degree - 1))  # [B, N, hidden_dim]
        h = h + deg_embed  # [B, N, hidden_dim]

        # 3) Add positional embeddings based on node IDs
        node_ids_clamped = torch.clamp(node_ids, max=self.max_nodes - 1)  # Clamp node IDs to max_nodes
        pos_embed = self.pos_embedding(node_ids_clamped)  # [B, N, hidden_dim]
        h = h + pos_embed  # [B, N, hidden_dim]

        # 4) Pass through each Graphormer layer
        for layer in self.layers:
            h = layer(h, edge_index=edge_index, edge_attr=edge_attr)  # [B, N, hidden_dim]

        # 5) Apply final layer normalization
        h = self.output_norm(h)  # [B, N, hidden_dim]

        return h  # [B, N, hidden_dim]


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
        - Downstream node-level supervised prediction (pretrain=False): Returns [B, N] predictions.
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
            context_batch (Data): PyG Data object for the context subgraph.
            target_batch (Data): PyG Data object for the target subgraph (only used if pretrain=True).
            pretrain (bool): Flag indicating whether to perform pretraining.

        Returns:
            Tensor:
                - If pretrain=True: Scalar loss.
                - If pretrain=False: Node-level predictions, shape [B, N].
        """
        # Encode the context subgraph
        context_h = self.context_encoder(
            context_batch.x, context_batch.degree, context_batch.node_ids,
            edge_index=context_batch.edge_index, edge_attr=context_batch.edge_attr
        )  # [B, N_c, hidden_dim]

        if pretrain:
            # Encode the target subgraph
            target_h = self.target_encoder(
                target_batch.x, target_batch.degree, target_batch.node_ids,
                edge_index=target_batch.edge_index, edge_attr=target_batch.edge_attr
            )  # [B, N_t, hidden_dim]

            # Global average pooling to get graph-level representations
            context_rep = context_h.mean(dim=1)  # [B, hidden_dim]
            target_rep = target_h.mean(dim=1)    # [B, hidden_dim]

            # Predict target representations from context representations
            predicted_target = self.predictor(context_rep)  # [B, hidden_dim]

            # Compute Huber-like loss between predicted and actual target representations
            loss = huber_like_loss(predicted_target, target_rep, self.delta)

            return loss  # Scalar loss
        else:
            # Downstream node-level supervised prediction
            # Pass the encoded context through the prediction head
            predicted_scores = self.prediction_head(context_h).squeeze(-1)  # [B, N]

            return predicted_scores  # Node-level predictions [B, N]
