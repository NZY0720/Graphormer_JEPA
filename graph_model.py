# graph_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch


class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == embed_dim), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, dist_matrix=None):
        # x: [B, N, D]
        # dist_matrix: [B, N, N]
        B, N, D = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # [B, h, N, N]

        if dist_matrix is not None:
            # Create mask: 1 indicates edge, 0 indicates non-edge
            mask = (dist_matrix != 1e9).float()  # [B, N, N]

            # Normalize valid edge distances to [0, 1]
            valid_dist = dist_matrix * mask  # Non-edge distances set to 0
            max_dist = valid_dist.view(B, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)  # [B, 1, 1]
            max_dist = max_dist.clamp(min=1.0)  # Prevent division by zero
            normalized_dist = valid_dist / max_dist  # [B, N, N]

            # Use mask to set non-edge positions to -inf, resulting in 0 in softmax
            # Use a tunable parameter (e.g., 10.0) to control the impact of distance on attention scores
            attn_scores = scores - (normalized_dist.unsqueeze(1) * 10.0)  # [B, h, N, N]
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        else:
            attn_scores = scores

        attn = F.softmax(attn_scores, dim=-1)  # [B, h, N, N]
        attn = self.attn_drop(attn)
        out = attn @ V  # [B, h, N, d]
        out = out.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        out = self.out_proj(out)
        return out


class GraphormerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attn = GraphormerMultiHeadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, dist_matrix=None):
        h = self.norm1(x)
        h = self.attn(h, dist_matrix=dist_matrix)
        x = x + self.dropout(h)

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)
        return x


class Graphormer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4, num_layers=4, dropout=0.3, max_degree=128, max_nodes=50000):
        super(Graphormer, self).__init__()
        self.max_degree = max_degree
        self.max_nodes = max_nodes

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.degree_embedding = nn.Embedding(self.max_degree, hidden_dim)
        self.pos_embedding = nn.Embedding(self.max_nodes, hidden_dim)

        self.layers = nn.ModuleList([
            GraphormerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, degree, node_ids, edge_index=None, edge_attr=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            degree = degree.unsqueeze(0)
            node_ids = node_ids.unsqueeze(0)

        B, N, _ = x.size()
        h = self.input_proj(x)
        deg_embed = self.degree_embedding(torch.clamp(degree, max=self.max_degree - 1))
        h = h + deg_embed

        node_ids_clamped = torch.clamp(node_ids, max=self.max_nodes - 1)
        pos_embed = self.pos_embedding(node_ids_clamped)
        h = h + pos_embed

        dist_matrix = None
        if edge_index is not None and edge_attr is not None:
            dist_matrix = torch.full((B, N, N), 1e9, device=h.device)
            dist_matrix[:, torch.arange(N), torch.arange(N)] = 0.0
            row, col = edge_index
            dist_matrix[:, row, col] = edge_attr.squeeze(-1)
            dist_matrix[:, col, row] = edge_attr.squeeze(-1)

        for layer in self.layers:
            h = layer(h, dist_matrix=dist_matrix)

        h = self.output_norm(h)
        return h, dist_matrix


class GraphormerJEPA(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_degree=128, max_nodes=50000, alpha=0.001):
        super(GraphormerJEPA, self).__init__()
        self.context_encoder = Graphormer(input_dim, hidden_dim, max_degree=max_degree, max_nodes=max_nodes)
        self.target_encoder = Graphormer(input_dim, hidden_dim, max_degree=max_degree, max_nodes=max_nodes)
        self.prediction_head = nn.Linear(hidden_dim, output_dim)
        self.alpha = alpha  # Weight for spatial loss

    def forward(self, context_batch, target_batch):
        context_embeddings, context_dist = self.context_encoder(
            context_batch.x, context_batch.degree, context_batch.node_ids,
            edge_index=context_batch.edge_index, edge_attr=context_batch.edge_attr
        )
        target_embeddings, target_dist = self.target_encoder(
            target_batch.x, target_batch.degree, target_batch.node_ids,
            edge_index=target_batch.edge_index, edge_attr=target_batch.edge_attr
        )

        combined_embeddings = context_embeddings
        predicted_scores = self.prediction_head(combined_embeddings).squeeze(-1)

        if target_batch.x.dim() == 3:
            target_scores = target_batch.x[:, :, 2]
        elif target_batch.x.dim() == 2:
            target_scores = target_batch.x[:, 2].unsqueeze(0)
        else:
            raise ValueError("Unexpected dimensions for target_batch.x")

        # Calculate spatial loss: average distance between nodes predicted as 1
        pred_probs = torch.sigmoid(predicted_scores)
        pred_mask = (pred_probs >= 0.5).float()  # Use a fixed threshold of 0.5 to calculate spatial loss
        # dist_matrix: [B, N, N] from context_dist
        # If no dist_matrix, do not calculate spatial loss
        spatial_loss = 0.0
        if context_dist is not None:
            count_ones = pred_mask.sum(dim=1)  # [B]
            if (count_ones > 1).any():
                # dist_sub retains distances between nodes predicted as 1
                # Expand pred_mask to [B, N, 1] and [B, 1, N] for broadcasting
                pred_mask_expanded = pred_mask.unsqueeze(2) * pred_mask.unsqueeze(1)  # [B, N, N]
                dist_sub = context_dist * pred_mask_expanded
                sum_dist = dist_sub.sum(dim=-1).sum(dim=-1)  # [B]
                pairs = count_ones * (count_ones - 1)  # [B]
                spatial_loss = (sum_dist / (pairs + 1e-8)).mean()  # Average spatial loss across all batches
                spatial_loss = torch.clamp(spatial_loss, max=1.0)  # Limit the maximum spatial loss
            else:
                spatial_loss = 0.0

        return predicted_scores, target_scores, spatial_loss * self.alpha
