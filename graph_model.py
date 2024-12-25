import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
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

    def forward(self, x, edge_attr=None):
        B, N, D = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (Q @ K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ V
        out = out.transpose(1, 2).reshape(B, N, D)
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

    def forward(self, x, edge_attr=None):
        h = self.norm1(x)
        h = self.attn(h, edge_attr=edge_attr)
        x = x + self.dropout(h)
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)
        return x

class Graphormer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4, num_layers=4, dropout=0.1, max_degree=128, max_nodes=50000):
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
        for layer in self.layers:
            h = layer(h, edge_attr=edge_attr)
        h = self.output_norm(h)
        return h, edge_attr

def huber_like_loss(pred, target, delta=1.0):
    diff = pred - target
    abs_diff = torch.abs(diff)
    quadratic = torch.minimum(abs_diff, torch.tensor(delta, device=diff.device))
    loss = 0.5 * quadratic**2 + delta * (abs_diff - quadratic) - 0.5 * (delta**2)
    mask = (abs_diff <= delta).float()
    huber_loss = 0.5 * (diff**2) * mask + (1 - mask)*(delta*(abs_diff - 0.5*delta))
    return huber_loss.mean()

class GraphormerJEPA(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_degree=128, max_nodes=50000, alpha=0.001):
        super(GraphormerJEPA, self).__init__()
        self.context_encoder = Graphormer(input_dim, hidden_dim, max_degree=max_degree, max_nodes=max_nodes)
        self.target_encoder = Graphormer(input_dim, hidden_dim, max_degree=max_degree, max_nodes=max_nodes)
        self.prediction_head = nn.Linear(hidden_dim, output_dim)
        self.pretrain_head = nn.Linear(hidden_dim, 2)
        self.alpha = alpha

    def forward(self, context_batch, target_batch, pretrain=False):
        context_embeddings, context_edge_attr = self.context_encoder(
            context_batch.x, context_batch.degree, context_batch.node_ids,
            edge_index=context_batch.edge_index, edge_attr=context_batch.edge_attr
        )
        target_embeddings, target_edge_attr = self.target_encoder(
            target_batch.x, target_batch.degree, target_batch.node_ids,
            edge_index=target_batch.edge_index, edge_attr=target_batch.edge_attr
        )

        if pretrain:
            D = huber_like_loss(context_embeddings, target_embeddings)
            return D
        else:
            predicted_scores = self.prediction_head(context_embeddings).squeeze(-1)
            if target_batch.x.dim() == 3:
                target_scores = target_batch.x[:, :, 2]
            elif target_batch.x.dim() == 2:
                target_scores = target_batch.x[:, 2].unsqueeze(0)
            else:
                raise ValueError("Unexpected dimension")
            if context_edge_attr is not None and context_edge_attr.dim() == 2:
                dist = context_edge_attr[:, 0]
                edge_graph_idx = context_batch.batch[context_batch.edge_index[0]]
                pred_probs = torch.sigmoid(predicted_scores)
                pred_mask = (pred_probs >= 0.5).float()
                mask1 = pred_mask.view(-1)[context_batch.edge_index[0]]
                mask2 = pred_mask.view(-1)[context_batch.edge_index[1]]
                edge_pred_mask = (mask1 * mask2).float()
                per_graph_dist = torch.bincount(edge_graph_idx, weights=dist * edge_pred_mask, minlength=context_batch.num_graphs)
                per_graph_edge_count = torch.bincount(edge_graph_idx, weights=edge_pred_mask, minlength=context_batch.num_graphs)
                per_graph_avg_dist = torch.where(
                    per_graph_edge_count > 0,
                    per_graph_dist / per_graph_edge_count,
                    torch.zeros_like(per_graph_dist)
                )
                spatial_loss = per_graph_avg_dist.mean() * self.alpha
            else:
                spatial_loss = 0.0
            return predicted_scores, target_scores, spatial_loss
