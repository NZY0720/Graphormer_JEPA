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

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        Q = self.q_proj(x) 
        K = self.k_proj(x) 
        V = self.v_proj(x) 

        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # [B, h, N, N]
        attn = F.softmax(scores, dim=-1)  # [B, h, N, N]
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
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim*4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        h = self.attn(h)
        x = x + self.dropout(h)

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)
        return x

class Graphormer(nn.Module):
    """
    仅使用节点度作为结构特征:
    - 节点特征包含: 输入特征 + 节点度嵌入 + 节点位置嵌入
    """
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

    def forward(self, x, degree, node_ids):
        # x: [B, N, input_dim] 或 [N, D]
        # degree: [B, N] 或 [N]
        # node_ids: [B, N] 或 [N]
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, N, D]
            degree = degree.unsqueeze(0)  # [1, N]
            node_ids = node_ids.unsqueeze(0)  # [1, N]

        B, N, _ = x.size()
        h = self.input_proj(x)  # [B, N, D]
        deg_embed = self.degree_embedding(degree)  # [B, N, D]
        h = h + deg_embed  # [B, N, D]

        pos_embed = self.pos_embedding(node_ids)  # [B, N, D]
        h = h + pos_embed  # [B, N, D]

        for layer in self.layers:
            h = layer(h)  # [B, N, D]

        h = self.output_norm(h)  # [B, N, D]
        return h

class GraphormerJEPA(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_degree=128, max_nodes=50000):
        super(GraphormerJEPA, self).__init__()
        self.context_encoder = Graphormer(input_dim, hidden_dim, max_degree=max_degree, max_nodes=max_nodes)
        self.target_encoder = Graphormer(input_dim, hidden_dim, max_degree=max_degree, max_nodes=max_nodes)
        self.prediction_head = nn.Linear(hidden_dim, output_dim)  # 输出 logits

    def forward(self, context_batch, target_batch):
        # context_batch 和 target_batch 形状: [B, N, D]
        context_embeddings = self.context_encoder(context_batch.x, context_batch.degree, node_ids=context_batch.node_ids)  # [B, N, D]
        target_embeddings = self.target_encoder(target_batch.x, target_batch.degree, node_ids=target_batch.node_ids)  # [B, N, D]

        # 假设 target_batch 是 context_batch 的克隆，因此可以直接使用 context_embeddings
        combined_embeddings = context_embeddings  # 或者其他结合方式，如 context_embeddings + target_embeddings

        predicted_scores = self.prediction_head(combined_embeddings).squeeze(-1)  # [B, N] logits

        # 获取真实标签，假设 'has_charging_station' 是节点特征的第三个元素（索引2）
        if target_batch.x.dim() == 3:
            target_scores = target_batch.x[:, :, 2]  # [B, N]
        elif target_batch.x.dim() == 2:
            target_scores = target_batch.x[:, 2].unsqueeze(0)  # [1, N]
        else:
            raise ValueError("Unexpected dimensions for target_batch.x")

        return predicted_scores, target_scores
