# graph_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

###################################################
# 1. Graphormer Module
###################################################

class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, edge_attr=None):
        """
            x (Tensor):  [B, N, D] => batch_size, number of nodes, dimension of emdedding
            edge_attr (Tensor, optional): feature of edges

        return:
            Tensor:  [B, N, D]
        """
        B, N, D = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out


class GraphormerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = GraphormerMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
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
    """
    Graphormer Encoder:
      input: (B, N, input_dim)
      output: (B, N, hidden_dim)
    """
    def __init__(self, input_dim, hidden_dim, num_heads=4, num_layers=4,
                 dropout=0.1, max_degree=128, max_nodes=50000):
        super().__init__()
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
        """
        x: (B, N, input_dim)
        degree: (B, N)
        node_ids: (B, N)
        """
        if x.dim() == 2:  # if batch_size=1, (N, input_dim)
            x = x.unsqueeze(0)
            degree = degree.unsqueeze(0)
            node_ids = node_ids.unsqueeze(0)

        B, N, _ = x.shape

        # 1) project
        h = self.input_proj(x)
        # 2) degree embedding
        deg_embed = self.degree_embedding(torch.clamp(degree, max=self.max_degree - 1))
        h = h + deg_embed
        # 3) node position embedding
        node_ids_clamped = torch.clamp(node_ids, max=self.max_nodes - 1)
        pos_embed = self.pos_embedding(node_ids_clamped)
        h = h + pos_embed

        # 4) layer pile
        for layer in self.layers:
            h = layer(h, edge_attr=edge_attr)

        # 5) norm
        h = self.output_norm(h)

        return h


###################################################
# 2. Huber-like Loss 
###################################################
def huber_like_loss(pred, target, delta=1.0):
    """
    Huber:
      if |diff| <= delta => 0.5 * diff^2
      else |diff| > delta  => delta*(|diff|-0.5*delta)
    """
    diff = pred - target
    abs_diff = torch.abs(diff)
    mask = (abs_diff <= delta).float()

    huber_part = 0.5 * (diff**2) * mask
    linear_part = delta * (abs_diff - 0.5 * delta) * (1 - mask)
    return (huber_part + linear_part).mean()


###################################################
# 3. GraphormerJEPA
###################################################
class GraphormerJEPA(nn.Module):
    """
      - if pretrain, return D Loss
      - else (downstream tasks), return elements needed

    JEPA:
      1) context embedding => (B, N_c, hidden_dim)
      2) target embedding  => (B, N_t, hidden_dim)
      3) mean-pool => (B, hidden_dim)  => huber_like_loss

    Downstream Tasks:
      - context embedding => (B, N, hidden_dim)
      - predictor_head => (B, N)
      - return prediction embedding
    """
    def __init__(self, input_dim, hidden_dim,
                 max_degree=128, max_nodes=50000,
                 num_heads=4, num_layers=4, dropout=0.1, delta=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.delta = delta

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

        # predictor: align context embedding and target embedding
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # prediction_head for task1 (B, N, hidden_dim) => (B, N)
        self.prediction_head = nn.Linear(hidden_dim, 1)

    def forward(self, context_batch, target_batch, pretrain=False):
        """
        context_batch, target_batch: PyG-like Data
        """
        # context embedding
        context_h = self.context_encoder(
            context_batch.x, context_batch.degree, context_batch.node_ids,
            edge_index=context_batch.edge_index, edge_attr=context_batch.edge_attr
        )  # => (B, N_c, hidden_dim)

        if pretrain:
            # 1) target embedding => (B, N_t, hidden_dim)
            target_h = self.target_encoder(
                target_batch.x, target_batch.degree, target_batch.node_ids,
                edge_index=target_batch.edge_index, edge_attr=target_batch.edge_attr
            )
            # 2) mean-pool => (B, hidden_dim)
            context_rep = context_h.mean(dim=1)
            target_rep = target_h.mean(dim=1)
            # 3) predictor => (B, hidden_dim)
            predicted_target = self.predictor(context_rep)
            # 4) huber-like loss
            loss = huber_like_loss(predicted_target, target_rep, self.delta)
            return loss
        else:
            # task1 => (B, N_c, hidden_dim)
            # => self.prediction_head => (B, N_c, 1)
            predicted_scores = self.prediction_head(context_h).squeeze(-1)
            # => (B, N_c)
            return predicted_scores
