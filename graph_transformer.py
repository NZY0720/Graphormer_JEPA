import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(GraphTransformerLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_index=None):
        """
        x: [N, E], 节点嵌入
        edge_index: [2, E], 边索引（当前未使用）
        """
        N, E = x.size()
        x = x.unsqueeze(1)  # [N, E] -> [N, 1, E]
        x = x.transpose(0, 1)  # [N, 1, E] -> [1, N, E]

        if edge_index is not None:
            # 创建注意力掩码，仅允许存在边的节点之间进行注意力
            mask = torch.zeros((N, N), dtype=torch.bool, device=x.device)
            mask[edge_index[0], edge_index[1]] = True
            attn_mask = ~mask  # Invert mask: True where no edge
        else:
            attn_mask = None

        # 自注意力机制
        attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask)  # [1, N, E], [1, N, N]
        attn_output = self.dropout_layer(attn_output)
        x = x + attn_output
        x = self.norm1(x)

        # 前馈网络（MLP）
        mlp_output = self.linear2(F.relu(self.linear1(x)))  # [1, N, E]
        mlp_output = self.dropout_layer(mlp_output)
        x = x + mlp_output
        x = self.norm2(x)

        # 转换回[N, E]
        x = x.transpose(0, 1).squeeze(1)  # [1, N, E] -> [N, E]

        return x

class GraphTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.0):
        super(GraphTransformer, self).__init__()
        self.layers = nn.ModuleList([
            GraphTransformerLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, edge_index=None):
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.norm(x)
        return x
