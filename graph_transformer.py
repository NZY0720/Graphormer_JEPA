import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree

class GraphTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super(GraphTransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, x):
        # x: [num_nodes * num_time_steps, embed_dim]
        x = self.transformer_encoder(x)
        return x

class GraphTransformer(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, num_layers, dropout=0.1):
        super(GraphTransformer, self).__init__()
        self.node_embed = nn.Linear(in_channels, embed_dim)
        self.pos_embed = nn.Embedding(1000, embed_dim)  # 假设最大节点数不超过1000
        self.transformer_encoder = GraphTransformerEncoder(embed_dim, num_heads, num_layers, dropout)
        self.embed_dim = embed_dim

    def forward(self, x, edge_index):
        """
        x: [num_nodes, num_time_steps, in_channels]
        edge_index: [2, num_edges]
        """
        num_nodes = x.size(0)
        num_time_steps = x.size(1)
        in_channels = x.size(2)

        # 节点嵌入
        x = x.reshape(num_nodes * num_time_steps, in_channels)
        node_feat = self.node_embed(x)  # [num_nodes * num_time_steps, embed_dim]
        
        # 位置编码（使用度信息）
        degrees = degree(edge_index[0], num_nodes=num_nodes).long()  # [num_nodes]
        degrees = degrees.unsqueeze(1).repeat(1, num_time_steps).reshape(-1)  # 展开到时间维度
        pos_feat = self.pos_embed(degrees)  # [num_nodes * num_time_steps, embed_dim]
        
        # 合并嵌入
        x = node_feat + pos_feat  # [num_nodes * num_time_steps, embed_dim]
        
        # 前向传播
        x = self.transformer_encoder(x)  # [num_nodes * num_time_steps, embed_dim]
        
        # 恢复形状
        x = x.reshape(num_nodes, num_time_steps, self.embed_dim)
        
        return x
