# graph_transformer.py

import torch
import torch.nn as nn

class GraphTransformer(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, num_layers, dropout=0.1):
        """
        初始化 GraphTransformer 模块。

        Args:
            in_channels (int): 输入特征的维度。
            embed_dim (int): Transformer 的嵌入维度。
            num_heads (int): 多头注意力机制的头数。
            num_layers (int): Transformer 编码器的层数。
            dropout (float): Dropout 概率。
        """
        super(GraphTransformer, self).__init__()
        self.embedding = nn.Linear(in_channels, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # 保持批次维度在第一维
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        """
        前向传播。

        Args:
            x (torch.Tensor): 节点特征，形状为 [num_nodes, num_time_steps, in_channels]。
            edge_index (torch.Tensor): 边索引，形状为 [2, num_edges]。

        Returns:
            torch.Tensor: Transformer 输出，形状为 [num_nodes, num_time_steps, embed_dim]。
        """
        x = self.embedding(x)  # [num_nodes, num_time_steps, embed_dim]
        x = self.dropout(x)
        x = self.transformer_encoder(x)  # [num_nodes, num_time_steps, embed_dim]
        return x
