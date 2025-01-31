import torch
import torch.nn as nn
import torch.functional as F


class ConvNormBlock(nn.Module):
    def __init__(
            self, 
            stride: int, 
            kernel_size: int, 
            embed_dim: int, 
            dropout: float
        ):
        
        ## Why 1D Convolutions tho? 
        self.conv_layer = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            dilation=1
        )
        self.bn_layer = nn.BatchNorm1d(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn_layer(x)
        x = F.relu
        x = self.dropout(x)

        return x
