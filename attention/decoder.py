import torch
from torch import nn
from embPosEncod.embedding import OSEmbedding
from .scaled_dot_product import MultiHeadAttention, CaulsalAttention

class Block(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Block, self).__init__()
        self.m_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm = nn.LayerNorm(normalized_shape=d_model)
        self.ffn = nn.Linear(in_features=d_model, out_features=d_model)
class Decoder(nn.Module):
    
    def __init__(self,voc_size,emb_dim,d_model,num_heads):
        super(Decoder,self).__init__()
        self.emb = OSEmbedding(voc_size, emb_dim)
        self.m_attn = MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.c_attn = CaulsalAttention(d_model=d_model,num_heads=num_heads)
        self.Norm = nn.LayerNorm(normalized_shape=d_model)
        self.ffn = nn.Linear(d_model,d_model)
    def forward(self,x_dec,out_en):
        
        x_dec = self.emb(x_dec)
        x_dec = self.m_attn(x_dec,x_dec,x_dec)
        # print(x_dec.size())
        x_dec = self.c_attn(x_dec,out_en,out_en)
        x_dec = self.Norm(x_dec) + x_dec
        x_dec = self.ffn(x_dec)
        return x_dec
