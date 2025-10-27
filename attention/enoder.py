import torch 
from torch import nn
from embPosEncod.embedding import OSEmbedding
from .scaled_dot_product import MultiHeadAttention

class Block(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Block, self).__init__()
        self.m_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm = nn.LayerNorm(normalized_shape=d_model)
        self.ffn = nn.Linear(in_features=d_model, out_features=d_model)


    def forward(self, x):
        x = self.m_attn(x, x, x)
        x = self.norm(x) + x
        x = self.ffn(x)
        return x
    

class Encoder(nn.Module):

    def __init__(self,vocab_size,d_model,num_heads, num_layers=6):
        super(Encoder,self).__init__()

        self.emb = OSEmbedding(voc_size= vocab_size, emb_dim=d_model)
        self.layers = nn.ModuleList([Block(d_model=d_model, num_heads=num_heads) for _ in range(num_layers)])
        self.act = nn.Sigmoid()
    def forward(self,x):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.act(x)
        return x
    
