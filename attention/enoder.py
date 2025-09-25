import torch 
from torch import nn
from embPosEncod.embedding import OSEmbedding
from .scaled_dot_product import MultiHeadAttention
class Encoder(nn.Module):

    def __init__(self,vocab_size,d_model,num_heads ):
        super(Encoder,self).__init__()

        self.emb = OSEmbedding(voc_size= vocab_size, emb_dim=d_model)
        self.m_attn = MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.norm =  nn.LayerNorm(d_model)
        self.ffn = nn.Linear(in_features=d_model,out_features=d_model)
        self.act = nn.Sigmoid()
    def forward(self,x):
        
        x = self.emb(x)
        x = self.m_attn(x,x,x)
        x = self.norm(x)
        x = self.ffn(x)
        x = self.act(x)
        return x

