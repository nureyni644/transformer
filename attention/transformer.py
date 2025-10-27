import torch
from torch import nn
from .enoder import Encoder
from .decoder import Decoder
class Transformer(nn.Module):

    def __init__(self,vocab_size,d_model,emb_dim,num_heads):
        super(Transformer,self).__init__()

        self.encoder = Encoder(vocab_size=vocab_size,d_model=d_model,num_heads=num_heads)
        self.decoder = Decoder(voc_size=vocab_size,emb_dim=emb_dim,d_model=d_model,num_heads=num_heads)
        self.ffn = nn.Linear(d_model,vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x_d,x_e):

        x_e = self.encoder(x_e)
        x_d = self.decoder(x_d,x_e)
        batch, seq_len, d = x_d.size()
        
        x_d = self.ffn(x_d)
        
        x_d = self.softmax(x_d)

        return x_d