from torch import nn
import numpy as np
import torch
class OSEmbedding(nn.Module):
    
    def __init__(self,voc_size,emb_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb = nn.Embedding(
            num_embeddings= voc_size,
            embedding_dim=emb_dim
        )
        self.d_model = emb_dim
    
    def getPositionalEncoding(self,seq_lenght,d,n=10_000):
        # print(int(d/2))
        P = torch.zeros((seq_lenght,d))
        for k in range(seq_lenght):
            for i in np.arange(int(d/2)):
                # print((k,2*i),(k,2*i +1))
                denominator = np.power(n,2*i/d)
                P[k,2*i] = np.sin(k/denominator)
                P[k,2*i+1] = np.cos(k/denominator)
        return P
    
    def forward(self,x):
        seq = x.size()[1]
        x = self.emb(x)
        print(x)
        x +=self.getPositionalEncoding(seq_lenght=seq,d=self.d_model)
        return x

