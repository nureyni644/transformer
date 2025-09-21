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
    
    def forward(self,x):
        return self.emb(x)

# tester

# x = [[1,2,3,4,4]]
# x = torch.LongTensor(x)
# print(f"shape de X : {x.shape}")
# emb = OSEmbedding(
#     voc_size=5,
#     emb_dim=3
# )
# x= emb(x)
# print(f"shape de X : {x.shape}")

def getPositionalEncoding(seq_lenght,d,n=10_000):
    print(int(d/2))
    P = torch.zeros((seq_lenght,d))
    for k in range(seq_lenght):
        for i in np.arange(int(d/2)):
            print((k,2*i),(k,2*i +1))
            denominator = np.power(n,2*i/d)
            P[k,2*i] = np.sin(k/denominator)
            P[k,2*i+1] = np.cos(k/denominator)
    return P
print(getPositionalEncoding(4,4,n=100))