from torch import nn

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

x = [[1,2,3,4,4]]
import torch
x = torch.LongTensor(x)
print(f"shape de X : {x.shape}")
emb = OSEmbedding(
    voc_size=5,
    emb_dim=3
)
x= emb(x)
print(f"shape de X : {x.shape}")

