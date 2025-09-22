import torch

def scaledDotProduct(q,k,v,mask=None):

    qk_matmul = torch.matmul(q,k.T)
    # scaled
    dk = torch.scalar_tensor(k.shape[-1])
    scaled_attenation_logits = qk_matmul / torch.sqrt(dk)
    if mask is not None:
        scaled_attenation_logits += (mask* -1e9)
    
    attention_weights = torch.softmax(scaled_attenation_logits,axis=-1)
    output = torch.matmul(attention_weights,v)
    return attention_weights,output
    

class MultiHeadAttention(torch.nn.Module):

    def __init__(self,n_head,d_model):
        super().__init__()