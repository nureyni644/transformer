import torch
from torch import nn
import math
def scaledDotProduct(q,k,v,mask=None):

    qk_matmul = torch.matmul(q,k.transpose(-2,-1))
    # scaled
    dk = torch.scalar_tensor(k.shape[-1])
    scaled_attenation_logits = qk_matmul / torch.sqrt(dk)
    if mask is not None:
        scaled_attenation_logits += (mask* -1e9)
    
    attention_weights = torch.softmax(scaled_attenation_logits,axis=-1)
    output = torch.matmul(attention_weights,v)
    return attention_weights,output
    

class SelfAttention(torch.nn.Module):
    
    def __init__(self,d_model):
        super(SelfAttention,self).__init__()
        self.depth = d_model

        self.wq = nn.Linear(in_features=d_model,out_features=d_model)
        self.wk = nn.Linear(in_features=d_model,out_features=d_model)
        self.wv = nn.Linear(in_features=d_model,out_features=d_model)

    def forward(self,x):

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        print(f"Q :{q.shape}\n K : {k.shape}\n V : {v.shape}")
        attention_weight , output = scaledDotProduct(q=q,k=k,v=v)
        return attention_weight , output
    
   

class MultiHeadAttention(nn.Module):

    def __init__(self,d_model, num_heads):
        super(MultiHeadAttention,self).__init__()

        assert d_model % num_heads == 0, "d_model doit etre divisible par hum_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o =nn.Linear(d_model,d_model)
    
    def scaled_dot_product_attention(self,Q,K,V,mask = None):

        qk_matmul = torch.matmul(Q,K.transpose(-2,-1))

        scaled_attenation_logits = qk_matmul / math.sqrt(self.d_k)
        if mask is not None:
            scaled_attenation_logits += (mask* -1e9)
        
        attention_weights = torch.softmax(scaled_attenation_logits,axis=-1)
        output = torch.matmul(attention_weights,V)
        # output.view()
        return attention_weights,output
    
    def split_heads(self,x):
        batch_size ,seq_len,_ = x.size()
        x = x.view(batch_size,seq_len, self.num_heads,self.d_k).transpose(1,2)
        
        return x
    def combine_heads(self,x):
        batch_size, num_head,seq_len,d_k = x.size()
        # print(f"In Combine heads:{x.size()}")
        x = x.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model)
        # print(f"End Combine heads:{x.size()}")
        return x

    def forward(self, query, key, value,pad_mask = None):
        
        Q = self.split_heads(self.w_q(query))
        K = self.split_heads(self.w_k(key))
        V = self.split_heads(self.w_v(value))
        attention_weights,output =self.scaled_dot_product_attention(Q,K,V,pad_mask)
        output = self.combine_heads(output)
        return output


# mha = MultiHeadAttention(d_model=4,num_heads=2)
