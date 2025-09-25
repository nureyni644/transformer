import torch
from attention.scaled_dot_product import scaledDotProduct, SelfAttention, MultiHeadAttention
from embPosEncod.embedding import OSEmbedding
from attention.enoder import Encoder
def main():

   
    x = torch.LongTensor([[2,3,4],[0,3,1]])
    # print(x)
    # m_attn = MultiHeadAttention(d_model=4, num_heads=2)
    # attn = SelfAttention(d_model=4)
    # emb = OSEmbedding(voc_size=5,emb_dim=4)
    # print(f"start====>\n")
    # x = emb(x)
    # print(f"end ======>\n")
    # print(x.size())
    # output = m_attn(x,x,x)
    # print(f"\n size of output : {output.size()}")
    # seq_len = 4
    # mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool()
    
    # print(mask)
    Voc_size = 20
    d_model = 8
    num_heads = 4

    enc = Encoder(vocab_size=Voc_size,d_model=d_model,num_heads=num_heads)

    x = enc(x)
    print(f"Apres ffn :{x.shape}\n")
if __name__ == "__main__":
    main()
