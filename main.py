import torch
from attention.scaled_dot_product import scaledDotProduct, SelfAttention, MultiHeadAttention, CaulsalAttention
from embPosEncod.embedding import OSEmbedding
from attention.enoder import Encoder
from attention.decoder import Decoder
from attention.transformer import Transformer
def main():
   
    x =torch.LongTensor([[2,3,4,1],[0,3,1,2]])

    vocab_size = 1000
    d_model = 8
    num_heads = 4

    tr = Transformer(vocab_size=vocab_size,d_model=d_model,emb_dim=d_model,num_heads=num_heads)
    out = tr(x,x)
    print(f"""{
                out.size()}
            """)
if __name__ == "__main__":
    main()
