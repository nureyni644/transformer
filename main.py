import torch
from attention.scaled_dot_product import scaledDotProduct
def main():
    q = torch.Tensor([[2,3,4],[2,3,4]])#(1,3)
    k = torch.Tensor([[10,0,0],[0,10,0],[0,0,10],[0,0,10]]) #(4,3)
    # (1,3)*(3,4)
    v = torch.Tensor([[1,0],[10,0],[100,5],[1000,6]])

    print(scaledDotProduct(q,k,v))

# class MultiHeadAttention(nn.)


if __name__ == "__main__":
    main()
