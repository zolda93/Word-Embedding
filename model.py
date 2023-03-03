import torch
import torch.nn as nn
from torch import Tensor



class CBOW(nn.Module):

    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """

    def __init__(self,vocab_size:int,embed_dim:int = 300):

        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_dim)
        self.linear = nn.Linear(embed_dim,vocab_size)

    def forward(self,x:Tensor)->Tensor:

        x = self.embedding(x)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


class SkipGram(nn.Module):

    """
    Implementation of skip-gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """

    def __init__(self,vocab_size:int,embed_dim:int=300):

        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_dim)
        self.linear = nn.Linear(embed_dim,vocab_size)

    def forward(self,x:Tensor)->Tensor:
        x = self.embedding(x)
        x = self.linear(x)
        return x





