import os
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_lr_scheduler(optimizer,total_epochs,verbose=True):
    
    """
    Scheduler to linearly decrease learning rate,
    so thatlearning rate after the last epoch is 0.
    """

    lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=verbose)
    return lr_scheduler



def save_vocab(vocab):
    """Save vocab file to `model_dir` directory"""
    vocab_path = os.path.join(os.getcwd(), "vocab.pt")
    torch.save(vocab, vocab_path)
