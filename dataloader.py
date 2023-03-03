import os
import torch
from functools import partial
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2, WikiText103


CBOW_N_WORDS = 4
SKIPGRAM_N_WORDS = 4
MIN_WORD_FREQUENCY = 50
MAX_SEQUENCE_LENGTH = 256

tokenizer = get_tokenizer("basic_english", language="en")

def data_iterator(name,data_type):
    if name == 'WikiText2':
        data = WikiText2(root=os.getcwd(),split=(data_type))
    elif name == 'WikiText103':
        data = WikiText103(root=os.getcwd(),split=(data_type))
    else:
        raise ValueError("Choose dataset from: WikiText2, WikiText103")

    data = to_map_style_dataset(data)
    return data

def build_vocab(data):

    vocab = build_vocab_from_iterator(map(tokenizer,data),
                                    specials = ["<unk>"],
                                    min_freq=MIN_WORD_FREQUENCY)
    vocab.set_default_index(vocab["<unk>"])
    return vocab



def collate_cbow(batch,text_pipeline):

    input_batch,output_batch = [],[]

    for text in batch:
        token_ids = text_pipeline(text)

        if len(token_ids) < CBOW_N_WORDS * 2 + 1:continue

        if MAX_SEQUENCE_LENGTH:
            token_ids = token_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(token_ids) - CBOW_N_WORDS*2):
            token_id_sequence = token_ids[idx:(idx + CBOW_N_WORDS*2 + 1)]
            output = token_id_sequence.pop(CBOW_N_WORDS)
            input_ = token_id_sequence
            input_batch.append(input_)
            output_batch.append(output)

    input_batch = torch.tensor(input_batch,dtype=torch.long)
    output_batch = torch.tensor(output_batch,dtype=torch.long)
    return input_batch,output_batch


def collate_skipgram(batch, text_pipeline):
    

    input_batch, output_batch = [], []
    for text in batch:
        tokens_ids = text_pipeline(text)

        if len(tokens_ids) < SKIPGRAM_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            tokens_ids = tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(tokens_ids) - SKIPGRAM_N_WORDS * 2):
            token_id_sequence = tokens_ids[idx : (idx + SKIPGRAM_N_WORDS * 2 + 1)]
            input_ = token_id_sequence.pop(SKIPGRAM_N_WORDS)
            outputs = token_id_sequence

            for output in outputs:
                input_batch.append(input_)
                output_batch.append(output)

    input_batch = torch.tensor(input_batch, dtype=torch.long)
    output_batch = torch.tensor(output_batch, dtype=torch.long)
    return input_batch,output_batch



def data_loader_and_vocab(model_name,data_name,data_type,batch_size,shuffle,vocab=None):
    
    data = data_iterator(data_name,data_type)

    if not vocab:
        vocab = build_vocab(data)

    text_pipeline = lambda x:vocab(tokenizer(x))

    collate_fn = collate_cbow if model_name == "cbow" else collate_skipgram

    dataloader = DataLoader(data,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=partial(collate_fn,text_pipeline=text_pipeline))
    return dataloader,vocab






