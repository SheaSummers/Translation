import torch
import torchtext;torchtext.disable_torchtext_deprecation_warning()
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader


# Special tokens
src_pad = 0
tgt_pad = 0
src_bos = 2
tgt_bos = 2
src_eos = 3
tgt_eos = 3

def yield_tokens(iter, tokenizer,lang):

    for data in iter[lang]:
        yield tokenizer(data)


def build_vocabulary(iter, tokenizer, lang):
    vocab = build_vocab_from_iterator( yield_tokens(iter, tokenizer, lang), min_freq=2, specials=["<pad>", "<unk>", "<bos>", "<eos>"], special_first=True)
    vocab.set_default_index(vocab["<unk>"])
    return vocab

def data_process(data_sample, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer):

    print(data_sample[0])

    src = [src_bos] + [src_vocab[token] for token in src_tokenizer(data_sample[0])] + [src_eos]
    tgt = [tgt_bos] + [tgt_vocab[token] for token in tgt_tokenizer(data_sample[1])] + [tgt_eos]
    return torch.tensor(src), torch.tensor(tgt)

def generate_batch(data_batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in data_batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
    src_batch = pad_sequence(src_batch, padding_value=src_pad, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=tgt_pad, batch_first=True)
    return src_batch, tgt_batch




def get_loader(data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, batch_size):

    prep_data = [data_process(data_sample, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer) for data_sample in data]
    dataloader = DataLoader(prep_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)


    return dataloader
