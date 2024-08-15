import torch
import pandas as pd
import torch.nn as nn
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

    for data in iter:
        yield tokenizer(data['translation'][lang])


def build_vocabulary(iter, tokenizer, lang):
    vocab = build_vocab_from_iterator( yield_tokens(iter, tokenizer, lang), min_freq=2, specials=["<pad>", "<unk>", "<bos>", "<eos>"], special_first=True)
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def generate_batch(data_batch):
    src_batch, tgt_batch = zip(*data_batch)

    return torch.stack(src_batch), torch.stack(tgt_batch)


def get_loader(data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, batch_size, src_lang, tgt_lang, max_length):
    def data_process(data_sample):
        src = torch.tensor([src_bos] + [src_vocab[token] for token in src_tokenizer(data_sample['translation'][src_lang])] + [src_eos])
        tgt = torch.tensor([tgt_bos] + [tgt_vocab[token] for token in tgt_tokenizer(data_sample['translation'][tgt_lang])] + [tgt_eos])

        src = nn.ConstantPad1d((0, max_length - len(src)), src_pad)(src)
        tgt = nn.ConstantPad1d((0, max_length - len(tgt)), tgt_pad)(tgt)
        src = src.squeeze(0)
        tgt = tgt.squeeze(0)



        return src, tgt

    prep_data = [data_process(data_sample) for data_sample in data]


    dataloader = DataLoader(prep_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)


    return dataloader
