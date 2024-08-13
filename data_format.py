import torch
import torchtext;torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


# Special tokens
src_pad = 0
trg_pad = 0
src_bos = 2
trg_bos = 2
src_eos = 3
trg_eos = 3

def yield_tokens(iter, tokenizer,index):
    for data in iter:
        yield tokenizer(data[index])


def build_vocabulary(data_iter, tokenizer, index):
    vocab = build_vocab_from_iterator(
        yield_tokens(data_iter, tokenizer, index),
        min_freq=2,
        specials=["<pad>", "<unk>", "<bos>", "<eos>"],
        special_first=True
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab

def data_process(data_sample, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer):
    src = [src_bos] + [src_vocab[token] for token in src_tokenizer(data_sample[0])] + [src_eos]
    tgt = [trg_bos] + [tgt_vocab[token] for token in tgt_tokenizer(data_sample[1])] + [trg_eos]
    return torch.tensor(src), torch.tensor(tgt)

def generate_batch(data_batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in data_batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
    src_batch = pad_sequence(src_batch, padding_value=src_pad)
    tgt_batch = pad_sequence(tgt_batch, padding_value=trg_pad)
    return src_batch, tgt_batch




def get_loaders(iter, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, batch_size):

    data = [data_process(data_sample, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer) for data_sample in iter]
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)

    return dataloader
