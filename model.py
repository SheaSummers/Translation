#Importing needed packages
import torch
import torch.nn as nn
import numpy as np
import math



class Embedding (nn.Module):
    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.vocab = vocab
        self.embedding = nn.Embedding(vocab, d_model)
    def forward(self, x):
        return self.embedding(x) * np.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout =0.1 ):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        positional_matrix = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        dinominator = torch.exp(torch.arange(0, d_model, 2).unsqueeze(0)* -(math.log(10000.0) / d_model))

        positional_matrix[:, 0::2] = torch.sin(position * dinominator)
        positional_matrix[:, 1::2] = torch.cos(position * dinominator)

        positional_matrix= positional_matrix.unsqueeze(0)

        self.register_buffer('positional_matrix', positional_matrix)

    def forward(self, x):
        x = x + self.positional_matrix[:, :x.size(1), :].requires_grad_(False)

        return self.dropout(x)


class LayerNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a * (x - mean) / torch.sqrt(std**2 + self.eps) + self.b

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1(x)

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.shape[-1]

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = torch.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    return torch.matmul(scores, value), scores



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.heads = heads

        assert d_model % heads == 0

        self.d_k = d_model // heads
        self.dropout = nn.Dropout(dropout)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        Q = self.q(query)
        K = self.k(key)
        V = self.v(value)

        Q = Q.view(Q.shape[0], Q.shape[1], self.heads, self.d_k).transpose(1, 2)

        K = K.view(K.shape[0], K.shape[1], self.heads, self.d_k).transpose(1, 2)

        V = V.view(V.shape[0], V.shape[1], self.heads, self.d_k).transpose(1, 2)

        x, self.scores = attention(Q, K, V, mask=mask)

        x = x.transpose(1, 2)

        x = x.contiguous().view(x.shape[0], x.shape[1], self.d_model)

        return self.out(x)

class SkipConnection(nn.Module):
    def __init__(self, dropout):
        super(SkipConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        x = sublayer(x)
        x = self.norm(x)
        x = self.dropout(x)

        return x

class EncoderBlock(nn.Module):
    def __init__(self, attention, feed_forward, dropout):
        super(EncoderBlock, self).__init__()
        self.attention = attention
        self.feed_forward = feed_forward
        self.skip = nn.ModuleList([SkipConnection(dropout), SkipConnection(dropout)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.skip[0](x, lambda x: self.attention(x, x, x, mask=mask))
        x = self.skip[1](x, lambda x: self.feed_forward(x))

        return x

class Encoder(nn.Module):
    def __init__(self, layers):
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, attention,cross_attention, feed_forward, dropout):
        super(DecoderBlock, self).__init__()
        self.attention = attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.ModuleList([SkipConnection(dropout), SkipConnection(dropout), SkipConnection(dropout)])


    def forward(self, x, enc_out,  enc_mask, dec_mask):
        x = self.skip[0](x, lambda x: self.attention(x, x, x, mask=dec_mask))
        x = self.skip[1](x, lambda x: self.cross_attention(x, enc_out, enc_out, mask=enc_mask))
        x = self.skip[2](x, lambda x: self.feed_forward(x))

        return x

class Decoder(nn.Module):
    def __init__(self, layers):
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, enc_out, enc_mask, dec_mask):
        for layer in self.layers:
            x = layer(x, enc_out, enc_mask, dec_mask)
        return self.norm(x)


class Projection(nn.Module):
    def __init__(self, d_model, vocab):
        super(Projection, self).__init__()
        self.d_model = d_model
        self.vocab = vocab
        self.project = nn.Linear(d_model, vocab)

    def forward(self, x):
        x = self.project(x)
        return x

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, source, target, src_pos, trg_pos, project):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source = source
        self.target = target
        self.src_pos = src_pos
        self.trg_pos = trg_pos
        self.project = project

    def encode(self, x, mask):
        x = self.source(x)
        x = self.src_pos(x)
        return self.encoder(x, mask)

    def decode(self, x, enc_out, src_mask, trg_mask):
        x = self.target(x)
        x = self.trg_pos(x)
        return self.decoder(x, enc_out, src_mask, trg_mask)

    def projection(self, x):
        return self.project(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.projection(dec_out)

def roll_out(src_vocab, trg_vocab, src_seq, trg_seq, layers, d_model = 512, heads = 8, dropout = 0.1, d_ff = 2048):
    src_embed = nn.Embedding(src_vocab, d_model)
    trg_embed = nn.Embedding(trg_vocab, d_model)

    src_pos = PositionalEncoding(d_model, max_len = src_seq, dropout = dropout)
    trg_pos = PositionalEncoding(d_model, max_len = trg_seq, dropout = dropout)

    encoders = []
    for i in range(layers):
        e_attention = MultiHeadAttention(d_model, heads = heads, dropout = dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout = dropout)
        encoder = EncoderBlock(e_attention, feed_forward, dropout)
        encoders.append(encoder)

    decoders = []
    for i in range(layers):
        d_attention = MultiHeadAttention(d_model, heads = heads, dropout = dropout)
        cross_attention = MultiHeadAttention(d_model, heads = heads, dropout = dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout = dropout)
        decoder = DecoderBlock(d_attention, cross_attention, feed_forward, dropout)
        decoders.append(decoder)

    All_Encoders = Encoder(encoders)
    All_Decoders = Decoder(decoders)

    project = Projection(d_model, trg_vocab)

    transformer = Transformer(All_Encoders, All_Decoders, src_embed, trg_embed, src_pos, trg_pos, project)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer



























