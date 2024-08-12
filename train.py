import torch
import Test
import data_format
import torch.nn as nn
import numpy as np
import math
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k

train_data, validation_data, test_data = Multi30k(language_pair=('en', 'de'))


english_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
german_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')


# Build vocabularies
src_vocab = data_format.build_vocabulary(train_data, english_tokenizer, index= 0)
tgt_vocab = data_format.build_vocabulary(train_data, german_tokenizer, index= 1)


batch_size = 128

train_dataloader, validation_dataloader = data_format.get_loaders(train_data, validation_data, src_vocab, tgt_vocab, english_tokenizer, german_tokenizer, batch_size)

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)
src_seq_length = max(len(sample[0]) for sample in train_data)
tgt_seq_length = max(len(sample[1]) for sample in train_data)

model = Test.roll_out(src_vocab_size, tgt_vocab_size, src_seq_length, tgt_seq_length, layers=6)

criterion = nn.CrossEntropyLoss(ignore_index=0)

optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, dataloader, optimizer, criterion, clip=1.0):
    model.train()
    total_loss = 0

    for src, tgt in dataloader:
        src = src.transpose(0, 1)  # Transpose to match the model's expected input shape
        tgt = tgt.transpose(0, 1)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt_input != 0).unsqueeze(1).unsqueeze(3)
        tgt_mask = tgt_mask & torch.tril(torch.ones(tgt_mask.shape[2], tgt_mask.shape[3])).bool()

        optimizer.zero_grad()

        output = model(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(output.contiguous().view(-1, output.shape[-1]), tgt_output.contiguous().view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, optimizer, criterion)
    print(f"Epoch: {epoch + 1}, Train loss: {train_loss:.4f}")






