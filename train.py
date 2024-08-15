import torch
import model
import data_format
import torch.nn as nn
import numpy as np
import math
import spacy
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torch.optim.lr_scheduler import MultiStepLR
from datasets import load_dataset
from weasel.util import validation

train_data = load_dataset("wmt/wmt14", "de-en", split="train")
validation_data = load_dataset("wmt/wmt14", "de-en", split="validation")
test_data = load_dataset("wmt/wmt14", "de-en", split="test")




src_lang = 'de'
tgt_lang = 'en'


tgt_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
src_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')



src_vocab = data_format.build_vocabulary(test_data , src_tokenizer, lang= 'de')
tgt_vocab = data_format.build_vocabulary(test_data, tgt_tokenizer, lang= 'en')






batch_size = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)
src_seq_length = max([len(sample['translation'][src_lang].split(' ')) for sample in test_data])
tgt_seq_length = max([len(sample['translation'][tgt_lang].split(' ')) for sample in test_data])
max_length = max(src_seq_length, tgt_seq_length)

train_dataloader = data_format.get_loader(train_data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, batch_size, src_lang, tgt_lang, max_length)

validation_dataloader = data_format.get_loader(validation_data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, batch_size, src_lang, tgt_lang, max_length)

test_dataloader = data_format.get_loader(test_data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, batch_size, src_lang, tgt_lang, max_length)



model = model.roll_out(src_vocab_size, tgt_vocab_size, max_length, layers=6)

model = model.to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.01)

optimizer = optim.Adam(model.parameters(), lr=0.1)



scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4000,80000,120000, 1600000], gamma=0.1)



def train(model, dataloader, optimizer, criterion, clip=1.0):
    model.train()
    total_loss = 0
    index = 0

    print('Training...')
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)



        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).repeat(1, 1, 1, max_length).to(device)


        optimizer.zero_grad()

        output = model(src, tgt, src_mask, tgt_mask)


        loss = criterion(output.contiguous().view(-1, output.shape[-1]), tgt.contiguous().view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (index) % 100 == 0:
            print(f"Batch {index}, LR: {scheduler.get_last_lr()[0]:.7f}")
            print(f"Loss: {loss}")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if param.grad.nelement() > 1:
                        print(f"{name} grad: mean = {param.grad.mean():.6f}, std = {param.grad.std():.6f}")
                    else:
                        print(f"{name} grad: value = {param.grad.item():.6f}")

        index += 1



    return total_loss / len(dataloader)


def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)




            src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
            tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).repeat(1, 1, 1, max_length).to(device)

            output = model(src, tgt, src_mask, tgt_mask)
            loss = criterion(output.contiguous().view(-1, output.shape[-1]), tgt.contiguous().view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


# Training the model
num_epochs = 2
training_losses = []
validation_losses = []
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, optimizer, criterion)
    val_loss = validate(model, validation_dataloader, criterion)
    training_losses.append(train_loss)
    validation_losses.append(val_loss)
    print(f"Epoch: {epoch + 1}, Train loss: {train_loss:.4f}")
    print(f"Epoch: {epoch + 1}, Validation loss: {val_loss:.4f}")



plt.plot(training_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Change in Loss over Epochs')
plt.show()
plt.close()









