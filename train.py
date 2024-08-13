import torch
import Test
import data_format
import torch.nn as nn
import numpy as np
import math
import pandas as pd
import openpyxl
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torch.optim.lr_scheduler import MultiStepLR


train_data = pd.read_excel('wmt14_translate_de-en_train.csv.xlsx' , engine= 'openpyxl')
validation_data = pd.read_excel('wmt14_translate_de-en_validation.csv.xlsx', engine= 'openpyxl')
test_data = pd.read_excel('wmt14_translate_de-en_test.csv.xlsx', names=['de', 'en'], engine= 'openpyxl')

src_lang = 'de'
tgt_lang = 'en'


tgt_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
src_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')



src_vocab = data_format.build_vocabulary(test_data , src_tokenizer, lang= 'de')
tgt_vocab = data_format.build_vocabulary(test_data, tgt_tokenizer, lang= 'en')



train_data = data_format.TranslationDataset(train_data, 'de', 'en', src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab )
validation_data = data_format.TranslationDataset(validation_data, 'de', 'en', src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab )
test_data = data_format.TranslationDataset(test_data, 'de', 'en', src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab )



batch_size = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_dataloader = data_format.get_loaders(train_data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, batch_size)

validation_dataloader = data_format.get_loaders(validation_data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, batch_size)

test_dataloader = data_format.get_loader(test_data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, batch_size)

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)
src_seq_length = max(len(sample[0]) for sample in train_data)
tgt_seq_length = max(len(sample[1]) for sample in train_data)

model = Test.roll_out(src_vocab_size, tgt_vocab_size, src_seq_length, tgt_seq_length, layers=6)

model = model.to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.01)

optimizer = optim.Adam(model.parameters(), lr=0.1)



scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400,800,1200, 1600], gamma=0.1)


def train(model, dataloader, optimizer, criterion, clip=1.0):
    model.train()
    total_loss = 0
    index = 0

    for src, tgt in dataloader:
        src = src.transpose(0, 1).to(device)  # Transpose to match the model's expected input shape
        tgt = tgt.transpose(0, 1).to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]


        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
        tgt_mask = (tgt_input != 0).unsqueeze(1).unsqueeze(3)
        tgt_mask = tgt_mask & torch.tril(torch.ones(tgt_mask.shape[2], tgt_mask.shape[3])).bool().to(device)

        optimizer.zero_grad()

        output = model(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(output.contiguous().view(-1, output.shape[-1]), tgt_output.contiguous().view(-1))

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

        index +=1

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.transpose(0, 1).to(device)
            tgt = tgt.transpose(0, 1).to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
            tgt_mask = (tgt_input != 0).unsqueeze(1).unsqueeze(3)
            tgt_mask = tgt_mask & torch.tril(torch.ones(tgt_mask.shape[2], tgt_mask.shape[3])).bool().to(device)

            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.contiguous().view(-1, output.shape[-1]), tgt_output.contiguous().view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


# Training the model
num_epochs = 10
training_losses = []
validation_losses = []
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, optimizer, criterion)
    val_loss = validate(model, validation_dataloader, criterion)
    training_losses.append(train_loss)
    validation_losses.append(val_loss)
    training_losses.append(train_loss)
    print(f"Epoch: {epoch + 1}, Train loss: {train_loss:.4f}")
    print(f"Epoch: {epoch + 1}, Validation loss: {val_loss:.4f}")

torch.save(model.state_dict(), 'transformer_model.pth')

plt.plot(training_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Change in Loss over Epochs')
plt.show()
plt.close()



