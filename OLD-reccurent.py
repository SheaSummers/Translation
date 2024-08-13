#Importing needed packages
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import random
import spacy
import math
import time
import matplotlib.pyplot as plt
from collections import Counter
import torchtext;torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import vocab
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader

#Getting tokenizers for English and German
english = get_tokenizer('spacy', language='en_core_web_sm')
german = get_tokenizer('spacy', language='de_core_news_sm')

#Loading the Multi30k Dataset for English to German translation
train_data , validation_data, test_data = Multi30k(language_pair = ('en', 'de'))


#Funtion to build a vocabulary for English and German, minimum of 2 occurances needed in order to add
def build_vocab(data, tokenizer, index):
  counter = Counter()
  for item in data:
    counter.update(tokenizer(item[index]))
  voc = vocab(counter, min_freq=2,  specials=['<bos>', '<eos>', '<unk>', '<pad>'])
  voc.set_default_index(voc['<unk>'])
  return voc

#Building the vocabularies
ger_vocab = build_vocab(train_data, german, 1)
english_vocab = build_vocab(train_data, english, 0)

#Defining the Encoder Class
class Encoder(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, num_layers, drop):
    super(Encoder, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    #A Dropout Layer is used to prevent overfitting of the model
    self.dropout = nn.Dropout(drop)
    #A Embedding Layer used to convert the input tokens into dense vectors
    self.embedding = nn.Embedding(input_size, embedding_size)
    # A Bidirectional LSTM Layer is used for the Recurrent Neural Network
    self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=drop, bidirectional=True)
    #A Fully Connected Layer is used to combine the forward and backwards LSTM outputs
    self.fc = nn.Linear(hidden_size * 2, hidden_size)


  def forward(self, x):
    embedded = self.dropout(self.embedding(x))
    outputs, (hidden, cell) = self.rnn(embedded)
    hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

    hidden = self.fc(hidden)
    hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)



    cell = cell[-2:].transpose(0, 1).contiguous().view(1, -1, self.hidden_size * 2)
    cell = self.fc(cell)
    cell = cell.repeat(self.num_layers, 1, 1)

    return outputs, hidden, cell


#Defining the Attention Class
class Attention(nn.Module):
  def __init__(self, hidden_size, embedding_size):
    super(Attention, self).__init__()
    self.attention = nn.Linear((hidden_size* 3), hidden_size)
    self.v = nn.Linear(hidden_size, 1, bias = False)

  def forward(self, hidden, enc_outputs):
    batch = enc_outputs.shape[1]
    src_len = enc_outputs.shape[0]

    hidden = hidden[-1]
    hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

    enc_outputs = enc_outputs.permute(1, 0, 2)


    energy = torch.tanh(self.attention(torch.cat((hidden, enc_outputs), dim=2)))
    attn = self.v(energy).squeeze(2)

    return torch.softmax(attn, dim=1)


#Defining the Decoder Class
class Decoder(nn.Module):
  def __init__(self, input_size, output_size, embedding_size, hidden_size, num_layers, attention, drop):
    super(Decoder, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = nn.Dropout(drop)
    self.attention = attention
    self.embedding = nn.Embedding(input_size, embedding_size)
    self.rnn = nn.LSTM((embedding_size + hidden_size*2), hidden_size, num_layers, dropout=drop)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x, hidden, cell, enc_outputs):
    x = x.unsqueeze(0)
    embedded = self.dropout(self.embedding(x))

    a = self.attention(hidden, enc_outputs)
    a = a.unsqueeze(1)

    weighted = torch.bmm(a, enc_outputs.permute(1, 0, 2))
    weighted = weighted.permute(1, 0, 2)

    rnn_correct = torch.cat((embedded, weighted), dim = 2)

    outputs, (hidden, cell) = self.rnn(rnn_correct, (hidden, cell))
    predict = self.fc(outputs.squeeze(0))
    return predict, hidden, cell

#Defining the Seq2Seq class which combines the Encoder and Decoder
class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, device):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.device = device

  def forward(self, src, trg, teacher_forcing_ratio=0.5):
    batch_size = src.shape[1]
    trg_len = trg.shape[0]
    trg_vocab_size = self.decoder.fc.out_features

    outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
    enc_out, hidden, cell = self.encoder(src)

    x = trg[0,:]

    for y in range(1, trg_len):
      output, hidden, cell = self.decoder(x, hidden,cell, enc_out)
      outputs[y] = output
      teacher_force = random.random() < teacher_forcing_ratio
      top = output.argmax(1)
      x = trg[y] if teacher_force else top

    return outputs

#Setting the hyperparameters
epochs = 10
batch_size = 128
learning_rate = 0.0008

load_model = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_enc = len(english_vocab)
input_size_dec = len(ger_vocab)
output_size = len(ger_vocab)
enc_embedding = 300
dec_embedding = 300
hidden_size = 514
num_layers = 2
drop = 0.5

#Instantiating the Encoder, Attention, and Decoder
enc = Encoder(input_size_enc, enc_embedding, hidden_size, num_layers, drop)
attn = Attention(hidden_size, dec_embedding)
dec = Decoder(input_size_dec, output_size, dec_embedding, hidden_size, num_layers, attn, drop)

#Creating the Seq2Seq model
model = Seq2Seq(enc, dec, device).to(device)

#Using DataParallel if multiple GPus are available
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

#Setting the padding index
pad_idx = english_vocab.get_stoi()['<pad>']

#Defining the Loss function
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

#Function to initialize weights
def init_weights(mod):
  for name, param in mod.named_parameters():
    if 'weight' in name:
      nn.init.normal_(param.data, mean=0, std=0.01)
    else:
      nn.init.constant_(param.data, 0)

#Applying the weights to the model
model.apply(init_weights)

#Defining the optimizer
optimizer = optim.AdamW(model.parameters())

#Definign the collate funtion for Dataloader
def collate_fn(batch):
  src_batch, tgt_batch = [], []
  for src_sample, tgt_sample in batch:
    try:
      src_batch.append(torch.tensor([english_vocab[token] for token in english(src_sample)]))
      tgt_batch.append(torch.tensor([ger_vocab[token] for token in german(tgt_sample)]))
    except UnicodeDecodeError:
      continue

    if not src_batch or not tgt_batch:
      return None, None

  src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=english_vocab['<pad>'])
  tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=ger_vocab['<pad>'])

  return src_batch, tgt_batch

#Instatiating the DataLoaders for the Training and Validation data
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
validaiton_loader = DataLoader(validation_data, batch_size=int(batch_size/4), shuffle=True, collate_fn=collate_fn)

#Creating a list to hold the loss for each translation
loss_plot = []

#Trainging Loop
for epoch in range(epochs):
  model.train()
  accumulation_steps = 4
  optimizer.zero_grad()
  for index, (input, target) in enumerate(train_loader):
    input = input.to(device)
    target = target.to(device)
    output = model(input, target)
    output = output[1:].reshape(-1, output.shape[2])
    target = target[1:].reshape(-1)
    loss = criterion(output, target)

    loss_plot.append(loss.item())
    if index % 50 == 0:
      print(f'Epoch: {epoch + 1}, Batch: {index}, Loss: {loss.item():.4f}')



    loss = loss/accumulation_steps
    loss.backward()
    if (index+1) % accumulation_steps == 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      optimizer.zero_grad()

#Validating the model after training
val_loss = []
model.eval()

for index, (input, target) in enumerate(validaiton_loader):
  input = input.to(device)
  target = target.to(device)
  output = model(input, target)
  if index % 5 == 0:
    input_words = [english_vocab.get_itos()[indx] for indx in input[:,0].cpu().detach().numpy()]
    target_words = [ger_vocab.get_itos()[indx] for indx in target[:,0].cpu().detach().numpy()]

    _, translated = torch.max(output, dim=2)

    output_words = [ger_vocab.get_itos()[indx] for indx in translated[:,0].cpu().detach().numpy()]

    print("Input: ", input_words)
    print("Target: ", target_words)
    print("Output: ", output_words)
  output = output[1:].reshape(-1, output.shape[2])
  target = target[1:].reshape(-1)
  loss = criterion(output, target)
  val_loss.append(loss.item())

#Ploting the training loss
plt.plot(loss_plot)
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.title('Change in Loss over Training Batches')
plt.show()
plt.close()


print(f'Validation Loss: {np.mean(val_loss):.4f}')
