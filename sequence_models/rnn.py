#!/usr/bin/env python

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
import wandb
from torchtext.data import RawField, ReversibleField, LabelField
from torchtext.datasets import WikiText2

device = 'cuda:1'
learning_rate = 0.1
embedding_dim = 300
data_dir = '/home/apower/data/text/wikitext-2'

# config = {
#             'device': device,
#             #'initializer': None,
#             #'init_gain': 5,
#             'learning_rate': learning_rate,
#             #'load_workers': os.cpu_count(), 
#             #'batch_size': 190,
#             'max_epochs': 5000,
#             #'training_loops': 4,
#             #'dropout': 0.5,
#             #'optimizer': 'SGD',
#             #'dataset': 'imagenette2-320',
#             #'dataset': 'oxford-iiit-pet',
#             #'random_seed': 1,
#          }
# wandb.init(project="my-rnn", config=config)


device = torch.device(device)
print("Using device:", device)

torch.manual_seed(1)

vocab = torchtext.vocab.GloVe()
#vocab.vectors[g.stoi['apple'],:]

tokenize = lambda x: x.split()
TEXT = ReversibleField(sequential=True, tokenize=tokenize, lower=True)
LABEL = LabelField()

wikitext2_path = '/home/apower/data/text/wikitext-2'
train_iter, test_iter, val_iter = WikiText2.iters(root=wikitext2_path, 
                                                  #vectors=vocab,
                                                  device=device,
                                                  #bptt_len=5,
                                                  )

train_batches = list(train_iter)

[vocab.itos[i] for i in train_batches[0].text[:,0]]
[vocab.itos[i] for i in train_batches[0].text[0,:]]

class Vocab():
    def __init__(self, path):
        itos = []
        for f_type in ('train', 'test', 'valid'):
            f_name = os.path.join(path, 'wiki.' + f_type + '.tokens')
            with open(f_name, 'r') as f: 
                itos.extend(f.read().split())
        self._itos = list(set(itos))
        self._itos.sort()
        self._stoi = dict([(s, i) for (i,s) in enumerate(self._itos)])
        self.embedding = nn.Embedding(len(self._itos), embedding_dim)
        self.len = len(self._itos)
        self.dim = embedding_dim

    def stoi(self, string):
        return self._stoi[string]

    def itos(self, index):
        return self._itos[index]

    def embed(self, value):
        if type(value) == str:
            index = self.stoi(value)
        elif type(value) == int:
            index = value
        else:
            raise Exception(repr(value) + ' is not a str or int')
        return self.embedding.weight[index, :]

class Loader():
    def __init__(self, path, vocab, bptt_len):
        self.path = path
        self.vocab = vocab
        self.bptt_len = bptt_len

    def __iter__(self):
        self.batches = []
        with open(self.path, 'r') as data_file:
            self._data_lines = data_file.readlines():
            self._data_index = 0
            self._data_max_index = len(self._data_lines)
        return self
 
    def __next__(self):
        if self._data_index >= self._data_max_index:
            raise StopIteration
        line = self._data_lines[self._data_index]
        self._data_index += 1
        tokens = line.split()
        token_len = len(tokens)
        if bptt_len > token_len:
            raise Exception(f'text line {self._data_index -1}' + 
                            f' does not contain {self.bptt_len} tokens')
        result = []
        for i in range(token_len - self.bptt_len + 1):
            X_tokens = tokens[i:i + self.bptt_len - 1]
            Y_token = tokens[i + self.bptt_len]
            X_embedded = [self.vocab.embed(t) for t in X_tokens]
            Y_embedded = self.vocab.embed(Y_token)
            result.append([X_embedded, Y_embedded])
        return result

vocab = Vocab(data_dir)

train_data = Loader(path=data_dir + '/wiki.train.tokens', 
                    vocab=vocab, 
                    bptt_len=bptt_len)
test_data = Loader(path=data_dir + '/wiki.test.tokens',
                    vocab=vocab, 
                    bptt_len=bptt_len)
val_data = Loader(path=data_dir + '/wiki.valid.tokens',
                    vocab=vocab, 
                    bptt_len=bptt_len)

# predict the next word

class RecurrentModule(nn.Module):
    def __init__(self, num_hidden_nodes, vocab):
        super().__init__()

        self.vocab = vocab
        self.num_hidden_nodes = 
        self.Wax = nn.Linear(in_features=vocab.dim, out_features=num_hidden_nodes)
        self.Waa = nn.Linear(in_features=num_hidden_nodes, out_features=num_hidden_nodes)
        self.Way = nn.Linear(in_features=num_hidden_nodes, out_features=vocab.dim)

    def forward(self, prev_activation, in_vector):
        activation = F.tanh(self.Waa(prev_activation) + self.Wax(in_vector))
        out_vector = F.tanh(self.Way(activation))
        return activation, out_vector


model = RecurrentModule(num_hidden_nodes=vocab.dim, vocab)

def learn_one_batch(vectors):
    optimizer.zero_grad()
    y = vectors.pop()
    prev_activation = torch.zeros((num_hidden_nodes, 1), device=device)
    for x in vectors:
        prev_activation, y_hat = model(prev_activation, vector)
    loss = criterion(y_hat, y)
    loss.backward()
    optimizer.step()

---------------------------

def train_model(model, train_loader, dev_loader, learning_rate=0.1, max_epochs=20):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, wandb.config.optimizer)(model.parameters(), lr=learning_rate)

    for epoch in range(max_epochs):
        model.train()
        t0 = time.time()
        for minibatch, minibatch_labels in train_loader:
            # Transfer to GPU
            X, y = minibatch.to(device), minibatch_labels.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.item()  # <-- If you delete this it won't learn
            loss.backward()
            optimizer.step()
        t1 = time.time()
        duration = t1-t0
        loss_num = loss.item()
        train_accuracy = accuracy(model, train_loader, 'train')
        dev_accuracy = accuracy(model, dev_loader, 'dev')
        relative_accuracy = dev_accuracy / train_accuracy
        #train_accuracy = 'False'
        #dev_accuracy = 'False'
        #relative_accuracy = 'False'
        torch.save(model.state_dict(), './resnet-augmenting-' + wandb.config.dataset + '.pt')
        wandb.log({'loss': loss.item(), 
                   'learning_rate': learning_rate,
                   'secs_per_epoch': duration, 
                   'train_accuracy': train_accuracy, 
                   'dev_accuracy': dev_accuracy, 
                   'relative_accuracy': relative_accuracy})
        print(' ' * 4, 
              '%.1f seconds -' % (duration), 
              'epoch:', epoch, 
              'lr: %.1f  ' % learning_rate,
              'loss: %.1f  ' % loss_num, 
              'train: %.1f  ' % train_accuracy, 
              'dev: %.1f  ' % dev_accuracy, 
              'relative_accuracy: %.1f  ' % relative_accuracy)

    return model
