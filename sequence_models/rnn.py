#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os

torch.manual_seed(1)

vocab_len = 10**4
embedding_dim = vocab_len

def create_vocab():
    vocab_set = set()
    base_dir = '/home/apower/data/text/wikitext-2'
    for f_type in ('train', 'test', 'valid'):
        file_name = 'wiki.' + f_type + '.tokens'
        file_name = os.path.join(base_dir, file_name)
        with open(file_name, 'r') as f: 
            for line in f.readlines():
                vocab_set.update(line.split())
    v = list(vocab_set)
    v.sort()
    return enumerate(v)

vocab = create_vocab()
# predict the next word

class ZerosModule(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.zeros = torch.zeros((size, 1))

        # turn off autograd here
        self.requires_grad(False)

    def forward(self):
        return self.zeros


class RecurrentModule(nn.Module):
    def __init__(self, vocab_len, prev_module=None, embedding_dim=0, num_hidden_nodes=0):
        super().__init__()

        if not embedding_dim:
            embedding_dim = vocab_len

        if not num_hidden_nodes:
            num_hidden_nodes=embedding_dim

        if prev_module:
            self.prev_module = prev_module
        else:
            self.prev_module = ZerosModule(size=embedding_dim)

        self.Wax = nn.Linear(in_features=vocab_len, out_features=num_hidden_nodes)
        self.Waa = nn.Linear(in_features=num_hidden_nodes, out_features=num_hidden_nodes)
        self.Way = nn.Linear(in_features=num_hidden_nodes, out_features=embedding_dim)


    def forward(self, X):
        last_word = 
 
 
        self.prev_activations = nn.functional.relu(self.Waa(self.prev_activations) + self.Wax(X))
        return 



def train():
    optimizer = torch.optim.Adam(RecurrentModule.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()


    X, y = local_batch.to(device), local_labels.to(device)
    optimizer.zero_grad()
    y_pred = model.forward(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
