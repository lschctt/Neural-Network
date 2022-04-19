# pad by hand

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence
import numpy as np
import csv

USE_GPU = 0
Input_size = 128
Hidden_size = 100
Emb_size = 80
Num_layers = 1
Bidirectional = True
Batch_size = 2


# Create dataset for train and test
class New_dataset(Dataset):
    def __init__(self, is_train):
        super().__init__()
        if is_train:
            filepath = "./data/names_train.csv"
        else:
            filepath = "./data/names_test.csv"

        with open(filepath, 'rt') as f:
            reader = csv.reader(f)
            data = list(reader)

        self.name = [n for n, _ in data]
        self.country = [c for _, c in data]
        self.len = len(self.name)

        # set(): create an unordered and non-repeating set
        self.country_list = list(sorted(set(self.country)))
        self.country_num = len(self.country_list)
        self.country_dict = self.get_country_dict()

    def __getitem__(self, index):
        return [self.name[index], self.country_dict[self.country[index]]]

    # to create a dictionary for countries
    def get_country_dict(self):
        country_dict = dict()
        for i in range(self.country_num):
            con = self.country_list[i]
            # con is key and i is value
            country_dict[con] = i
        return country_dict

    def get_country_num(self):
        return self.country_num

    def __len__(self):
        return self.len


train_set = New_dataset(is_train=True)
train_loader = DataLoader(train_set, batch_size=Batch_size, shuffle=True, num_workers=0)
test_set = New_dataset(is_train=False)
test_loader = DataLoader(test_set, batch_size=Batch_size, shuffle=False, num_workers=0)
Output_size = train_set.get_country_num()

# Create model
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, emb_size, output_size, num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.embedding_size = emb_size
        self.bidirection = 2 if bidirectional else 1
        self.output_size = output_size

        self.emb = torch.nn.Embedding(self.input_size, self.embedding_size)
        self.gru = torch.nn.GRU(self.embedding_size, self.hidden_size, bidirectional=bidirectional)
        self.linear = torch.nn.Linear(hidden_size * self.bidirection, output_size)

    def forward(self, x, seq_length):
        x = x.t()  # x:(batch_size, seq_len) --> (seq_len, batch_size)
        hidden = torch.zeros(self.num_layers * self.bidirection, Batch_size,
                             self.hidden_size)       # h:(num_layers,batch_size,hidden_size)
        x_emb = self.emb(x)  # x:(seq_len, batch_size) --> (seq_len, batch_size, embedding_size)

        # pack_padded_sequence: get useful information, invalid information(like 0) for padding is not counted
        # seq_length: list of true sequence lengths of each batch element
        x_pack = pack_padded_sequence(x_emb, seq_length)
        # hidden_new: final hidden
        output, hidden_new = self.gru(x_pack, hidden)   # hidden_new: (seq_len, batch_size, )

        if self.bidirection == 1:
            output_l = self.linear(hidden)
        else:
            hidden_cat = torch.cat((hidden_new[0], hidden_new[1]), dim=1)
            output_l = self.linear(hidden_cat)

        return output_l.view(-1, self.output_size)


def create_tensor(data):
    if USE_GPU:
        device = torch.device('cuda:0')
        data = data.to(device)
    return data


def become_tensor(names, countries):
    # change word into list of letters
    name_char = [list(names[i]) for i in range(len(names))]
    # change letters into ASCII
    name_ord = []   # to record the ASCII of names
    name_len = []   # to record the length of names
    for j in range(len(name_char)):
        name_ord.append([ord(t) for t in name_char[j]])
        name_len.append(len(name_char[j]))
    name_len = torch.LongTensor(name_len)

    # make tensor of name, --> (batch_size, seq_len)
    seq_tensor = torch.zeros(len(name_char), max(name_len)).long()   # initialize a zero matrix
    for i in range(len(name_ord)):
        seq_tensor[i, :name_len[i]] = torch.LongTensor(name_ord[i])
    # sort by length to use pack_padded_sequence
    name_len, perm_idx = name_len.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensor(seq_tensor), create_tensor(name_len), create_tensor(countries)

def train(a):
    num = 0
    for data_train in train_loader:
        name_train = data_train[0]
        country_train = data_train[1]

        # forward
        INPUT_train, len_list_train, Country_train = become_tensor(name_train, country_train)
        Country_hat = model.forward(INPUT_train, len_list_train)

        # backword()
        l = loss(Country_hat, Country_train)
        optimization.zero_grad()
        l.backward()

        # optim
        optimization.step()

        num += 1
        if num % 200 == 0:
            print('epoch:', a, 'num:', num, 'Loss:', l.item())


def test():
    correct = 0
    total = len(test_set)
    with torch.no_grad():
        for data_test in test_loader:
            name_test = data_test[0]
            country_test = data_test[1]

            INPUT_test, len_list_test, Country_test = become_tensor(name_test, country_test)

            output_test = model.forward(INPUT_test, len_list_test)
            pred = output_test.max(dim=1, keepdim=True)[1]
            correct += pred.eq(Country_test.view_as(pred)).sum().item()
        percent = (correct/total) * 100
        print('Currency is: ', percent, '%')
    return correct/total


if __name__ == '__main__':
    model = Model(Input_size, Hidden_size, Emb_size, Output_size, num_layers=Num_layers, bidirectional=Bidirectional)
    if USE_GPU:
        Device = torch.device('cuda:0')
        model.to(Device)
    loss = torch.nn.CrossEntropyLoss()
    optimization = torch.optim.Adam(params=model.parameters(), lr=0.001)

    acc_list = []
    for e in range(1, 10):
        # Train cycle
        train(e)
    acc = test()
    acc_list.append(acc)




















