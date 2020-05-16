import argparse

import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=6, type=int, help='number of songs')
parser.add_argument('--hidden_units', default=400, type=int)
parser.add_argument('--num_layers', default=2, type=int)
args = parser.parse_args(args=[])


class Generator(nn.Module):
    def __init__(self, input_dim_0=50, input_dim=23, hidden_units=400, dropout_prob=0.6, noise_dim=30, midi_dim=3):
        super(Generator, self).__init__()

        self.fc_1 = nn.Linear(in_features= input_dim_0, out_features= hidden_units)
        self.relu = nn.ReLU()
        self.LSTM_1 = nn.LSTM(input_size= hidden_units,hidden_size = hidden_units, num_layers = 2, dropout = dropout_prob)
        #self.lstmcell0 = nn.LSTMCell(input_size=input_dim_0, hidden_size=hidden_units)
        #self.lstmcell1 = nn.LSTMCell(input_size= hidden_units, hidden_size=hidden_units)  # unidirectional

        self.tanh = nn.Tanh()

        #self.lstmcell2 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)

        self.fc_4 = nn.Linear(in_features=hidden_units, out_features = midi_dim)

    def forward(self, condition, noise):
        length, batch_size, condition_dim = condition.shape  # L,B,D

        #print('1', condition.shape)
        #print("2", noise.shape)
        condtion = self.relu(self.fc_1(torch.cat([condition,noise],dim=2)))
        #print(condtion.shape)

        hidden_units = args.hidden_units

        h0 = torch.randn(2, batch_size, hidden_units)
        c0 = torch.randn(2, batch_size, hidden_units)

        output, (hn, cn) = self.LSTM_1(condtion, (h0, c0))
        #output.size (L * B * Hidden size)

        output = self.tanh(output)
        gen_midi = self.fc_4(output)

        return gen_midi

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


class Discriminator(nn.Module):
    def __init__(self, input_dim=23, hidden_units=400, output_dim=1, num_layers=2, dropout_prob=0.6):
        super(Discriminator, self).__init__()

        self.LSTM = nn.LSTM(input_size=input_dim, hidden_size=hidden_units, num_layers=num_layers, dropout=dropout_prob)
        self.Linear = nn.Linear(in_features=hidden_units, out_features=output_dim)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, condition, midi):
        num_layers = args.num_layers
        batch_size = args.batch_size
        hidden_units = args.hidden_units
        h3 = torch.rand(num_layers, batch_size, hidden_units)
        c3 = torch.rand(num_layers, batch_size, hidden_units)

        D_input = torch.cat((condition, midi), dim=2)
        dtype = torch.float32
        D_input = D_input.clone().type(dtype)
        out, (h3, c3) = self.LSTM(D_input, (h3, c3))
        out = self.Linear(out)
        validity = self.Sigmoid(out)
        return validity

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

