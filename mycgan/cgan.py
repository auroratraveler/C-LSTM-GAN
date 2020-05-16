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

        self.lstmcell0 = nn.LSTMCell(input_size=input_dim_0, hidden_size=hidden_units)
        self.lstmcell1 = nn.LSTMCell(input_size=input_dim, hidden_size=hidden_units)
        self.lstmcell2 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(in_features=hidden_units, out_features=midi_dim)

    def forward(self, condition, noise):
        length, batch_size, condition_dim = condition.shape  # L,N,D
        hidden_units = args.hidden_units
        h1 = h2 = torch.rand(batch_size, hidden_units)
        c1 = c2 = torch.rand(batch_size, hidden_units)
        output = []

        for len in range(length):
            if len == 0:
                h1, c1 = self.lstmcell0(torch.cat((condition[len], noise), dim=1), (h1, c1))
            else:
                h1, c1 = self.lstmcell1(torch.cat((condition[len], out), dim=1), (h1, c1))
            h1 = self.dropout(h1)
            h2, c2 = self.lstmcell2(h1, (h2, c2))
            out = self.fc(h2)  # out.shape(N,3)
            output.append(out)  # output element should be (N,3), output is a list which len(output)=length

        gen_midi = torch.stack(output)  # (L,N,3)
        # print(gen_midi.shape)
        return gen_midi


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