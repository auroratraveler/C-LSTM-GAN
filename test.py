import argparse
import numpy as np
import pretty_midi

import midi_util
from cgan import Generator
from cgan import Discriminator

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', default=2, type=int)
parser.add_argument('--batchsize', default=6, type=int, help='number of songs')
parser.add_argument('--noise_dim', default=30, type=int)
parser.add_argument('--midi_dim', default=3, type=int)
parser.add_argument('--hidden_units', default=400, type=int)
parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.999, type=float)
args = parser.parse_args(args=[])

cuda = True if torch.cuda.is_available() else False

bce
# load training data
train_data = np.load('.\\data\\dataset_matrices\\train_data_matrix.npy')   # shape=(num_sequences_per_song*num_songs,
# (num_syll_features + num_midi_features) * seqlength)
np.random.shuffle(train_data)
data = train_data  # (11149, 460)
batch_size = args.batchsize  # choose first 10 songs to train
data = data[:batch_size]  # (6, 460)
midi = data[:,:60]  # (6, 60)  num_midi_features * seqlength = 3 * 20 = 60
syll = data[:, 60:]  # (6, 400)
syll = torch.from_numpy(syll)
midi = torch.from_numpy(midi)


# convert loaded data to the type we need
dtype = torch.float32

noise_dim = args.noise_dim
midi_dim = args.midi_dim
hidden_units = args.hidden_units
num_layers = args.num_layers

condition = torch.stack(torch.split(syll, 20, dim=1))  # syllable embedding condition (20 * 6 * 20) sequence_len * batch_size * emb_dim
print('condition_shape', condition.shape)
midi = torch.stack(torch.split(midi, 3, dim=1))
print('midi_shape', midi.shape)
noise = torch.rand(20, batch_size, noise_dim).type(dtype)
print('noise_shape', noise.shape)

condition = condition.clone().type(dtype)
midi = midi.clone().type(dtype)


generator = Generator()
discriminator = Discriminator()

gen_midi = generator(condition, noise).type(dtype)
logits_fake = discriminator(condition, gen_midi)
print(logits_fake.shape)
