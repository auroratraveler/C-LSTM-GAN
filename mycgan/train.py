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


# loss
def bce_loss(input, target):
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def G_loss(logits_fake):
    size = logits_fake.size()
    true_labels = torch.ones(size).type(dtype)
    loss = bce_loss(logits_fake, true_labels)
    return loss


def D_loss(logits_real, logits_fake):
    size = logits_real.size()
    true_labels = torch.ones(size).type(dtype)
    loss = bce_loss(logits_real, true_labels) + bce_loss(logits_fake, true_labels-1)
    return loss


generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    loss.cuda()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
# SGD更糟糕
# optimizer_G = torch.optim.SGD(generator.parameters(), lr=args.lr, momentum=0.9, weight_decay=0, nesterov=False)
# optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=args.lr, momentum=0.9, weight_decay=0, nesterov=False)

# FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


# load training data
train_data = np.load('.\\data\\dataset_matrices\\train_data_matrix.npy')
np.random.shuffle(train_data)
data = train_data  # (11149, 460)
batch_size = args.batchsize  # choose first 10 songs to train
data = data[:batch_size]  # (10, 460)
midi = data[:,:60]  # (10, 60)
syll = data[:, 60:]  # (10, 400)

syll = torch.from_numpy(syll)
midi = torch.from_numpy(midi)


# convert loaded data to the type we need
dtype = torch.float32

noise_dim = args.noise_dim
midi_dim = args.midi_dim
hidden_units = args.hidden_units
num_layers = args.num_layers

condition = torch.stack(torch.split(syll, 20, dim=1))  # syllable embedding condition
print('condition_shape', condition.shape)
midi = torch.stack(torch.split(midi, 3, dim=1))
print('midi_shape', midi.shape)
noise = torch.rand(batch_size, noise_dim).type(dtype)

condition = condition.clone().type(dtype)
midi = midi.clone().type(dtype)


# train the GAN
for epoch in range(args.num_epochs):
    for i in range(batch_size):
        optimizer_G.zero_grad()
        gen_midi = generator(condition, noise).type(dtype)
        logits_fake = discriminator(condition, gen_midi)
        g_loss = G_loss(logits_fake)
        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        logits_real = discriminator(condition, midi).type(dtype)
        gen_midi = generator(condition, noise).detach().type(dtype)
        logits_fake = discriminator(condition, gen_midi)
        d_loss = D_loss(logits_real, logits_fake)
        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch + 1, args.num_epochs, i + 1, batch_size, d_loss.item(), g_loss.item())
        )


# see the generated midi value
print('gen_midi', gen_midi)
print('dis_gen_midi', midi_util.discretize_midi(gen_midi))
# print(midi_util.discretize_midi(gen_midi).data.type())


# generate midi files
dis_gen_midi = midi_util.discretize_midi(gen_midi)
new_midi = midi_util.create_midi_pattern_from_discretized_data(dis_gen_midi)
midi_util.create_midi_files(new_midi)
print('generate midi files successfully')
