import pretty_midi

import torch
import numpy as np

def discretize_midi(midi):
    length, batch_size, midi_dim = midi.shape
    resonable_pitch = list(range(127))
    resonable_duration = [0.25,  0.5, 0.75, 1., 1.5, 2., 3., 4., 6., 8., 16., 32.]  # 0.25-->sixteenth note
    resonable_rest = [0., 1., 2., 4., 8., 16., 32.]  # 0-->no rest, 4-->whole rest
    discretized_midi = torch.zeros_like(midi)
    for dim in range(midi_dim):
        if dim == 0:
            resonable_list = resonable_pitch
        if dim == 1:
            resonable_list = resonable_duration
        if dim == 2:
            resonable_list = resonable_rest
        for length in range(length):
            for num in range(batch_size):
                gen_attribute = midi[length][num][dim]
                distance = 1000
                for i, value in enumerate(resonable_list):
                    if torch.pow(gen_attribute-value, 2)<distance:
                        distance = torch.pow(gen_attribute-value, 2)
                        discretized_midi[length][num][dim] = value
    return discretized_midi


def create_midi_pattern_from_discretized_data(midi):
    _, batch_size, _ = midi.shape  # (L,N,D)
    midi = torch.split(midi, 1, dim=1)
    # a tuple contain N elements, each one is a tensor of shape(L,1,D)print(dis_gen_midi.data.type())

    midi_patterns = []
    new_midi = pretty_midi.PrettyMIDI()
    voice = pretty_midi.Instrument(1)  # here to change the used instruments
    tempo = 120
    starttime = 0  # Time since the beginning of the song, in seconds

    for i in range(batch_size):
        each_song = torch.squeeze(midi[i])
        each_song = each_song.numpy()
        length, _ = each_song.shape

        for j in range(length):
            duration = each_song[j][1] * 60 / tempo  # use time to denote duration
            if j < length - 1:
                rest = each_song[j + 1][2] * 60 / tempo  # rest refers to the rest time before this note
            else:
                rest = 0
            note = pretty_midi.Note(velocity=100, pitch=int(each_song[j][0]),
                                    start=starttime, end=starttime + duration)
            voice.notes.append(note)
            starttime += duration + rest
        starttime = 0
        new_midi.instruments.append(voice)

        midi_patterns.append(new_midi)
    return midi_patterns  # a list contain N new_midis


def create_midi_files(midi_patterns):
    # midi_patterns = create_midi_pattern_from_discretized_data(midi)
    for num, each_midi_pattern in enumerate(midi_patterns):
        destination = ('train%d.mid') % (num+1)
        each_midi_pattern.write(destination)
