import torch
import torch.nn as nn
import numpy as np
import pychord as pyc
import random

# not yet optimized for gpu (cuda)

WINDOW_LENGTH = 20
NUM_CHORD_PROGRESSIONS = 200
LOAD_MODEL = False

EPOCHS = 20
num_chords = 4095 # 2 possibilities foreach of the 12 notes: either played or not (-1 for all not played)
path = "lstm_real_book/chord_sentences.txt"

comps = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def index_to_name(num: int):
    binary = bin(num)[2:].zfill(12) # see env_build.py for encoding
    action_dec = [1. if char == "1" else 0. for char in str(binary)]

    key_pattern = []

    for index in range(12):
        if action_dec[index] == 1.:
            key_pattern.append(comps[index])
    
    return key_pattern


def chord_to_array(chord: str) -> []:
    # print(chord)
    chord = chord.replace(":", "")
    chord = chord.replace("min7", "m7")
    chord = chord.replace("hdim", "m7b5")

    chord = chord.replace("maj/b7", "7")

    chord = chord.replace("maj6", "6")
    chord = chord.replace("min6", "m6")
    chord = chord.replace("min9", "m9")

    if "(" in chord:
        chord = chord.split("(")[0]

    try:
        c = pyc.Chord(chord)

    except:
        if "/" in chord:
            chord = chord.split("/")[0]

        c = pyc.Chord(chord)

    out = []

    for note in comps: # loop over all notes
        components = c.components()
        appearing = 0

        for key in components: # loop over components of chord
            if pyc.Chord(note) == pyc.Chord(key): # note in chord?
                appearing = 1 # -> note appeard in chord
                break
        out.append(appearing)
    
    # print(out)
    return out

class ChordProgressions:
    def __init__(self, chord: list):
        self.prog_array = torch.Tensor([chord_to_array(ch) for ch in chord])

    def __len__(self):
        return self.prog_array.shape[0]

    def sub_chords(self, win_length: int = WINDOW_LENGTH):
        # return all sub-chord-progression in this Progression of length win_length

        sub_chords = []

        for i in range(len(self)-win_length):
            sub_chords.append(self.prog_array[i:(i+20)])

        return [ten.flatten() for ten in sub_chords]

def generate_random_progression(win_length: int = WINDOW_LENGTH):
    return torch.randint(2, (win_length, 12)).float().flatten()

class AutoRewardSystem(nn.Module):
    def __init__(self, window_length: int = WINDOW_LENGTH):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(12*window_length, 256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 20),
            nn.SiLU(),
            nn.Linear(20, 1),
            nn.Tanh()
        )

        self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam(self.parameters(), lr=10**-3)

    def forward(self, x_in: torch.Tensor):
        return self.net(x_in)

    def adjust(self, chord_prog: ChordProgressions, target: float = 1.):
        self.train()

        self.optim.zero_grad(True)

        out = self(chord_prog)

        loss = self.loss(out, torch.Tensor([target]))
        loss.backward()

        self.optim.step()

        return loss.item()

reward_net = AutoRewardSystem()

def fetch_data(filename: str):
    dataset = []
    with open(filename) as f:
        string = f.read()

        progressions = string.split(" _END_ _START_ ")
        progressions[0] = progressions[0].replace("_START_ ", "")
        progressions[-1] = progressions[-1].replace(" _END_", "")

        for i, prog in enumerate(progressions):
            if i >= NUM_CHORD_PROGRESSIONS:
                break

            chord = ChordProgressions(prog.split(" "))
            dataset.append(chord)

    random.shuffle(dataset)
    trainsize = 4*(len(dataset)//5)

    return dataset[:trainsize], dataset[trainsize:]

trainset, evalset = fetch_data(path)

def training():
    aloss = 0
    lentrainset = 0 # not easy calculatable due to subchordprogs

    for data in trainset:
        sub_chords = data.sub_chords()

        for chord in sub_chords:
            loss = reward_net.adjust(chord)
            aloss += loss

            # foreach element of the dataset, train also with a random chord progression (reward -1)

            random_chord = generate_random_progression()

            loss_random = reward_net.adjust(random_chord, -1.)
            aloss += loss_random
            
            lentrainset += 2

    acc = 0
    lenevalset = 0
    for test in evalset:
        sub_chords = data.sub_chords()

        for chord in sub_chords:
            acc += (float(reward_net(chord)) + 1) / 2 # some sort of estimate, since not a classification

            random_chord = generate_random_progression() # also test on random_chord
            acc += (1-float(reward_net(random_chord))) / 2 # yields at acc 0 when giving reward 1. to random_chord and acc 1 when giving reward -1. to random_chord
            
            lenevalset += 2

    return aloss / lentrainset, acc / lenevalset

if LOAD_MODEL:
    reward_net = torch.load(f"AutoReward.pt", weights_only=False)
else:
    print("Starting the training of AutoRewardSystem...")

    for epoch in range(EPOCHS):
        loss, accuracy = training()

        print(f"Average loss in epoch {epoch}: {loss}, Average accuracy in epoch {epoch}: {accuracy}")
        print(reward_net(generate_random_progression()), reward_net(evalset[0].sub_chords()[0])) # output for random chord progression and actually progression

        torch.save(reward_net, "AutoReward.pt")