import torch
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
from tutor_net import *

class Env:
    def __init__(self, model, state_hst_size: int = WINDOW_LENGTH):
        self.state_hyst = state_hst_size
        self.shape = 12 # 12 notes in octave

        self.state = [[0. for _ in range(self.shape)] for _ in range(self.state_hyst)] # initial state
        self.reward_net = model

    def start_mdp(self):
        self.state = [[0. for _ in range(self.shape)] for _ in range(self.state_hyst)]
        return torch.Tensor(self.state).flatten()

    def step(self, action: int):
        binary = bin(action)[2:].zfill(self.shape) # action to binary -> delete the pattern "0b" and fill to 12 bits
        
        action_dec = [1. if char == "1" else 0. for char in str(binary)] # -> convert bit-string to tensor

        self.state.pop(0) # delete oldest entry
        self.state.append(action_dec) # push new state
        
        with torch.no_grad():
            rew = self.reward_net(torch.Tensor(self.state).flatten())

        return torch.Tensor(self.state).flatten(), rew
