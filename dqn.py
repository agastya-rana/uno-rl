import numpy as np
import random
import itertools as it
import torch
import torch.nn as nn

## Need a learning and a target network network

## Need to define parameters of each network (which should be the same)
## Number of each kinda card in own deck
## How do we capture the agent knowing that the opponent is out of a single color - need only the last #num_players cards on the deck

def conv_state_to_tensor(state):
    """ Returns a PyTorch tensor"""
    pass


class DQNAgent():
    def __init__(self):
        ## Create Learning, Target Network
        pass
    
    def train(self, num_rounds=100000, batch_update=1000):
        ## Train by running sims on the target network, storing in database and then updating learning


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        ## Duplicate for target network
        super().__init__()

        # Define constants.
        self.latent_space_size = latent_space_size

    def forward(self, state):
        pass

    