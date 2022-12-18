import numpy as np
import random
import itertools as it
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

## Need a learning and a target network network

## Need to define parameters of each network (which should be the same)
## Number of each kinda card in own deck
## How do we capture the agent knowing that the opponent is out of a single color - need only the last #num_players cards on the deck

ranks = [str(n) for n in range(10)] + ["R", "S", "D"]
colors = ["R", "Y", "B", "G"]

def conv_cards_to_list(state, cards):
    for card in cards:
        if card.rank == 'W':
            state[-2] += 1
        elif card.rank == 'WD':
            state[-1] += 1
        else:
            state[ranks.index(card.rank) + 13*colors.index(card.color)] += 1
    return state

def conv_state_to_tensor(game, discard_memory=1):
    """ Returns a PyTorch tensor"""
    ## Tensor contains one 4 x 13 + 2 = 56 list containing number of each card in own hand 
    ## Then contains number of cards in others hands: num_players - 1 normalized by 7
    ## Then contains the discard pile last x elements ## 56 x 7 list of one hot encodings - could try to boil this down by
    ## ignoring rank unless special card
    temp = [0 for i in range(56)]
    state = [conv_cards_to_list(temp.copy(), game.deck.player_pile[game.current_player])]
    state.extend([conv_cards_to_list(temp.copy(), game.deck.discard_pile[-(i+1)]) for i in range(discard_memory)])
    state.extend([len(game.deck.player_pile[player_idx])/7 for player_idx in range(game.num_players) if player_idx != game.current_player])
    return state

class DQNAgent():
    def __init__(self, num_players=2, discard_memory=1):
        ## Create Learning, Target Network
        self.state_size = 56*(discard_memory+1) + (num_players-1)
        self.action_size = 8
        self.hidden_dim = 24
        self.learning_rate = 1e-3
        pass
    
    def train(self, num_rounds=100000, batch_update=1000):
        ## Train by running sims on the target network, storing in database and then updating learning
        ## Figure out how to deal with softmaxed invalid outputs
        pass
    
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(self.hidden_dim, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.hidden_dim, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    