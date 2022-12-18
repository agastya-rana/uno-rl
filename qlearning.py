import time
from uno import *
from strat import *
import random
import numpy as np
from bisect import bisect

## We want to approximate the state space as:
## - Num cards of each color that we have
## - Num of each special card that we have 
## - Num of each special card in the discard pile color (playable)
## - Num of cards all the other players have
## - Discard pile rank (could discretize as number for num) and color
## - Could also add discard pile number of cards of each color played already

## Action space as:
## - Draw card
## - Playing R, S, D, W, WD
## - Playing a same color card
## - Playing a same rank card

class QLearningAgent():
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((num_states, num_actions))
        self.last_state = None
        self.last_action = None

    def bucket_state(self, state):
        ## Return index of Q-learning bucket that corresponds to this state
        idx = 0
        return idx
    
    def bucket_action(self, action, game):
        ## Return index of Q-learning bucket that corresponds to this action
        ## 8 actions
        idx = None
        specials = ["R", "S", "D", "W", "WD"]
        card, param = action
        if card == None:
            idx = 0
        if card.rank in specials:
            idx = specials.index(card.rank) + 1
        if idx is None:
            if card.color = game.discard_card().color:
                idx = 6
            ## Prioritize color if rank and color are equal
            elif card.rank = game.discard_card().rank:
                idx = 7
        return idx

    def get_action(self, game, state_bucket):
        possible = game.possible_actions(game.current_player)
        if random.random() < self.epsilon:
            return random.randint(0, len(possible) - 1)
        else:
            return np.argmax(self.Q[state_bucket])

    def update(self, state, action, reward, next_state):
        ## Update below to allow convergence at end
        self.Q[self.last_state, self.last_action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[self.last_state, self.last_action])
        self.last_state = state
        self.last_action = action

    def reset(self):
        self.last_state = None
        self.last_action = None

    def train(self, timelimit=10, agent_idx=0, other_player=random_strategy):
        start_time = time.time()
        num_players = 2
        num_decks = 1
        while time.time() - start_time < timelimit:
            ## Init game
            game = Uno(num_players, Deck(num_decks))
            game.initial_state()
            while game.current_player != agent_idx:
                ## Take other_player strat actions until it's my turn
                other_action = random_strategy(game)
                game.take_action(other_action, game.current_player)
            while not game.is_over():
                ## Choose action
                state_bucket = bucket_state(game)
                action = self.get_action(game, state_bucket)
                game.take_action(action, agent_idx)
                reward = 1 if game.has_won(agent_idx) else 0
                ## Compute the reward and final state (by simulating the other players)
                while game.current_player != agent_idx:
                    other_action = random_strategy(game)
                    game.take_action(other_action, game.current_player)
                    reward = -1 if game.has_won(game.current_player) else reward
                next_state = bucket_state(game)
                ## Update Q-values
                self.update(state_bucket, action, reward, next_state)