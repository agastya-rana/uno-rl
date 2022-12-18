import time
from uno import *
from strat import *
import random
import numpy as np
from bisect import bisect

# ADD NOTE FOR GLEN: we take in whole game, but dont look at other players hands

## We want to approximate the state space as:
## - Num cards of each color that we have (0, 1-2, 3+)^(4 + [W OR WD]) = 243
## - Num of each special cards, per color (0, 1+) ^ (([R OR S OR D]) (4)) = 8
## - Num of cards all the other players have (1, 2-4, 5+)(num_players) = 3(num_players-1)
## - Discard pile color (4) = 4
## - Prev Discard pile color (4) = 4 (see whether opponent is out of color)


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

    def bucket_state(self, game):
        hand = game.deck.player_pile[game.current_player]

        ## - Num cards of each color that we have (0, 1-2, 3+)^(4 + [W OR WD]) = 243
        def bucket_num_cards_color():
            # count number of cards of each color
            cards_per_color = [len([card for card in hand if card.color == color]) for color in game.deck.colors]
            # bucket cards into 0, 1-2, 3+
            for color in cards_per_color:
                if cards_per_color[color] == 0:
                    cards_per_color[color] = 0
                elif cards_per_color[color] <= 2:
                    cards_per_color[color] = 1
                else:
                    cards_per_color[color] = 2
            # index into 0-242
            return sum([cards_per_color[color] * (3 ** color) for color in range(len(cards_per_color))])
        
        ## - Num of each special cards, per color (0, 1+) ^ (([R OR S OR D]) (4)) = 8 
        def bucket_special_cards_per_color():
            colors = ['R', 'G', 'Y', 'B']
            num_special = [len([card for card in hand if card.color == color and card.rank in ["R", "S", "D"]]) for color in colors]
            # bucket cards into 0, 1+
            for color in num_special:
                if num_special[color] == 0:
                    num_special[color] = 0
                else:
                    num_special[color] = 1
            # index into 0-7
            return sum([num_special[color] * (2 ** color) for color in range(len(num_special))])


        ## - Num of cards all the other players have (1, 2-4, 5+)(num_players) = 3(num_players-1)
        def bucket_num_cards_other_players():
            # count number of cards of each player
            num_cards_other = [len(game.deck.player_pile[player]) for player in range(len(game.deck.player_pile)) if player != game.current_player]
            # bucket cards into 1, 2-4, 5+
            for player in num_cards_other:
                if num_cards_other[player] == 1:
                    num_cards_other[player] = 0
                elif num_cards_other[player] <= 4:
                    num_cards_other[player] = 1
                else:
                    num_cards_other[player] = 2
            # index into 0-2(num_players-1)
            return sum([num_cards_other[player] * (3 ** player) for player in range(len(num_cards_other))])

        ## - Discard pile color (4) = 4
        def discard_pile_color_idx():
            return game.deck.colors.index(game.discard_pile[:-1].color)

        ## - Prev Discard pile color (4) = 4 
        def discard_pile_color_idx_prev():
            return game.deck.colors.index(game.discard_pile[-2].color)
        
        def index_all():
            (aH, bH, cH, dH, eH) = (bucket_num_cards_color(), bucket_special_cards_per_color(), bucket_num_cards_other_players(), discard_pile_color_idx(), discard_pile_color_idx_prev())
            (aN, bN, cN, dN, eN) = (243, 8, 3 * (len(game.deck.player_pile) - 1), 4, 4)
            return aH + (bH * aN) + (cH * aN * bN) + (dH * aN * bN * cN) + (eH * aN * bN * cN * dN)
            

    
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