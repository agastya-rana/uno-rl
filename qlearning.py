import time
from uno import *
from strat import *
import random
import numpy as np
#from numba import jit, cuda

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
    specials = ["R", "S", "D", "W", "WD"]
    colors_w_none = ['R', 'G', 'Y', 'B', None]
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
            cards_per_color = [len([card for card in hand if card.color == color]) for color in self.colors_w_none]
            # bucket cards into 0, 1-2, 3+
            for color in range(len(cards_per_color)):
                if cards_per_color[color] == 0:
                    cards_per_color[color] = 0
                elif cards_per_color[color] <= 2:
                    cards_per_color[color] = 1
                else:
                    cards_per_color[color] = 2
            # index into 0-242
            return sum([cards_per_color[color] * (3 ** color) for color in range(len(cards_per_color))])
        
        ## - Num of each special cards, per color (0, 1+) ^ (([R OR S OR D]) (4)) = 16
        def bucket_special_cards_per_color():
            num_special = [len([card for card in hand if card.color == color and card.rank in ["R", "S", "D"]]) for color in game.deck.colors]
            # bucket cards into 0, 1+
            for color in range(len(num_special)):
                if num_special[color] == 0:
                    num_special[color] = 0
                else:
                    num_special[color] = 1
            # index into 0-15
            return sum([num_special[color] * (2 ** color) for color in range(len(num_special))])


        ## - Num of cards all the other players have (1, 2-4, 5+)(num_players) = 3(num_players-1)
        def bucket_num_cards_other_players():
            # count number of cards of each player
            num_cards_other = [len(game.deck.player_pile[player]) for player in range(game.num_players) if player != game.current_player]
            # bucket cards into 1, 2-4, 5+
            for player in range(len(num_cards_other)):
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
            return game.deck.colors.index(game.discard_card().color)

        ## - Prev Discard pile color (4) = 4 
        def discard_pile_color_idx_prev():
            return game.deck.colors.index(game.deck.discard_pile[-2].color)

        
        def index_all():
            (aH, bH, cH, dH, eH) = (bucket_num_cards_color(), bucket_special_cards_per_color(), bucket_num_cards_other_players(), discard_pile_color_idx(), discard_pile_color_idx_prev())
            (aN, bN, cN, dN, eN) = (243, 16, 3 * (len(game.deck.player_pile) - 1), 4, 4)
            return aH + (bH * aN) + (cH * aN * bN) + (dH * aN * bN * cN) + (eH * aN * bN * cN * dN)
        
        return index_all()
    
    def bucket_action(self, action, game):
        ## Return index of Q-learning bucket that corresponds to this action
        ## 8 actions
        card, param = action
        #if card is not None:
        #    print("Bucket action", card.color, card.rank, param, game.discard_card().color, game.discard_card().rank)
        #else:
        #    print("Bucket action, none card")
        if card is None:
            idx = 0
        elif card.rank in self.specials:
            idx = self.specials.index(card.rank) + 1
        elif card.color == game.discard_card().color:
            idx = 6
            ## Prioritize color if rank and color are equal
        elif card.rank == game.discard_card().rank:
            idx = 7
        return idx

    def get_action(self, game, state_bucket):
        possibilities = game.possible_actions(game.current_player)
        cards = [p[0] for p in possibilities]
        if random.random() < self.epsilon:
            action = possibilities[random.randint(0, len(possibilities) - 1)]
            #print("Get_action in epsilon", action)
            return self.bucket_action(action, game), action
        else:
            action_q = self.Q[state_bucket]
            action_order = (-action_q).argsort()
            for action_bucket in action_order:
                if action_bucket == 0:
                    if cards[0] is None:
                        move_idx = 0
                    else:
                        continue
                elif action_bucket in [1, 2, 3, 4, 5]:
                    try:
                        move_idx = [c.rank for c in cards].index(self.specials[action_bucket-1])
                    except:
                        continue
                elif action_bucket == 6:
                    try:
                        move_idx = [c.color for c in cards].index(game.discard_card().color)
                    except:
                        continue
                elif action_bucket == 7:
                    try:
                        move_idx = [c.rank for c in cards].index(game.discard_card().rank)
                    except:
                        continue
                else:
                    print("Should not be here, since one of the above action buckets should be satisfied first")
                    print(action_order, possibilities, game.discard_card())
                    raise ValueError
                return action_bucket, possibilities[move_idx]
            print("Nothing worked")
            print(action_order, possibilities, [c.color for c in cards], [c.rank for c in cards], game.discard_card().color, game.discard_card().rank)
            raise ValueError

    def update(self, state, action, reward, next_state):
        ## Update below to allow convergence at end
        self.Q[self.last_state, self.last_action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[self.last_state, self.last_action])
        self.last_state = state
        self.last_action = action

    def reset(self):
        self.last_state = None
        self.last_action = None

    def learn_over_game(self, num_players=2, num_decks=1, agent_idx=0, other_player=random_strategy):
        game = Uno(num_players, Deck(num_decks))
        game.initial_state()
        while game.current_player != agent_idx:
            ## Take other_player strat actions until it's my turn
            other_action = other_player(game)
            game.take_action(other_action)
        state_bucket = self.bucket_state(game)
        while not game.is_over():
            ## Choose action
            action_bucket, action = self.get_action(game, state_bucket)
            game.take_action(action)
            reward = 1 if game.has_won(agent_idx) else 0
            ## Compute the reward and final state (by simulating the other players)
            while game.current_player != agent_idx:
                other_action = random_strategy(game)
                game.take_action(other_action)
                reward = -1 if game.has_won(game.current_player) else reward
            next_state = self.bucket_state(game)
            ## Update Q-values
            self.update(state_bucket, action_bucket, reward, next_state)
            state_bucket = next_state
    
    
    def train(self, timelimit=10, games_limit=None, agent_idx=0, other_player=random_strategy):
        if timelimit is not None:
            start_time = time.time()
            while time.time() - start_time < timelimit:
                self.learn_over_game(agent_idx=agent_idx, other_player=other_player)
        else:
            self.train_games(games_limit, agent_idx, other_player)
    
    #@jit(target_backend='cuda')
    def train_games(self, games_limit, agent_idx, other_player):
        for i in range(games_limit):
            self.learn_over_game(agent_idx=agent_idx, other_player=other_player)            

    def play(self, game, agent_idx=0, other_player=random_strategy):
        while not game.is_over():
            ## Choose action
            state_bucket = self.bucket_state(game)
            _, action = self.get_action(game, state_bucket)
            game.take_action(action)
            while game.current_player != agent_idx:
                other_action = random_strategy(game)
                game.take_action(other_action)
        return game.has_won(agent_idx)

    def simulate(self, num_games=100, agent_idx=0, other_player=random_strategy):
        wins = 0
        for i in range(num_games):
            ## add a discard memory later
            game = Uno(2, Deck(1))
            game.initial_state()
            while game.current_player != agent_idx:
                other_action = other_player(game)
                game.take_action(other_action)
            wins += self.play(game, agent_idx, other_player)
        return wins / num_games

    def save_q(self, filename):
        np.save(filename, self.Q)
    
    def load_q(self, filename):
        self.Q = np.load(filename)
    
    def train_and_save(self, filename, timelimit=10, gamelimit=None, agent_idx=0, other_player=random_strategy):
        self.train(timelimit=timelimit, games_limit=gamelimit, agent_idx=agent_idx, other_player=other_player)
        self.save_q(filename)
    
    def load_and_simulate(self, filename, num_games=100, agent_idx=0, other_player=random_strategy):
        self.load_q(filename)
        return self.simulate(num_games, agent_idx, other_player)