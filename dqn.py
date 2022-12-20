import numpy as np
import random
import itertools as it
from uno import *
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from strat import *
import time

## Need a learning and a target network network

## Need to define parameters of each network (which should be the same)
## Number of each kinda card in own deck
## How do we capture the agent knowing that the opponent is out of a single color - need only the last #num_players cards on the deck

class DQNAgent():

    def __init__(self, num_players=2, num_decks=1, discard_memory=1, memory_size=10000, other_player=random_strategy, hidden_dim=24, num_layers=2, lr=1e-3):
        ## Create Learning, Target Network
        self.state_size = 56*(discard_memory+1) + (num_players-1)
        self.action_size = 8
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_players = num_players
        self.num_decks = num_decks
        self.discard_memory = discard_memory
        self.learning_rate = lr
        self.other_player = other_player
        self.target_NN = self._build_model()
        self.learning_NN = self._build_model()
        self.memory = deque(maxlen=memory_size)
        self.epsilon = 0.2
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.98

    ranks = [str(n) for n in range(10)] + ["R", "S", "D"]
    colors = ["R", "Y", "B", "G"]
    specials = ["R", "S", "D", "W", "WD"]

    def conv_cards_to_list(self, state, cards):
        for card in cards:
            if card.rank == 'W':
                state[-2] += 1
            elif card.rank == 'WD':
                state[-1] += 1
            else:
                state[self.ranks.index(card.rank) + 13*self.colors.index(card.color)] += 1
        return state

    def conv_game_to_state(self, game):
        """ Returns a list containing the entire state"""
        ## Tensor contains one 4 x 13 + 2 = 56 list containing number of each card in own hand 
        ## Then contains number of cards in others hands: num_players - 1 normalized by 7
        ## Then contains the discard pile last x elements ## 56 x 7 list of one hot encodings - could try to boil this down by
        ## ignoring rank unless special card
        temp = [0 for i in range(56)]
        state = self.conv_cards_to_list(temp.copy(), game.deck.player_pile[game.current_player])
        [state.extend(self.conv_cards_to_list(temp.copy(), [game.deck.discard_pile[-(i+1)]])) for i in range(self.discard_memory)]
        state.extend([len(game.deck.player_pile[player_idx])/7 for player_idx in range(game.num_players) if player_idx != game.current_player])
        return state

    def filter_action(self, action_idxs, game):
        ## Given an ordered list of action_idxs preferred, find the topmost one that works for the game
        possibilities = game.possible_actions(game.current_player)
        cards = [p[0] for p in possibilities]
        for action_bucket in action_idxs:
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
    
    def decide_action(self, nn_input, game):
        if np.random.rand() <= self.epsilon:
            ## Choose random action in possible action space
            possibilities = game.possible_actions(game.current_player)
            action = possibilities[random.randint(0, len(possibilities) - 1)]
            return self.bucket_action(action, game), action
        nn_input = np.asarray(nn_input).reshape(1, -1)
        act_values = self.target_NN.predict(nn_input)[0]
        action_order = (-act_values).argsort()
        return self.filter_action(action_order, game)
        ## Now need to go through these actions and filter out those that are possible
        ## Check online for alternate solutions to this validation problem

    def replay(self, sample_size):
        minibatch = random.sample(self.memory, sample_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.asarray(state).reshape(1, -1)
            next_state = np.asarray(next_state).reshape(1, -1)
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.learning_NN.predict(next_state)[0]))
            target_f = self.learning_NN.predict(state)
            target_f[0][action] = target
            self.learning_NN.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def copy_train_learn(self):
        self.target_NN.set_weights(self.learning_NN.get_weights())
    
    def train(self, agent_idx=0, num_learning_iter=100, batch_update=1000, sample_size=1000):
        ## Train by running sims on the target network, storing in database and then updating learning
        ## Figure out how to deal with softmaxed invalid outputs
        assert sample_size < batch_update*10
        for learn_iter in range(num_learning_iter):
            print("Iter", learn_iter)
            for game in range(batch_update):
                game = Uno(self.num_players, Deck(self.num_decks, self.discard_memory))
                game.initial_state()
                nn_input = self.conv_game_to_state(game)
                while game.current_player != agent_idx:
                    ## Take other_player strat actions until it's my turn
                    other_action = other_player(game)
                    game.take_action(other_action)
                while not game.is_over():
                    ## Choose action
                    action_bucket, action = self.decide_action(nn_input, game)
                    game.take_action(action)
                    reward = 1 if game.has_won(agent_idx) else 0
                    while game.current_player != agent_idx:
                        other_action = random_strategy(game)
                        game.take_action(other_action)
                        reward = -1 if game.has_won(game.current_player) else reward
                    next_state = self.conv_game_to_state(game)
                    self.memory.append((nn_input, action_bucket, reward, next_state, game.is_over()))
                    nn_input = next_state
            print("Sims done, replay starting")
            self.replay(sample_size)
            ## Copy training to learning
            self.copy_train_learn()
    
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(self.hidden_dim, input_shape=[self.state_size,], activation='relu'))
        for i in range(self.num_layers-1):
            model.add(Dense(self.hidden_dim, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def save_NN(self, folder_name):
        self.target_NN.save(folder_name)
    
    def load_NN(self, folder_name):
        self.target_NN = load_model(folder_name)
    
    def train_and_save(self, folder_name, agent_idx=0, num_learning_iter=100, batch_update=1000, sample_size=1000):
        self.train(agent_idx=0, num_learning_iter=num_learning_iter, batch_update=batch_update, sample_size=sample_size)
        self.save_NN(folder_name)
    
    def load_and_simulate(self, folder_name, num_games=100, agent_idx=0, other_player=random_strategy):
        self.load_NN(folder_name)
        return self.simulate(num_games, agent_idx, other_player)

    def play(self, game, agent_idx=0, other_player=random_strategy):
        self.other_player = other_player
        while not game.is_over():
            ## Choose action
            nn_in = self.conv_game_to_state(game)
            _, action = self.decide_action(nn_in, game)
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