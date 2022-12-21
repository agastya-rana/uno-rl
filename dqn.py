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
from multiprocessing import Pool
from functools import partial

## Need a learning and a target network network

## Need to define parameters of each network (which should be the same)
## Number of each kinda card in own deck
## How do we capture the agent knowing that the opponent is out of a single color - need only the last #num_players cards on the deck

class DQNAgent():

    def __init__(self, num_players=2, num_decks=1, discard_memory=1, memory_size=10000, other_player=random_strategy, hidden_dim=24, num_layers=2, lr=1e-3):
        ## Create Learning, Target Network
        self.state_size = 54*(discard_memory+1) + (num_players-1)
        self.action_size = 8
        self.unique_card = 54
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
        self.epsilon_decay = 0.99
        self.gamma = 0.98

    ranks = [str(n) for n in range(10)] + ["R", "S", "D"]
    colors = ["R", "Y", "B", "G"]
    specials = ["R", "S", "D", "W", "WD"]
    act_dict = {}
    for i in range(52):
        act_dict[i] = (Card(ranks[i%13], colors[i//13]), None)
    for i in range(52, 56):
        act_dict[i] = (Card("W", None), colors[i%4])
    for i in range(56, 60):
        act_dict[i] = (Card("WD", None), colors[i%4])
    act_dict[60] = (None, None)

    state_dict = {}
    for i in range(52):
        state_dict[Card(ranks[i%13], colors[i//13])] = i
    for c in colors+[None]:
        state_dict[Card("W", c)] = 52
        state_dict[Card("WD", c)] = 53

    def conv_game_to_state(self, game):
        """ Returns a state_size numpy containing the entire state"""
        ## Tensor contains one 4 x 13 + 2 = 54 elements containing number of each card in own hand 
        ## Then contains the discard pile last x elements ## 54 times x elements 
        ## Then contains number of cards in others hands: num_players - 1 element normalized by 7
        state = np.zeros(self.state_size)
        ## Add in my own cards
        cards = game.deck.player_pile[game.current_player]
        for card in cards:
            state[self.state_dict[card]] += 1
        for i in range(self.discard_memory):
            state[self.unique_card*i + self.state_dict[game.deck.discard_pile[-(i+1)]]] = 1
        state[-1] = len(game.deck.player_pile[1-game.current_player])/7
        ## TODO: change to following to allow for more than 2 players
        #for i in range(self.num_players-1):
        #    state[-(i+1):] = [len(game.deck.player_pile[player_idx])/7 for player_idx in range(game.num_players) if player_idx != game.current_player]
        return state.reshape(1, -1)

    def bucket_action(self, action, game):
        ## Return index of Q-learning bucket (8 buckets) that corresponds to this action
        card, param = action
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
        possibilities = game.possible_actions(game.current_player)
        if possibilities[0][0] is None:
            return 0, (None, None)
        if np.random.rand() <= self.epsilon:
            ## Choose random action in possible action space
            action = possibilities[random.randint(0, len(possibilities) - 1)]
            return self.bucket_action(action, game), action
        act_values = self.target_NN.predict(nn_input, verbose = 0)[0]
        cards = [p[0] for p in possibilities]
        iters = 0
        while True:
            if iters >= 100:
                sample_act = (-act_values).argsort()[iters-100]
                if iters > 150:
                    raise ValueError
            else:
                sample_act = np.random.choice(self.action_size, p=act_values)
            if sample_act == 0:
                    move_idx = 0
                    break
            elif sample_act in [1, 2, 3, 4, 5]:
                try:
                    move_idx = [c.rank for c in cards].index(self.specials[sample_act-1])
                    break
                except:
                    continue
            elif sample_act == 6:
                try:
                    move_idx = [c.color for c in cards].index(game.discard_card().color)
                    break
                except:
                    continue
            elif sample_act == 7:
                try:
                    move_idx = [c.rank for c in cards].index(game.discard_card().rank)
                    break
                except:
                    continue
            iters += 1
        #print("hi")
        return sample_act, possibilities[move_idx]
        #action_order = (-act_values).argsort()
       
    def replay(self, sample_size):
        minibatch = random.sample(self.memory, sample_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma*np.amax(self.learning_NN.predict(next_state, verbose = 0)[0]))
            target_f = self.learning_NN.predict(state, verbose = 0)
            target_f[0][action] = target
            self.learning_NN.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def copy_train_learn(self):
        self.target_NN.set_weights(self.learning_NN.get_weights())
    
    def train(self, agent_idx=0, num_learning_iter=100, batch_update=1000, sample_size=1000, fname=None):
        ## Train by running sims on the target network, storing in database and then updating learning
        ## Figure out how to deal with softmaxed invalid outputs
        assert sample_size < batch_update*10
        #start_time=0
        for learn_iter in range(num_learning_iter):
            print("Iter", learn_iter)
            #print(time.time()-start_time)
            for g in range(batch_update):
                game = Uno(self.num_players, Deck(self.num_decks, self.discard_memory))
                game.initial_state()
                nn_input = self.conv_game_to_state(game)
                while game.current_player != agent_idx:
                    ## Take other_player strat actions until it's my turn
                    other_action = other_player(game)
                    game.take_action(other_action)
                while not game.is_over():
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
            #start_time = time.time()
            print("Sims done, replay starting for iter", learn_iter)
            self.replay(sample_size)
            self.copy_train_learn()
            if learn_iter == 100 or learn_iter == 150:
                self.save_NN(fname)

    
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
        self.train(agent_idx=0, num_learning_iter=num_learning_iter, batch_update=batch_update, sample_size=sample_size, fname=folder_name)
        self.save_NN(folder_name)
    
    def load_and_simulate(self, folder_name, num_games=100, agent_idx=0, other_player=random_strategy):
        self.load_NN(folder_name)
        print("Loaded")
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

    def simulate_par_helper(self, agent_idx, other_player, num_games):
        game = Uno(2, Deck(1, discard_memory=self.discard_memory))
        game.initial_state()
        print(num_games)
        while game.current_player != agent_idx:
            other_action = other_player(game)
            game.take_action(other_action)
        wins = self.play(game, agent_idx, other_player)
        return wins
    
    def simulate_par(self, num_games=100, agent_idx=0, other_player=random_strategy):
        p = Pool(60)
        partial_simulate = partial(self.simulate_par_helper, self, agent_idx, other_player)
        results = p.map(self.simulate_par_helper, range(num_games))
        print(results)
        return sum(results)/num_games
    
    def simulate(self, num_games=100, agent_idx=0, other_player=random_strategy):
        ep = self.epsilon
        self.epsilon = 0
        wins = 0
        for i in range(num_games):
            game = Uno(2, Deck(1, discard_memory=self.discard_memory))
            game.initial_state()
            while game.current_player != agent_idx:
                other_action = other_player(game)
                game.take_action(other_action)
            wins += self.play(game, agent_idx, other_player)
            if i % 100 == 0:
                print(i)
        self.epsilon = ep
        return wins/num_games


class DQNAltAgent(DQNAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_size = 52+8+1 ## all cards (minus wilds) + wilds + draw card move
        self.target_NN = self._build_model()
        self.learning_NN = self._build_model()
    
    def decide_action(self, nn_input, game):
        possibilities = game.possible_actions(game.current_player)
        if possibilities[0][0] is None:
            return 60, (None, None)
        if np.random.rand() <= self.epsilon:
            ## Choose random action in possible action space
            action = possibilities[random.randint(0, len(possibilities) - 1)]
            card, p = action
            try:
                bucket = self.ranks.index(card.rank) + self.colors.index(card.color)*13
            except:
                if card.rank == "W":
                    bucket = 52 + self.colors.index(p)
                elif card.rank == "WD":
                    bucket = 56 + self.colors.index(p)
                else:
                    raise KeyError
            return bucket, action
        act_values = self.target_NN.predict(nn_input, verbose = 0)[0]
        while True:
            sample_act = np.random.choice(self.action_size, p=act_values)
            for p in possibilities:
                try:
                    if self.act_dict[sample_act] == p:
                        return sample_act, p
                except:
                    print(p)
                    raise ValueError


        