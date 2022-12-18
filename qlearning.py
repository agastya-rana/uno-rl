import time
from uno import *
import random
import numpy as np
from bisect import bisect

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

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state):
        self.Q[self.last_state, self.last_action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[self.last_state, self.last_action])
        self.last_state = state
        self.last_action = action

    def reset(self):
        self.last_state = None
        self.last_action = None