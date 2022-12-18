import itertools as it
import random
from abc import ABC, abstractmethod


def simulate_game(strategy, num_players, num_decks=1, verbose=False):
    game = Uno(num_players, Deck(num_decks))
    game.initial_state()
    while True:
        if verbose:
            print("Player {}'s hand: {}".format(game.current_player, game.deck.player_pile[game.current_player]))
            print("Discard pile: {}".format(game.deck.discard_pile[-1]))
        action = strategy[game.current_player](game) # Strategy is a list of strategies, one for each player, and takes in the whole game state
        game.take_action(action, game.current_player)
        ## TODO: change above line and take_action function to not take in a player since it should always be the current player
        if game.has_won(game.current_player):
            if verbose:
                print("Player {} wins!".format(game.current_player))
                print("Discard pile: {}".format(game.deck.discard_pile))
            return game.current_player
    
class Uno(ABC):
    def __init__(self, num_players, deck):
        self.direction = 1
        self.current_player = 0
        self.num_players = num_players
        self.deck = deck
        pass

    def initial_state(self):
        self.deck.deal_deck(self.num_players)
        self.deck.discard_pile.append(self.deck.draw_pile.pop())
        while not self.deck.discard_pile[-1].rank in ["W", "WD", "D", "R", "S"]:
            self.deck.draw_pile.append(self.deck.discard_pile.pop())
            self.deck.shuffle_deck()
            self.deck.discard_pile.append(self.deck.draw_pile.pop())

    def take_action(self, action, player_idx):
        card, param = action
        if card == None:    # Player has no valid plays
            self.deck.draw_cards(1, player_idx)
            self.next_player()
            return None
        if not self.is_valid_play(card, player_idx):    # Player tries to play an invalid card
            raise ValueError("Card cannot be legally played by player.")
        if param != None:   # Player plays a wild card
            card.color = param
        self.deck.play_card(card, player_idx)
        self.effect(card, player_idx)
        self.next_player()

    def possible_actions(self, player_idx):
        actions = []
        cards = [card for card in self.deck.player_pile[player_idx] if self.is_valid_play(card, player_idx)]    
        for card in cards:
            if card.rank in ["W", "WD"]:
                actions.extend([(card, color) for color in self.deck.colors])
            else:
                actions.append((card, None))
        if len(actions) == 0:
            actions.append((None, None))
        return actions

    def next_player(self):
        self.current_player = self.calc_next_player(self.current_player)

    def calc_next_player(self, player_idx):
        return (player_idx + self.direction) % self.num_players
    
    def is_valid_play(self, card, player_idx):
        if card.color == self.deck.discard_pile[-1].color or card.rank == self.deck.discard_pile[-1].rank: # Deal with colored cards
            return True
        elif card.color == None:  # Deal with wild cards
            return True
        else:
            return False
    
    def effect(self, card, player_idx):
        ## Player who draws cards can play
        if card.rank == "D":
            self.deck.draw_cards(2, self.calc_next_player(player_idx))
        elif card.rank == "R":
            self.direction *= -1
        elif card.rank == "S":
            self.next_player()
        elif card.rank == "WD":
            self.deck.draw_cards(4, self.calc_next_player(player_idx))
        elif card.rank == "W":
            pass

    def has_won(self, player_idx):
        return len(self.deck.player_pile[player_idx]) == 0
    
    def is_over(self):
        return any([len(self.deck.player_pile[player_idx]) == 0 for player_idx in range(self.num_players)])

class Deck():
    colors = ['R', 'G', 'Y', 'B', None]
    colored_ranks = [str(n) for n in range(10)] + [str(n) for n in range(1, 10)] + ["D", "R", "S"]*2 ## Draw2, Reverse, Skip
    uncolor_ranks = ["W", "WD"]*4 ## Wild, wild draw 4

    def __init__(self, deck_copies):
        ## Initializes a base deck
        self.player_pile = []
        self.discard_pile = []
        self.draw_pile = self.base_cards(deck_copies)
        self.shuffle_deck()
        
    def base_cards(self, deck_copies):
        cards = []
        for copy in range(deck_copies):
            cards.extend(map(lambda c: Card(*c), it.product(self.colored_ranks, self.colors)))
            cards.extend([Card(c, None) for c in self.uncolor_ranks])
        return cards
        
    def shuffle_deck(self):
        random.shuffle(self.draw_pile)
    
    def deal_deck(self, num_players):
        self.player_pile = [[] for player in range(num_players)]
        for player in range(num_players):
            self.draw_cards(7, player)

    def draw_cards(self, num_cards, player):
        cards = []
        for card in range(num_cards):
            if len(self.draw_pile) == 0:
                self.draw_pile = self.discard_pile[:-1]
                self.discard_pile = [self.discard_pile[-1]]
                self.shuffle_deck()
            cards.append(self.draw_pile.pop())
        self.player_pile[player].extend(cards)
        return cards

    def play_card(self, card, player_idx):
        try:
            self.player_pile[player_idx].remove(card)
            self.discard_pile.append(card)
        except:
            raise ValueError("Card not in player's hand")
class Card():
    def __init__(self, rank, color=None):
        self.rank = rank
        self.color = color
