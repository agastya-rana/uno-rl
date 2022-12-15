import itertools as it


class Uno(ABC):

    def __init__(self, num_players):
        self.direction = 1
        self.deck = Deck()
        pass
    
    def initial_state(self, Deck, num_players=2):
        deal_deck()
    


class Deck():

    colors = ['R', 'G', 'Y', 'B', None]
    colored_ranks = [str(n) for n in range(10)] + [str(n) for n in range(1, 10)] + ["D", "R", "S"]*2 ## Draw2, Reverse, Skip
    uncolor_ranks = ["W", "WD"]*4 ## Wild, wild draw 4

    def __init__(self, deck_copies):
        ## Initializes a base deck
        self.player_pile = []
        self.discard_pile = []
        self.draw_pile = self.base_cards(deck_copies)
    
    def deal_deck(self, num_players):
        pass

    def base_cards(self, deck_copies):
        cards = []
        for copy in range(copies):
            cards.extend(map(lambda c: Card(*c), it.product(self.colored_ranks, self.colors)))
            cards.extend([Card(c, None) for c in self.uncolor_ranks])
        return cards

    def play_card(self, card, player_idx):
        if self.is_valid_play(card, player_idx):
            try:
                self.player_pile[player_idx].remove(card)
                discard_pile.append(card)
            except:
                raise ValueError("Card not in player's hand")
        else:
            raise ValueError("Card cannot be legally played by player.")
        

    def is_valid_play(self):
        pass
    


class Card():

    def __init__(self, rank, color=None):
        self._rank = rank
        self._color = color   

    def rank(self):
        return self._rank

    def color(self):
        return self._color
