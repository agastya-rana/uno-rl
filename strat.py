import itertools as it
import random
import uno

def random_strategy(game):
    return random.choice(game.possible_actions(game.current_player))

if __name__ == "__main__":
    num_players = 2
    num_decks = 1
    num_games = 10000
    strategy = [random_strategy for i in range(num_players)]
    results = [0 for i in range(num_players)]
    for i in range(num_games):
        results[uno.simulate_game(strategy, num_players, num_decks)] += 1
    print(results)