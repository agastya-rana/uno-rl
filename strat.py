import itertools as it
import random
import uno
import numpy as np

def random_strategy(game):
    return random.choice(game.possible_actions(game.current_player))


## From below, we see that for 100k games, we get 0.1% error in win percentage.
if __name__ == "__main__":
    num_players = 2
    num_decks = 1
    num_reps = 100 ## Number of reps to find sd
    num_games = 100 ## Simulation size
    res = []
    for i in range(num_reps):
        strategy = [random_strategy for i in range(num_players)]
        results = [0 for i in range(num_players)]
        for i in range(num_games):
            results[uno.simulate_game(strategy, num_players, num_decks)] += 1/num_games
        res.append(results[0])
    print("Number of games is", num_games, "so expect", np.sqrt(1000/num_games), "win percent error")
    print(num_games, np.mean(res), np.std(res))