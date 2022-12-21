# Reinforcement Learning for Uno
The following is a Reinforcement Learning project applied to Uno for CS474, created by Elven Shum and Agastya Rana.

# Summary

We worked in a team of two by splitting up the work as follows:
1. Both of us worked together on the implementation of the Uno game
2. Elven Shum worked on the Q-Learning Algorithm (devising approximators, hyperparameter tuning)
3. Agastya Rana worked on the DQN Algorithm (implementation, devising appropriate state + action space, hyperparameter tuning)

We believe that the imperfect information and highly luck-based aspects to this game severly limits the amount of `strategy' involved, which is what explains our low win margins, even after different approximators and hyperparameters were experimented with.

Note that due to the large size of the dense NN used for DQN, even simulating the model requires ~1second/game; therefore, we tested just 4,000 games for DQN to generate an estimate with 0.5% precision.


# How to Run
Run `python qlearning_test.py` to simulate 100,000 games of Elven's Q-learning agent, and run `python dqn_test.py` to simulate
1000 games of Agastya's DQN agent.

Since these RL models took a lot of time to train, the trained information for both models (numpy array of Q-values for Q-learning (qlearning.npy) and NN model for DQN (in the zip file)) is submitted alongside the code. Code required to run these models is provided in `qlearning_test.py` and `dqn_test.py` respectively.


# Success Rate, Discusison
We quickly realized through these simulations that contrary to our expectations of a reasonable amount
of strategy in Uno (as our families had led us to believe!), Uno gameplay was highly stochastic, which
definitely limited the performance of our agents to low margins (< 5% over random). Furthermore, due to
the lack of a reasonable heuristic-based comparative agent, we had to resort to a random agent to evaluate
the performance of our algorithms as applied to the game of Uno.  We noted that we
required the simulation of 100,000 games to obtain a win percentage precise to 0.1%, and achieved the following performance with our models: Q-Learning: 51.8% +/- 0.1%, DQN: 53.5% +/- 1%.

We believe, this result is actually quite reasonable. As a benchmark, although a [stanford paper on beating Uno with RL](https://web.stanford.edu/class/aa228/reports/2020/final79.pdf) gets a winrate of 63.5%, their DQN agent **can see the opponent's cards**, which we believe gives it a serious (and perhaps unreasonable) advantage. Our model, to properly simulate a real uno player, can only access it's own cards. Thus, we believe given the highly stochastic nature of Uno, our game play percentage is reasonably successful.



## File Structure
* `README.md` - This file
* `uno.py` - Our implmentation of the Uno game
* `qlearning.py` - Our implementation of the Q-Learning agent
* `qlearning_test.py` - Tests for the Q-Learning agent
* `dqn.py` - Our implementation of the DQN agent
* `dqn_test.py` - Tests for the DQN agent
* `run.py` - Runs the Q-Learning agent against a random agent
* `strats.py` - Our implementation of the random and greedy agents
