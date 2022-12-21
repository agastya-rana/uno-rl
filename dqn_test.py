from dqn import *
import time

if __name__ == "__main__":
    num_layers=5
    discard_memory=3
    hidden_dim=100
    lr = 1e-3
    batch = 100 ## Update learning network every _ games
    itera = 150 ## Number of batch updates
    ## The below agent uses the same function approximator as in Q-learning to approximate the action. This was done since
    ## Uno features different possible actions for each hand of cards, and we wanted to allow for an action representation that
    ## was the same across cards.
    agent = DQNAgent(num_layers=num_layers, discard_memory=discard_memory, hidden_dim=hidden_dim, memory_size=10*batch*10, lr=lr)
    ## The below agent is a model with an alternative action approximator. Its action space is each unique card that is playable, along with the draw card move.
    #agent = DQNAltAgent(num_layers=num_layers, discard_memory=discard_memory, hidden_dim=hidden_dim, memory_size=10*batch*10, lr=lr)
    start = time.time()
    simulate_num = 2000
    agent.train_and_save("./DQN_%dlyr_%dhid_%ddis_%.5flr_%dit_%dba" % (num_layers, hidden_dim, discard_memory, lr, itera, batch), num_learning_iter=itera, batch_update=batch, sample_size=batch*7)
    print(agent.load_and_simulate("./DQN_%dlyr_%dhid_%ddis_%.5flr_%dit_%dba" % (num_layers, hidden_dim, discard_memory, lr, itera, batch), num_games=simulate_num))
    print(f'Number of Simulated Games: {simulate_num}, Total Duration: {time.time()-start}')