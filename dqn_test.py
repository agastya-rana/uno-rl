from dqn import *
import time

if __name__ == "__main__":
    agent = DQNAgent()
    # agent.train_and_save("qlearning.npy", timelimit=2000)
    start = time.time()
    gamesNum = 100
    #agent.train_and_save("./qlearning", num_learning_iter=5, batch_update=10, sample_size=50)
    print(agent.load_and_simulate("./qlearning", gamesNum))
    print(f'GamesNum: {gamesNum}, Duration: {time.time()-start}')