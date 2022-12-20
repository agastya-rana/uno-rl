from qlearning import *

if __name__ == "__main__":
    agent = QLearningAgent(2*93312,8)
    # agent.train_and_save("qlearning.npy", timelimit=2000)
    start = time.time()
    gamesNum = 1000000
    agent.train_and_save("qlearning.npy", timelimit=None, gamelimit=gamesNum)
    print(agent.load_and_simulate("qlearning.npy", 1000))
    print(f'GamesNum: {gamesNum}, Duration: {time.time()-start}')