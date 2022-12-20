from qlearning import *

if __name__ == "__main__":
    agent = QLearningAgent(2*93312,8)
    # agent.train_and_save("qlearning.npy", timelimit=2000)
    start = time.time()
    gamesNum = 1000000
    agent.train_and_save("qlearning.npy", timelimit=None, gamelimit=gamesNum)
    print(f'Train Duration: {time.time()-start}')
    print("Win percentage over 100k games", agent.load_and_simulate("qlearning.npy", 100000))
    print(f'Train over {int(gamesNum/1000)}k Games, Duration: {time.time()-start}')