from qlearning import *

if __name__ == "__main__":
    agent = QLearningAgent(2*93312,8)
    agent.train_and_save("qlearning.npy", timelimit=2000)
    #agent.train_and_save("qlearning.npy", timelimit=None, gamelimit=10000)
    print(agent.load_and_simulate("qlearning.npy", 1000))