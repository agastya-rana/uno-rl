from qlearning import *

if __name__ == "__main__":
    agent = QLearningAgent(93312,8)
    agent.train_and_save("qlearning.npy", 2000)
    print(agent.load_and_simulate("qlearning.npy", 1000))