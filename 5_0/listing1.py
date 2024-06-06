from tqdm import tqdm
import gym
from agent import Agent
import math
import random
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

path="./models/abc.txt"

class RandomAgent(Agent):
    def __init__(self, action_sapce):
        self.action_space=action_sapce
    def select_action(self, state):
        return ((self.action_space.low+self.action_space.high)/2)
    def select_exploratory_action(self, state):
        return self.action_space.sample()
    def save_models(self, path):
        torch.save(self.action_space, path)
    def load_models(self, path):
        self.action_space=torch.load(path)


def evaluate_action(agent:RandomAgent, start_seed:int):
    reward_list = []
    for i in range(100):
        seed=i+start_seed
        env = gym.make('Pendulum-v0')
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        state = env.reset()
        Reward=0
        for t in tqdm(range(1000)):
            # env.render()
            action = agent.select_action(state)
            # print(action)
            next_state, reward, done, info = env.step(action)
            Reward+=reward
            state = next_state
            if done:
                state = env.reset()
                break
        env.close()
        # print(Reward)
        reward_list.append(Reward)
    return reward_list

env = gym.make('Pendulum-v0')

#乱数固定
seed=1009
env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

agent = RandomAgent(env.action_space)
state = env.reset()
for t in tqdm(range(10)):
    env.render()
    action = agent.select_exploratory_action(state)
    next_state, reward, done, info = env.step(action)
    agent.train(state, action, next_state, reward, done)
    state = next_state
    if done:
        state = env.reset()
        print(agent.action_space)
        agent.save_models(path)
        break
env.close()

agent2 = RandomAgent(env.action_space)
agent2.load_models(path)
print(agent2.action_space)
# evaluate_action(agent)


env = gym.make('Pendulum-v0')
train_seed = 0
env.seed(train_seed)
env.action_space.seed(train_seed)
env.observation_space.seed(train_seed)
state = env.reset()

rdict={}

for num in tqdm(range(5)):
    if num==0:
        agent = RandomAgent(env.action_space)
    else:
        agent.load_models(path+str(num-1))
    action = agent.select_exploratory_action(state)
    next_state, reward, done, info = env.step(action)
    agent.train(state, action, next_state, reward, done)
    state = next_state
    rlist=evaluate_action(agent, 10)
    rdict[str(num)]=rlist

    npath=path+str(num)
    agent.save_models(npath)

print(rdict)

points = (rdict["0"], rdict["1"], rdict["2"], rdict["3"], rdict["4"])
fig, ax = plt.subplots()
bp = ax.boxplot(points)
ax.set_xticklabels(["0", "1", "2", "3", "4"])
plt.xlabel("iteration")
plt.ylabel("total reward")
plt.show()
