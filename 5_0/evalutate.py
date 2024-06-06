from tqdm import tqdm
import gym
from agent import Agent
import math
import random
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

path="./models/weight"

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
    for i in range(10):
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
step_list = [i+1 for i in range(10000) if (i+1)%1000==0]
reward_lst=[[],[],[],[],[],[],[],[],[],[]]

count=0
for i in range(5):
    for step in step_list:
        seed = 1009 + i
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        agent = RandomAgent(env.action_space)
        state = env.reset()
        load_path = "./models/weight_" + str(i+1) + "_" + str(step) + "_" + str(seed) + ".pth"

        agent = RandomAgent(env.action_space)
        agent.load_models(load_path)
        reward_list = evaluate_action(agent, seed)
        reward_lst[int(step/1000-1)].extend(reward_list)
        env.close()

points = reward_lst
fig, ax = plt.subplots()
bp = ax.boxplot(points)
ax.set_xticklabels(["1000", "2000", "3000", "4000", "5000", "6000", "7000", "8000", "9000", "10000"])
plt.xlabel("step")
plt.ylabel("total reward")
plt.show()
