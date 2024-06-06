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
        print((self.action_space.low+self.action_space.high)/2)
        return ((self.action_space.low+self.action_space.high)/2)
    def select_exploratory_action(self, state):
        print(type(self.action_space.sample()))
        return self.action_space.sample()
    def save_models(self, path):
        torch.save(self.action_space, path)
    def load_models(self, path):
        self.action_space=torch.load(path)

env = gym.make('Pendulum-v0')

#乱数固定
for i in range(5):
    seed=1009+i
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    agent = RandomAgent(env.action_space)
    state = env.reset()
    for t in tqdm(range(10000)):
        env.render()
        action = agent.select_exploratory_action(state)
        next_state, reward, done, info = env.step(action)
        agent.train(state, action, next_state, reward, done)
        state = next_state
        if t>=100 and (t + 1) % 1000 == 0:
            save_path = path + "_" + str(i+1) + "_" + str(t+1) + "_" + str(seed) + ".pth"
            agent.save_models(save_path)
    env.close()
