from pathlib import Path
import math
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from replaybuffer import ReplayBuffer
from tqdm import tqdm
from ActorCritic import ACAgent
from TD3 import TD3Agent
from BipedalWalker import BWAgent
import pickle

def evaluate_action(agent : BWAgent, seed : int):
    reward_list = []
    for i in range(10):
        seed = seed + i
        np.random.seed(seed=seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.use_deterministic_algorithms = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        state = env.reset()
        done = False
        Reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            Reward += reward
            state = next_state
        reward_list.append(Reward)
    return reward_list


if __name__ == "__main__":
    path = "./BW1/"
    env = gym.make('BipedalWalker-v3')
    save_step = 2000
    step_list = [i + 1 for i in range(save_step * 100) if (i + 1) % save_step == 0]
    reward_lst=[[] for _ in range(100)]
    n = env.observation_space.shape[0]
    m = env.action_space.shape[0]
    dim = 256
    actor_H = 4
    critic_H = 4
    alpha = 0.0003
    gamma = 0.99
    sigma = 0.1
    tau = 0.005
    sigma_target = 0.2
    c = 0.5
    d = 2
    action_max = env.action_space.high[0]
    action_min = env.action_space.low[0]
    T_expl = 50000
    # agent = ACAgent(n, m, dim, actor_H, critic_H, action_max, action_min, alpha, gamma, sigma)
    # agent = TD3Agent(n, m, dim, actor_H, critic_H, action_max, action_min, alpha, gamma, sigma, tau, sigma_target, c, d)
    agent = BWAgent(n, m, dim, actor_H, critic_H, action_max, action_min, alpha, gamma, sigma, tau, sigma_target, c, d)

    for i in range(5):
        for step in step_list:
            seed = i

            load_path = path + str(i+1) + "_" + str(step) + "_" + str(seed)

            agent.load_models(load_path)
            reward_list = evaluate_action(agent, seed)
            reward_lst[int(step / save_step - 1)].extend(reward_list)

    points = reward_lst
    steps = ([save_step * (i + 1) for i in range(100)])

    with open('BW1.pkl', 'wb') as f:
        pickle.dump(points, f)

    medians = [np.median(rewards) for rewards in points]
    q1s = [np.percentile(rewards, 25) for rewards in points]
    q3s = [np.percentile(rewards, 75) for rewards in points]

    # Plot the learning curve with quartiles
    plt.plot(steps, medians)
    plt.fill_between(steps, q1s, q3s, alpha=0.3)

    plt.xlabel('Training Step')
    plt.ylabel('Reward')
    plt.title('Evaluation results in BipedalWalker-v3')
    plt.legend()
    plt.grid(True)
    plt.show()
