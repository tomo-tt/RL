import gym
import numpy as np
import torch

from ActorCritic import ACAgent
from TD3 import TD3Agent
from BipedalWalker import BWAgent


if __name__ == "__main__":
    batch_size = 256
    step_size = 250000
    buffer_size = 250000
    num_trains = 1
    path = "./folder_bw4/"

    for i in range(1):
        seed = i
        np.random.seed(seed=seed)
        env = gym.make('BipedalWalker-v3')
        np.random.seed(seed=seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.use_deterministic_algorithms = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

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
        # agent = ACAgent(n, m, dim, action_max, action_min, alpha, gamma, sigma)
        # agent = TD3Agent(n, m, dim, actor_H, critic_H, action_max, action_min, alpha, gamma, sigma, tau, sigma_target, c, d)
        agent = BWAgent(n, m, dim, actor_H, critic_H, action_max, action_min, alpha, gamma, sigma, tau, sigma_target, c, d)

        load_path = path + str(i+1) + "_" + str(200000) + "_" + str(seed)
        agent.load_models(load_path)

        state = env.reset()
        done = False
        while not done:
            env.render()
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
