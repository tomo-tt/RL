import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from agent import Agent
from replaybuffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, n, m, dim, H, action_max, action_min):
        super(Actor, self).__init__()
        self.H = H
        self.action_max = action_max
        self.action_min = action_min

        self.actor_layers = nn.ModuleList()
        self.actor_layers.append(nn.Linear(n, dim))
        for _ in range(H - 2):
            self.actor_layers.append(nn.Linear(dim, dim))
        self.actor_layers.append(nn.Linear(dim, m))

    def forward(self, state):
        actor_output = state
        for i in range(self.H - 1):
            actor_output = F.relu(self.actor_layers[i](actor_output))
        actor_output = (self.action_max + self.action_min) / 2 + (self.action_max - self.action_min) / 2 * torch.tanh(self.actor_layers[self.H - 1](actor_output))
        return actor_output


class Critic(nn.Module):
    def __init__(self, n, m, dim, H):
        super(Critic, self).__init__()
        self.H = H

        self.critic_layers = nn.ModuleList()
        self.critic_layers.append(nn.Linear(n + m, dim))
        for _ in range(H - 2):
            self.critic_layers.append(nn.Linear(dim, dim))
        self.critic_layers.append(nn.Linear(dim, 1))

    def forward(self, state, action):
        critic_output = torch.cat((state, action), dim=1)
        for i in range(self.H - 1):
            critic_output = F.relu(self.critic_layers[i](critic_output))
        critic_output = self.critic_layers[self.H - 1](critic_output)
        return critic_output


class TD3Agent3(Agent):
    def __init__(self,
                 n,
                 m,
                 dim,
                 actor_H,
                 critic_H,
                 action_max,
                 action_min,
                 alpha,
                 gamma,
                 sigma,
                 tau,
                 sigma_target,
                 c,
                 d
                 ):
        self.actor = Actor(n, m, dim, actor_H, action_max, action_min).to(device)
        self.target_actor = Actor(n, m, dim, actor_H, action_max, action_min).to(device)
        self.critic = Critic(n, m, dim, critic_H).to(device)
        self.critic2 = Critic(n, m, dim, critic_H).to(device)
        self.target_critic = Critic(n, m, dim, critic_H).to(device)
        self.target_critic2 = Critic(n, m, dim, critic_H).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=alpha)

        self.action_max = action_max
        self.action_min = action_min
        self.action_num = m
        self.gamma = gamma
        self.sigma = sigma
        self.tau = tau
        self.sigma_target = sigma_target
        self.c = c
        self.d = d

    def save_model(self, path):
        actor_path = path + "actor.pth"
        target_actor_path = path + "target_actor.pth"
        critic_path = path + "critic.pth"
        critic2_path = path + "critic2.pth"
        target_critic_path = path + "target_critic.pth"
        target_critic2_path = path + "target_critic2.pth"

        torch.save(self.actor.to('cpu').state_dict(), actor_path)
        torch.save(self.target_actor.to('cpu').state_dict(), target_actor_path)
        torch.save(self.critic.to('cpu').state_dict(), critic_path)
        torch.save(self.critic2.to('cpu').state_dict(), critic2_path)
        torch.save(self.target_critic.to('cpu').state_dict(), target_critic_path)
        torch.save(self.target_critic2.to('cpu').state_dict(), target_critic2_path)

        self.actor.to(device)
        self.target_actor.to(device)
        self.critic.to(device)
        self.critic2.to(device)
        self.target_critic.to(device)
        self.target_critic2.to(device)

    def load_models(self, path):
        actor_path = path + "actor.pth"
        target_actor_path = path + "target_actor.pth"
        critic_path = path + "critic.pth"
        critic2_path = path + "critic2.pth"
        target_critic_path = path + "target_critic.pth"
        target_critic2_path = path + "target_critic2.pth"

        self.actor.load_state_dict(torch.load(actor_path))
        self.target_actor.load_state_dict(torch.load(target_actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        self.critic2.load_state_dict(torch.load(critic2_path))
        self.target_critic.load_state_dict(torch.load(target_critic_path))
        self.target_critic2.load_state_dict(torch.load(target_critic2_path))

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def select_exploratory_action(self, state):
        state = torch.FloatTensor(state).to(device)
        pi = self.actor(state).cpu().data.numpy().flatten()
        noise = np.random.normal(0, (self.action_max - self.action_min) * self.sigma / 2, size=self.action_num)
        beta = pi + noise
        return np.clip(beta, self.action_min, self.action_max)

    def soft_update_target_networks(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_noise_to_target_actor(self, target_actions):
        noise = torch.clamp(torch.randn_like(target_actions) * self.sigma_target, -self.c, self.c).to(device)
        perturbed_actions = torch.clamp(target_actions + noise, self.action_min, self.action_max)
        return perturbed_actions

    def train(self, states, actions, next_states, rewards, dones, current_step):
        states = torch.FloatTensor(np.array(states)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards).reshape(-1, 1)).to(device)
        dones = torch.FloatTensor(np.array(dones).reshape(-1, 1)).to(device)

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_actions = self.add_noise_to_target_actor(next_actions)
            next_q_values_target = torch.min(self.target_critic(next_states, next_actions),
                                                self.target_critic2(next_states, next_actions))
            next_q_values = rewards + (1 - dones) * self.gamma * next_q_values_target

        q_values = self.critic(states, actions)
        q_values2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(q_values, next_q_values)
        critic2_loss = F.mse_loss(q_values2, next_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        actor_loss = - self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update_target_networks()


if __name__ == "__main__":
    print(device)
    batch_size = 256
    step_size = 100000
    buffer_size = 100000
    num_trains = 5
    path = "./net_7/"
    save_step = 1000

    for i in range(5):
        env = gym.make('Pendulum-v1')

        seed = i
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
        actor_H = 3
        critic_H = 3
        alpha = 0.0003
        gamma = 0.99
        sigma = 0.1
        tau = 0.005
        sigma_target = 0.2
        c = 0.5
        d = 2
        T_expl = 10000
        action_max = env.action_space.high[0]
        action_min = env.action_space.low[0]
        agent = TD3Agent3(n, m, dim, actor_H, critic_H, action_max, action_min, alpha, gamma, sigma, tau, sigma_target, c, d)
        rb = ReplayBuffer(buffer_size, seed)

        state = env.reset()
        done = False

        for j in tqdm(range(step_size)):
            if j < T_expl:
                action = env.action_space.sample()
            else:
                action = agent.select_exploratory_action(state)
            next_state, reward, done, _ = env.step(action)
            rb.add(state, action, next_state, reward, done)
            state = next_state

            if done:
                state = env.reset()

            if rb.q_size() >= batch_size:
                train_datas = rb.sample(batch_size)
                train_states, train_actions, train_next_states, train_rewards, train_dones = train_datas[0], train_datas[1], train_datas[2], train_datas[3], train_datas[4]
                agent.train(train_states, train_actions, train_next_states, train_rewards, train_dones, j)

            if j % save_step == save_step - 1:
                save_path = path + str(i + 1) + "_" + str(j + 1) + "_" + str(seed)
                agent.save_model(save_path)
