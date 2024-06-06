from agent import Agent
from TD3 import Actor, Critic, TD3Agent
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3AgentWithout1(Agent):
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
        self.critic = Critic(n, m, dim, critic_H).to(device)
        self.critic2 = Critic(n, m, dim, critic_H).to(device)

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
        critic_path = path + "critic.pth"
        critic2_path = path + "critic2.pth"

        torch.save(self.actor.to('cpu').state_dict(), actor_path)
        torch.save(self.critic.to('cpu').state_dict(), critic_path)
        torch.save(self.critic2.to('cpu').state_dict(), critic2_path)

        self.actor.to(device)
        self.critic.to(device)
        self.critic2.to(device)

    def load_models(self, path):
        actor_path = path + "actor.pth"
        critic_path = path + "critic.pth"
        critic2_path = path + "critic2.pth"

        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        self.critic2.load_state_dict(torch.load(critic2_path))

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def select_exploratory_action(self, state):
        state = torch.FloatTensor(state).to(device)
        pi = self.actor(state).cpu().data.numpy().flatten()
        noise = np.random.normal(0, ((self.action_max - self.action_min) * self.sigma / 2) ** 2, size=self.action_num)
        beta = pi + noise
        return np.clip(beta, self.action_min, self.action_max)

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
            next_actions = self.actor(next_states)
            next_actions = self.add_noise_to_target_actor(next_actions)
            next_q_values_target = torch.min(self.critic(next_states, next_actions),
                                                self.critic2(next_states, next_actions))
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

        if current_step % self.d == 0:
            actor_loss = - self.critic(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()


class TD3AgentWithout2(TD3Agent):
   def train(self, states, actions, next_states, rewards, dones, current_step):
        states = torch.FloatTensor(np.array(states)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards).reshape(-1, 1)).to(device)
        dones = torch.FloatTensor(np.array(dones).reshape(-1, 1)).to(device)

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
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

        if current_step % self.d == 0:
            actor_loss = - self.critic(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update_target_networks()
