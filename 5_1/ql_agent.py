from agent import Agent
from tqdm import tqdm
import gym
import numpy as np

class Qlagent(Agent):
    def __init__(self,
                 state_num,
                 action_num,
                 alpha,
                 epsilon,
                 gamma,
                 ):
        self.state_num=state_num
        self.action_num=action_num
        self.alpha=alpha
        self.epsilon=epsilon
        self.gamma=gamma
        self.table=self.init_table()

    def init_table(self):
        return np.random.normal(0, 1, size=(self.state_num**3, self.action_num)) * 10e-8

    def save_model(self, path):
        np.savetxt(path, self.table)

    def load_models(self, path):
        self.table = np.loadtxt(path)

    def select_action(self, state):
        return np.array([np.argmax(self.table[state])])

    def select_exploratory_action(self, state):
        if self.epsilon <= np.random.uniform(0, 1):
            return np.array([np.argmax(self.table[state])])
        return np.array(np.random.randint(0, self.action_num))

    def train(self, state, action, next_state, next_action, reward):
        alpha = self.alpha
        gamma = self.gamma
        self.table[state, action] = (1 - alpha) * self.table[state, action] + \
        alpha * (reward + gamma * self.table[next_state, next_action])

    def bins(self, min, max, num):
        return np.linspace(min, max, num + 1)[1:-1]

    def digitize_state(self, observation, state_num):
        sin_theta, cos_theta, theta_dot = observation
        digitized = [np.digitize(sin_theta, bins=self.bins(-1.0, 1.0, state_num)),
        np.digitize(cos_theta, bins = self.bins(-1.0, 1.0, state_num)),
        np.digitize(theta_dot, bins= self.bins(-8.0, 8.0, state_num))]

        return sum([x* (state_num**i) for i, x in enumerate(digitized)])

    def undigitize_action(self, action):
        return float((action / 2.0) - 2.0)


if __name__ == "__main__":
    step_size = 500000
    num_trains = 5
    path = "./table/"
    save_step = 50000


    for i in range(num_trains):
        env = gym.make('Pendulum-v0')

        seed = i
        np.random.seed(seed = seed)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        state_num = 10
        action_num = 9
        alpha = 0.0003
        epsilon = 0.05
        gamma = 0.99
        agent = Qlagent(state_num, action_num, alpha, epsilon, gamma)

        observation = env.reset()
        state = agent.digitize_state(observation, state_num)
        done = False

        for j in tqdm(range(step_size)):
            action = agent.select_exploratory_action(state)
            observation, reward, done, info = env.step([agent.undigitize_action(action)])
            next_state = agent.digitize_state(observation, state_num)

            next_action = agent.select_action(next_state)
            agent.train(state, action, next_state, next_action, reward)
            state = next_state

            if done:
                next_observation = env.reset()
                state = agent.digitize_state(next_observation, state_num)

            # if j % save_step == save_step - 1:
            #     save_path = path + "_" + str(i+1) + "_" + str(j+1) + "_" + str(seed) + ".txt"
            #     agent.save_model(save_path)
