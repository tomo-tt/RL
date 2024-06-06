from tqdm import tqdm
import gym
import numpy as np

from ql_agent import Qlagent
from replaybuffer import ReplayBuffer


if __name__ == "__main__":
    batch_size = 256
    step_size = 250000
    buffer_size = 250000
    num_trains = 5
    path = "./table_with_rb_extra"
    save_step = 25000

    index_list = [8]
    for index in index_list:
        for i in range(num_trains):
            env = gym.make('Pendulum-v0')

            seed = i
            np.random.seed(seed=seed)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)

            state_num = 10
            action_num = 9
            alpha = 0.0003
            epsilon = 10 ** (index * (-1))
            gamma = 0.99
            agent = Qlagent(state_num, action_num, alpha, epsilon, gamma)
            rb = ReplayBuffer(buffer_size, seed)

            observation = env.reset()
            state = agent.digitize_state(observation, state_num)
            done = False

            for j in tqdm(range(step_size)):
                action = agent.select_exploratory_action(state)
                observation, reward, done, info = env.step([agent.undigitize_action(action)])
                next_state = agent.digitize_state(observation, state_num)
                rb.add(state, action, next_state, reward, done)
                state = next_state

                if done:
                    next_observation = env.reset()
                    state = agent.digitize_state(next_observation, state_num)

                if rb.q_size() >= batch_size:
                    train_datas = rb.sample(batch_size)
                    for k in range(batch_size):
                        train_state, train_action, train_next_state, train_reward, train_done = train_datas[0][k], train_datas[1][k], train_datas[2][k], train_datas[3][k], train_datas[4][k]
                        train_next_action = agent.select_action(train_next_state)
                        agent.train(train_state, train_action, train_next_state, train_next_action, train_reward)

                if j % save_step == save_step - 1:
                    save_path = path + "/" + str(index) + "/" + str(i+1) + "_" + str(j+1) + "_" + str(seed) + ".txt"
                    agent.save_model(save_path)
