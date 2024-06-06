import gym
import numpy as np

from ql_agent import Qlagent



if __name__ == "__main__":
    batch_size = 256
    step_size = 250000
    buffer_size = 250000
    num_trains = 1
    path = "./table_with_rb_extra/0/"

    for i in range(3):
        seed=i+25
        np.random.seed(seed=seed)
        env = gym.make('Pendulum-v0')
        agent = Qlagent(10, 9, 0.0003, 0.05, 0.99)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        # load_path = path + "_" + str(i+1) + "_" + "250000" + "_" + str(seed) + ".txt"
        load_path = path + str(i+1) + "_" + "250000" + "_" + str(seed-25) + ".txt"
        agent.load_models(load_path)

        observation = env.reset()
        state = agent.digitize_state(observation, agent.state_num)
        done = False
        while not done:
            env.render()
            action = agent.select_action(state)
            observation, reward, done, info = env.step([agent.undigitize_action(action)])
            next_state = agent.digitize_state(observation, agent.state_num)
            state = next_state
