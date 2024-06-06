import gym
import matplotlib.pyplot as plt
from ql_agent import Qlagent


def evaluate_action(agent : Qlagent, seed : int):
    reward_list = []
    for i in range(10):
        seed = seed + i
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        observation = env.reset()
        state = agent.digitize_state(observation, agent.state_num)
        done = False
        Reward = 0

        while not done:
            action = agent.select_action(state)
            observation, reward, done, info = env.step([agent.undigitize_action(action)])
            next_state = agent.digitize_state(observation, agent.state_num)
            Reward += reward
            state = next_state
        reward_list.append(Reward)

    return reward_list


if __name__ == "__main__":
    path = "./table_with_rb_extra/8/"
    env = gym.make('Pendulum-v0')
    save_step = 25000
    step_list = [i + 1 for i in range(save_step * 10) if (i + 1) % save_step == 0]
    reward_lst=[[] for _ in range(10)]
    agent = Qlagent(10, 9, 0.0003, 0.05, 0.99)

    for i in range(5):
        for step in step_list:
            seed = i

            load_path = path + str(i+1) + "_" + str(step) + "_" + str(seed) + ".txt"

            agent.load_models(load_path)
            reward_list = evaluate_action(agent, seed)
            reward_lst[int(step / save_step - 1)].extend(reward_list)

    points = reward_lst
    fig, ax = plt.subplots()
    bp = ax.boxplot(points)
    ax.set_xticklabels([str(save_step * (i + 1)) for i in range(10)])
    plt.xlabel("step")
    plt.ylabel("total reward")
    plt.title("evaluation result of Q-learning with replay buffer(ε=10^-8)")
    plt.show()
