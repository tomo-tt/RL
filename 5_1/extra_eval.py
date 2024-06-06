import gym
import matplotlib.pyplot as plt
from ql_agent import Qlagent
from eval import evaluate_action


if __name__ == "__main__":
    path = "./table_with_rb_extra/0/"
    # path = "./extra_table/5/"
    env = gym.make('Pendulum-v0')
    save_step = 25000
    step_list = [i + 1 for i in range(save_step * 10) if (i + 1) % save_step == 0]
    reward_lst=[[] for _ in range(10)]
    agent = Qlagent(10, 9, 0.0003, 0.05, 0.99)

    for i in range(5):
        for step in step_list:
            seed = i
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)

            state = env.reset()
            load_path = path + str(i+1) + "_" + str(step) + "_" + str(seed) + ".txt"

            agent.load_models(load_path)
            reward_list = evaluate_action(agent)
            reward_lst[int(step / save_step - 1)].extend(reward_list)

    points = reward_lst
    fig, ax = plt.subplots()
    bp = ax.boxplot(points)
    ax.set_xticklabels([str(save_step * (i + 1)) for i in range(10)])
    plt.xlabel("step")
    plt.ylabel("total reward")
    plt.title("evaluation result of Q-learning with replay buffer(ε=10^-1)")
    plt.show()
