import gym
env = gym.make('BipedalWalker-v3')
state = env.reset()
for t in range(1000000):
    env.render()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    if done:
        break
env.close()
