import random


class ReplayBuffer:
    def __init__(self,
                 buffer_size,
                 seed):
        self.buffer_size = buffer_size
        self.rb = list()
        random.seed(seed)

    def q_size(self):
        return len(self.rb)

    def add(self, state, action, next_state, reward, done):
        if self.q_size() > self.buffer_size:
            self.rb.pop(0)
        self.rb.append([state, action, next_state, reward, done])

    def sample(self, batch_size):
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []

        random_numbers = list(random.randint(0, self.q_size() - 1) for _ in range(batch_size))

        for random_number in random_numbers:
            states.append(self.rb[random_number][0])
            actions.append(self.rb[random_number][1])
            next_states.append(self.rb[random_number][2])
            rewards.append(self.rb[random_number][3])
            dones.append(self.rb[random_number][4])

        return [states, actions, next_states, rewards, dones]
