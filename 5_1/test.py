import random
import numpy as np

def undigitize_action(action):
        return (action / 2.0) - 2.0

sin_theta = 1.0
print(np.digitize(sin_theta, bins=np.linspace(-1, 1, 10 + 1)[1:-1]))

state_num = 10
action_num = 9
table = np.random.normal(0, 1, size=(state_num**3, action_num)) * 10e-8
print(table.shape)
array = np.random.randn(1000, 9)
print(array.shape)

# for i in range(9):
#     print(undigitize_action(i))

print(list(random.randint(0, 10 - 1) for _ in range(5)))

save_step = 50000
print([str(save_step * (i + 1)) for i in range(10)])
print(["50000", "100000", "150000", "200000", "250000", "300000", "350000", "400000", "450000", "500000"])

def bins(min, max, num):
        return np.linspace(min, max, num + 1)[1:-1]
print(bins(-1,1,10))
print(np.digitize(sin_theta, bins=bins(-1.0, 1.0, state_num)))

# print(np.digitize(-1.11, bins=bins(-8.0, 8.0, state_num)))

# index_list = [i * 2 for i in range(6)]
# print(index_list)
# print(10 ** 0)

print(10 ** (-1 * 2))


print(2/2-2)
