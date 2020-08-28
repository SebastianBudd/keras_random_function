#  We will use this program to generate a random function and then to create a set of noisy data points using this
#  function. We will then plot the data and export it as a csv file

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from random import seed
from random import random

print('Packages successfully imported!')

#  Create list of integers from 0 to 1000
integer_list = []
count = 0
while count != 1001:
    integer_list.append(count)
    count += 1

#  Create 200 random numbers between 0 and 100
random_list = []
seed(2)
for x in range(200):
    random_list.append(random() * 10)

#  Create random ...
c_constant = (random() * 2) - 1
c_linear = ((random() * 2) - 1) * 200
c_quadratic = ((random() * 2) - 1) * 20
c_cubic = ((random() * 2) - 1)
c_sin = ((random() * 2) - 1) * 500
c_cos = ((random() * 2) - 1) * 500


def f(x):
    """Randomly generated function"""
    z = c_constant + c_linear * x + c_quadratic * (x ** 2) + c_cubic * (x ** 3) + c_sin * math.sin(x) \
        + c_cos * math.cos(x * 3)
    return z

X=[]
Y=[]
for x1 in integer_list:
    X.append(x1/100)
    Y.append(f(x1/100))

print("y = {} + {} x + {} x^2 + {} x^3 + {} sin(x) + {} cos(x * 3)".format(c_constant, c_linear, c_quadratic, c_cubic,
                                                                           c_sin, c_cos))
y_list = []
for x in range(len(random_list)):
    y_list.append(f(random_list[x]))

#  add noise

noise1 = np.random.normal(0, 0.3, 200)
noise2 = np.random.normal(0, 30, 200)
x_noisy = []
for x in range(len(random_list)):
    x_noisy.append(random_list[x] + noise1[x])

y_noisy = []
for y in range(len(y_list)):
    y_noisy.append(y_list[y] + noise2[y])

plt.scatter(x_noisy, y_noisy)
plt.plot(X, Y, 'r')

d = {"x": X, "y": Y}
df = pd.DataFrame(d)
df.to_csv(r'random_function.csv', index=False)

d = {"x": x_noisy, "y": y_noisy}
df_noisy = pd.DataFrame(d)
df_noisy.to_csv(r'noisy_function.csv', index=False)

plt.show()

