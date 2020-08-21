from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

noisy_df = pd.read_csv('noisy_function.csv')
print(noisy_df)
df = pd.read_csv('random_function.csv')
print(df)

np.random.seed(34)
x1 = np.array(noisy_df['x'])

y1 = np.array(noisy_df['y'])

max_y = np.max(y1)
min_y = np.min(y1)
print(max_y, min_y)
for z in range(len(y1)):
    y1[z] = (y1[z] - min_y) / (max_y - min_y)

x2 = np.array(df['x'])

y2 = np.array(df['y'])
for z in range(len(y2)):
    y2[z] = (y2[z] - min_y) / (max_y - min_y)

model = Sequential()

# Layer 1
model.add(Dense(32, activation='sigmoid', input_dim=1))
# Layer 2
model.add(Dense(64, activation='sigmoid'))
# Layer 3
model.add(Dense(64, activation='sigmoid'))
# Output Layer
model.add(Dense(1, activation='sigmoid'))

print(model.summary())
print('')

sgd = optimizers.SGD(lr=1)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(x1, y1, epochs=100000, verbose=True)

y3 = model.predict(x2)

plt.scatter(x1, y1)
plt.plot(x2, y2, 'r')
plt.plot(x2, y3, 'g')
plt.show()
