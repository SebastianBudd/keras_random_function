import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense
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

x2 = np.array(df['x'])
y2 = np.array(df['y'])

model = Sequential()

# Layer 1
model.add(Dense(units=len(x1), activation='sigmoid', input_dim=1))
# Output Layer
model.add(Dense(units=1, activation='sigmoid'))

print(model.summary())
print('')

sgd = optimizers.SGD(lr=1)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(x1, y1, epochs=100, verbose=False)

y3 = model.predict(x2)

plt.scatter(x1, y1)
plt.plot(x2, y2, 'r')
plt.plot(x2, y3, 'g')
plt.show()
