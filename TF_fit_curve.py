import matplotlib.pyplot as plt  # package to plot graphs
import numpy as np  # arrays package
import pandas as pd  # database package
from tensorflow.keras import Sequential  # I will use sequential neural networks
from tensorflow.keras import optimizers  # Optimizers improve the network
from tensorflow.keras.layers import Dense  # Dense neural network layers

noisy_df = pd.read_csv('noisy_function.csv')  # read training data
print(noisy_df)
df = pd.read_csv('random_function.csv')   # read test data and true data
print(df)

# Create arrays
x1 = np.array(noisy_df['x'])
y1 = np.array(noisy_df['y'])

# Calculate maximum and minimum for normalisation
max_y = np.max(y1)
min_y = np.min(y1)
print(max_y, min_y)
for z in range(len(y1)):
    y1[z] = (y1[z] - min_y) / (max_y - min_y)  # Normalise y1

# Same for test data
x2 = np.array(df['x'])
y2 = np.array(df['y'])
for z in range(len(y2)):
    y2[z] = (y2[z] - min_y) / (max_y - min_y)


# Create model
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

# Fit model
model.fit(x1, y1, epochs=100000, verbose=True)

# Predict values
y3 = model.predict(x2)

plt.scatter(x1, y1)  # Plot noisy data
plt.plot(x2, y2, 'r')  # Plot true function
plt.plot(x2, y3, 'g')  # Plot neural network predictions
plt.show()
