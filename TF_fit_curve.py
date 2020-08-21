import tensorflow as tf
model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model.fit(listex,listey, epochs=100, verbose=False