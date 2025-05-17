import numpy as np

# inputs
temperature = 20
humidity = 60

x = np.array([temperature, humidity])

weights = np.array([0.4, 0.6])
bias = -20
# noise eklemesi gibi çıkıntılık yapıyor yani

output = np.dot(x, weights) + bias

print(f"Nöronun ham çıktısı: {output}")

# sigmoid fonksiyonu ile çıktıyı 0-1 aralığına sıkıştırıyoruz
def sigmoid(x):
    return 1 / (1 + np.exp(-2))

activated_output = sigmoid(output)
print(f"Nöronun aktivasyon çıktısı: {activated_output}")

#-------------------------------------------------

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# import scaler  # Removed as it is not used or not found
# ages
x = np.array([5, 6, 7, 8, 9, 10], dtype=float) # tensorflow için float64 gerekli

# heights
y= np.array([110, 116, 123, 130, 136, 142], dtype=float)

# katmanları deneme yanılma ile buluyoruz

x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_scaled = x_scaler.fit_transform(x.reshape(-1, 1))
y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, activation='relu', input_shape=[1]),
    tf.keras.layers.Dense(units=1)

])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_scaled, y_scaled, epochs=1000, verbose=0)
               # verbose=0: eğitim sırasında çıktı vermesin diye

test_age = ([[7.5]]) # scale etmen gerekiyor
test_age_scaled = x_scaler.transform(test_age)


predicted_height_scaled = model.predict([test_age_scaled])
predicted_height = y_scaler.inverse_transform(predicted_height_scaled)

print(f"Yaşı {test_age} olan çocuğun tahmini boyu: {predicted_height[0][0]} cm")

