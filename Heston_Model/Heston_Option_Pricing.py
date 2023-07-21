import pickle
import numpy as np
np.random.seed(42)
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, save_model, load_model
from google.colab import drive
drive.mount('/content/drive/')


with open('/content/drive/MyDrive/Colab Notebooks/df_1000000.pickle', 'rb') as f:
    df = pickle.load(f)

df.dropna(inplace=True)

df.reset_index(drop=True, inplace=True)

df_filtered = df[df['option_price'] < 1]
df_filtered = df_filtered[df_filtered['option_price'] >= 0]

df_filtered.head(10)

# Check if GPU is available

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

# Create a TensorFlow session and set it to use the GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# Split the dataset into train and test sets

train_df, test_df = train_test_split(df_filtered, test_size=0.1, random_state=55)
train_df, eval_df = train_test_split(train_df, test_size=1/9, random_state=55)

#################################################

def calc_metrics(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'mse': mse, 'mae': mae, 'r2': r2}

model = Sequential()
model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform(), input_shape=(8,)))
model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform()))
model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform()))
model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform()))
model.add(Dense(1))
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Train the model on the GPU

batch_size = 1024
epochs = 100
with tf.device('/GPU:0'):
    history = model.fit(
        x = train_df[['m', 'tau', 'r', 'rho', 'kappa', 'theta', 'sigma', 'v0']],
        y = train_df['option_price'],
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )

# Evaluate the trained model on the test sets on the GPU

with tf.device('/GPU:0'):
    wide_test_loss = model.evaluate(
        test_df[['m', 'tau', 'r', 'rho', 'kappa', 'theta', 'sigma', 'v0']],
        test_df['option_price'],
        verbose=0
    )

wide_test_predictions = model.predict(test_df[['m', 'tau', 'r', 'rho', 'kappa', 'theta', 'sigma', 'v0']])


# Reshape wide_test_predictions to match the shape of test_df['option_price']
wide_test_predictions = wide_test_predictions.flatten()

# Calculate performance metrics for wide test set
res = calc_metrics(test_df['option_price'], wide_test_predictions)
print(res)

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt
import random


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(test_df['m'],  test_df['tau'])
zs = np.array(wide_test_predictions)
print(zs.shape)
Z = zs.reshape(3897 , 1)
print(Z)
print(Z.shape)
ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()