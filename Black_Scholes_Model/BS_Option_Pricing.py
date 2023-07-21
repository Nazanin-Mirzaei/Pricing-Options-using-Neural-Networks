# Tools
import numpy as np
import pandas as pd
import warnings
import pickle
from keras import backend as K
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as geek
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

##########################

# Check if GPU is available

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

# Create a TensorFlow session and set it to use the GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

#########################

# Load Data Generated 

with open('df_10000.pickle', 'rb') as f:
  df = pickle.load(f)

# Split the dataset into train and test sets
# This article takes 90% of the data as the training data and 10% as test data
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Scale the output option prices
scaler = MinMaxScaler()
train_df['Price'] = scaler.fit_transform(train_df['Price'].values.reshape(-1, 1))
test_df['Price'] = scaler.transform(test_df['Price'].values.reshape(-1, 1))

#########################

# Model definition

model = Sequential()
model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform(), input_shape=(4,)))
model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform()))
model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform()))
model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform()))
model.add(Dense(1))
model.compile(optimizer=Adam(), loss='mean_squared_error')
batch_size = 1024
epochs = 1000
with tf.device('/GPU:0'):
    history = model.fit(
        train_df[['Sk', 'r', 'T', 'sigma']],
        train_df['Price'],
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )

#########################

def calc_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'mse': mse, 'mae': mae, 'r2': r2}

#########################

# Evaluate the trained model on the test sets on the GPU

with tf.device('/GPU:0'):
    wide_test_loss = model.evaluate(
        test_df[['Sk', 'r', 'T', 'sigma']],
        test_df['Price'],
        verbose=0
    )

wide_test_predictions = model.predict(test_df[['Sk', 'r', 'T', 'sigma']])
wide_test_predictions = scaler.inverse_transform(wide_test_predictions)  # Inverse scaling

# Reshape wide_test_predictions to match the shape of test_df['Price']
wide_test_predictions = wide_test_predictions.flatten()

# Calculate performance metrics for wide test set
res= calc_metrics(test_df['Price'], wide_test_predictions)
print(res)

######################### PLOT

# Train with different lrs
batch_size = 1024
epochs = 2
average_losses = []
learning_rate= []
for i in range(4):
  model = Sequential()
  model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform(), input_shape=(4,)))
  model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform()))
  model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform()))
  model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform()))
  model.add(Dense(1))
  model.compile(optimizer=Adam(), loss='mean_squared_error')
  K.set_value(model.optimizer.learning_rate, 0.01+i*0.02)
  learning_rate.append(0.01+i*0.02)
  with tf.device('/GPU:0'):
      history = model.fit(
          train_df[['Sk', 'r', 'T', 'sigma']],
          train_df['Price'],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1
      )
  average_losses.append(
      np.average(history.history['loss'])
  )

import matplotlib.pyplot as plt
import numpy as np
plt.plot(learning_rate, average_losses, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Average Loss')
plt.title('Learning Rate vs. Average Loss')
plt.show()

#########################

# Train with differnt dataset size
batch_size = 1024
epochs = 2
R2s = []
mses = []
x=[]
for i in range(3):
  model = Sequential()
  model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform(), input_shape=(4,)))
  model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform()))
  model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform()))
  model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform()))
  model.add(Dense(1))
  model.compile(optimizer=Adam(), loss='mean_squared_error')
  K.set_value(model.optimizer.learning_rate, 0.0001)
  #print("Learning rate before second fit:", model.optimizer.learning_rate.numpy())
  with tf.device('/GPU:0'):
      history = model.fit(
          train_df[['Sk', 'r', 'T', 'sigma']][:int((i+1)/8*len(train_df))],
          train_df['Price'][:int((i+1)/8*len(train_df))],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1
      )
  train_predicteds = model.predict(train_df[['Sk', 'r', 'T', 'sigma']][:int((i+1)/8*len(train_df))])
  # print(train_predicteds[:100])
  # print(train_df['Price'][:100])
  _metrics = calc_metrics(train_df['Price'][:int((i+1)/8*len(train_df))], train_predicteds)
  R2s.append(_metrics['r2']*100)
  mses.append(np.log(_metrics['mse']))
  #print("_metrics: ", _metrics)
  x.append(i)
fig, ax1 = plt.subplots()

ax1.plot(x, mses, lw=2, color="blue")
ax1.set_ylabel(r"Log(MSE)", fontsize=10)

ax2 = ax1.twinx()
ax2.plot(x, R2s, lw=2, color="red")
ax2.set_ylabel(r"R2(%)", fontsize=10 )

