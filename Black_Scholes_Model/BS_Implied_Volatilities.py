#################  Part 1 –The resulting input is then given by {V/K, S0/K,r, τ}  #################

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

def calc_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'mse': mse, 'mae': mae, 'r2': r2}

#########################

# Train the model on the GPU

model = Sequential()
model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform(), input_shape=(4,)))
model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform()))
model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform()))
model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform()))
model.add(Dense(1))
model.compile(optimizer=Adam(), loss='mean_squared_error')
batch_size = 1024
epochs = 2
with tf.device('/GPU:0'):
    history = model.fit(
        train_df[['Sk', 'r', 'T', 'Price']],
        train_df['sigma'],
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )

#########################

# Evaluate the trained model on the test sets on the GPU

with tf.device('/GPU:0'):
    wide_test_loss = model.evaluate(
        test_df[['Sk', 'r', 'T', 'Price']],
        test_df['sigma'],
        verbose=0
    )

wide_test_predictions = model.predict(test_df[['Sk', 'r', 'T', 'Price']])
print(len(test_df['sigma']))
print(len(wide_test_predictions))
res= calc_metrics(test_df['sigma'], wide_test_predictions)
print(res)

#################  Part 2 –  Vhat = V − MAX(S − Ke^{−rτ}, 0)  #################

"""
 Where Vhat is the option time value.
 The proposed approach to overcome approximation issues is to reduce the gradient’s steepness by furthermore working under a log-transformation of the option value. The resulting input is then given by {log (Vhat/K), S0/K,r, τ}.
 The adapted gradient approach increases the prediction accuracy significantly.
"""

warnings.filterwarnings('ignore')

Vhat=np.log(option_prices-np.maximum(Sk -  np.exp(-r * T), 0))

df2= pd.DataFrame({'Sk': Sk, 'r': r, 'T': T, 'sigma': sigma, 'Price': Vhat})
df2.replace([-np.inf, -np.inf], np.nan, inplace=True)
df2.dropna(inplace=True)

# Split the dataset into train and testnan sets
train_df2, test_df2 = train_test_split(df2, test_size=0.1, random_state=42)

#########################

# Train the model on the GPU
model = Sequential()
model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform(), input_shape=(4,)))
model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform()))
model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform()))
model.add(Dense(400, activation='relu', kernel_initializer=GlorotUniform()))
model.add(Dense(1))
model.compile(optimizer=Adam(), loss='mean_squared_error')
batch_size = 1024
epochs = 2
with tf.device('/GPU:0'):
    history = model.fit(
        train_df2[['Sk', 'r', 'T', 'Price']],
        train_df2['sigma'],
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )

#########################

# Evaluate the trained model on the test sets on the GPU

with tf.device('/GPU:0'):
    wide_test_loss = model.evaluate(
        test_df2[['Sk', 'r', 'T', 'Price']],
        test_df2['sigma'],
        verbose=0
    )

wide_test_predictions = model.predict(test_df2[['Sk', 'r', 'T', 'Price']])

res= calc_metrics(test_df2['sigma'], wide_test_predictions)
print(res)