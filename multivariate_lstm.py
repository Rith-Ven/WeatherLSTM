import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler  # Used to normalize dataset feature values 0-1
import tensorflow as tf                         # Open source framework for deep learning
from keras.models import Sequential             # Layers stacked linearly   
from keras.layers import LSTM, Dense, Dropout   # Recurrent layer to learn temporal sequence patterns, a fully connected layer to output predictions, a regularization layer to mitigate overfitting

# Fixes RNGs in NumPy and Tensorflow to a random seed, ensures model weight intialization stays consistent
np.random.seed(42)
tf.random.set_seed(42)

# **Load data
df = pd.read_csv('charlotte-weather.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop = True)

df['TAVG'] = (df['TMAX'] + df['TMIN']) / 2.0

df['PRCP'] = df['PRCP'].fillna(0.0)

# Calendar cyclical feature using sin/cos of day of year
df['DayOfYear'] = df['DATE'].dt.dayofyear
df['Sin_Day'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
df['Cos_Day'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)

FEATURE_COLS = ['TAVG', 'PRCP', 'TMAX', 'TMIX', 'Sin_Day', 'Cos_Day']
NUM_FEATURES - len(FEATURE_COLS)

# **Feature and Target Scaling - need two scalers now bc X has 6 features while y only have 1
scaler_X = MinMaxScaler(feature_range = (0,1))
scaled_features = scaler_X.fit_transform(df[FEATURE_COLS])
scaler_y = MinMaxScaler(feature_range =(0,1))
scaled_target = scaler_y.fit_transform(df[['TAVG']])

WINDOW_SIZE = 14
def create_multivariate_sequences(features, target, window_size):
    X,y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i : i + window_size])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)

X,y = create_multivariate_sequences(scaled_features, scaled_target, WINDOW_SIZE)

