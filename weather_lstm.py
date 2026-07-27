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

# Load and clean the csv data
print("Loading Charlotte weather data...")
df = pd.read_csv("charlotte-weather.csv")

df['DATE'] = pd.to_datetime(df['DATE']) # Convert string date into pd datetime objects
df = df.sort_values('DATE').reset_index(drop=True) # Order rows chronologically and reindex 

df['TAVG'] = (df['TMAX'] + df['TMIN']) / 2.0  # Populate the average daily temp col
data = df[['DATE', 'TAVG']].copy() # Creates a new dataframe with JUST data and avg daily temp

print(f"Data successfully loaded. Date range: {data['DATE'].min().strftime('%Y-%m-%d')} to {data['DATE'].max().strftime('%Y-%m-%d')}")

# Data scaling
scaler = MinMaxScaler(feature_range=(0,1)) # Standardizer
scaled_temp = scaler.fit_transform(data[['TAVG']]) # Finds global min and max and applies scaling x_scaled = (x-x_min) / (x_max - x_min)

