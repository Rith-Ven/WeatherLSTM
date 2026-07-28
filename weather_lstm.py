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

# **Load and clean the csv data
print("Loading Charlotte weather data...")
df = pd.read_csv("charlotte-weather.csv")

df['DATE'] = pd.to_datetime(df['DATE']) # Convert string date into pd datetime objects
df = df.sort_values('DATE').reset_index(drop=True) # Order rows chronologically and reindex 

df['TAVG'] = (df['TMAX'] + df['TMIN']) / 2.0  # Populate the average daily temp col
data = df[['DATE', 'TAVG']].copy() # Creates a new dataframe with JUST data and avg daily temp

print(f"Data successfully loaded. Date range: {data['DATE'].min().strftime('%Y-%m-%d')} to {data['DATE'].max().strftime('%Y-%m-%d')}")

# **Data scaling
scaler = MinMaxScaler(feature_range=(0,1)) # Standardizer
scaled_temp = scaler.fit_transform(data[['TAVG']]) # Finds global min and max and applies scaling x_scaled = (x-x_min) / (x_max - x_min)

# **Creating a sequence 
WINDOW_SIZE = 14 # Model inspects 14 consecutive past days (t-14) to predict temp for day t

# Iterates through scaled dataset to build pairs of input sequences (X) and target outputs (y)
def create_sequences(dataset, window_size):
    X, y = [], []
    for i in range(len(dataset) - window_size):
        X.append(dataset[i : i + window_size]) # Gets 14 consecutive vals into X
        y.append(dataset[i + window_size]) # Gets 15th val as label
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_temp, WINDOW_SIZE)

# 80/20 train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
dates_test = data['DATE'].iloc[WINDOW_SIZE + train_size:].reset_index(drop = True) #Gets calendar dates for test predictions for accurate time-series plotting

# **Building LSTM Network Architecture
# Sequential lets data flow through stacked layers
model = Sequential([ 
    # Has 64 memory units, outputs a hidden state sequence across 14 time steps, lets next layer analyze temporal patterns
    LSTM(64,return_sequences = True, input_shape = (WINDOW_SIZE, 1)), 
    # Zeroes out 20% of network activations during training to mitigate overfitting
    Dropout(0.2), 
    # Second layer with 32 memory units, summaries the output across time steps
    LSTM(32, return_sequences = False), 
    # Drops 20% when exiting second LSTM
    Dropout(0.2), 
    # Fully connected linear project mapping the 32D context vector to 1 continuous numerical prediction
    Dense(1)])

# Configures Adaptive Moment Estimation optimizer to dynamically compute learning rates for model weights
# Defines loss function to minimize squared error terms ( 1/n Sum(y - y_hat) ^ 2)
# Tracks Mean Absolute Error during training
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae'])
model.summary()

# **Training the model
print("\nTraining the LSTM model...")
# Training loop across training pairs, completes 25 full forward backward training passes, updates weights every 32 pairs, computes validation loss
history = model.fit(X_train, y_train, epochs = 25, batch_size = 32, validation_data = (X_test, y_test), verbose = 1)

# **Generating predictions
print("\nGenerating predictions on test set...")
predictions_scaled = model.predict(X_test)
# Converts normalized outputs back into readable degrees
predictions = scaler.inverse_transform(predictions_scaled)
y_test_actual = scaler.inverse_transform(y_test)

# Calculates final performance, MAE measures average scalar pred error magnitude and RMSE penalizes larger localized pred errors
mae = np.mean(np.abs(predictions - y_test_actual))
rmse = np.sqrt(np.mean((predictions - y_test_actual) **2))

# **Visualization - renders values + predictions
plt.figure(figsize = (12, 6))
plt.plot(dates_test, y_test_actual, label = 'Actual Temperature (°F)', color = '#1f77b4', linewidth = 1.5)
plt.plot(dates_test, predictions, label = 'LSTM Forecast (°F)', color = '#ff7f0e', linestyle = '--', linewidth = 1.5)

plt.title('Charlotte, NC - Historical Temperature Prediction (LSTM)', fontsize = 14, fontweight = 'bold')
plt.xlabel('Date', fontsize = 12)
plt.ylabel('Average Temperature (°F)', fontsize = 12)
plt.legend(fontsize = 11)
plt.grid(True, linestyle = ":", alpha = 0.6)
plt.tight_layout()

plt.savefig('charlotte_weather_plot.png', dpi = 300)
print("Plot saved as charlotte_weather_plot.png")
plt.show()