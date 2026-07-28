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
print("\nLoading Charlotte weather data...")
df = pd.read_csv('charlotte-weather.csv')

df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop = True)

df['TAVG'] = (df['TMAX'] + df['TMIN']) / 2.0

df['PRCP'] = df['PRCP'].fillna(0.0)

# Calendar cyclical feature using sin/cos of day of year
# Transform 1D day number onto a 2D circle using Sine and Cosine (so Dec 31st and Jan 1st considered "next to each other")
df['DayOfYear'] = df['DATE'].dt.dayofyear
df['Sin_Day'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
df['Cos_Day'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)

FEATURE_COLS = ['TAVG', 'PRCP', 'TMAX', 'TMIN', 'Sin_Day', 'Cos_Day']
NUM_FEATURES = len(FEATURE_COLS)

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

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

dates_test = df['DATE'].iloc[WINDOW_SIZE + train_size:].reset_index(drop = True)
print(f"Training shape X: {X_train.shape} | Testing shape X: {X_test.shape}")

# Build multivariate lstm model
model = Sequential([
    LSTM(64, return_sequences = True, input_shape = (WINDOW_SIZE, NUM_FEATURES)),
    Dropout(0.2),
    LSTM(32, return_sequences = False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae'])
model.summary()

# ** Training the model
print("\nTraining the Multivariate LSTM...")
history = model.fit(X_train, y_train, epochs = 25, batch_size = 32, validation_data = (X_test, y_test), verbose = 1)

# Evaluating and unscaling predictions
print("\nGenerating predictions...")
predictions_scaled = model.predict(X_test)

predictions = scaler_y.inverse_transform(predictions_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

mae = np.mean(np.abs(predictions - y_test_actual))
rmse = np.sqrt(np.mean((predictions - y_test_actual)**2))

mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100
accuracy_pct = 100 - mape

print("\n" + "=" * 40)
print(f"Mean Absolute Error (MAE):    {mae:.2f}°F")
print(f"Mean Absolute % Error (MAPE): {mape:.2f}%")
print(f"Model Accuracy Percentage:   {accuracy_pct:.2f}%")
print("=" * 40)

# ** Plot
plt.figure(figsize=(12, 6))
plt.plot(dates_test, y_test_actual, label='Actual Temperature (°F)', color='#1f77b4', linewidth=1.5)
plt.plot(dates_test, predictions, label='Multivariate LSTM Forecast (°F)', color='#ff7f0e', linestyle='--', linewidth=1.5)

plt.title('Charlotte, NC - Multivariate Temperature Prediction (LSTM)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Temperature (°F)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plt.savefig('charlotte_multivariate_weather_plot.png', dpi=300)
print("Plot saved as 'charlotte_multivariate_weather_plot.png'")
plt.show()

# Plot dual-axis temp and rain overlay
prcp_test = df['PRCP'].iloc[WINDOW_SIZE + train_size:].reset_index(drop = True)

fig, ax1 = plt.subplots(figsize = (12,6))
# Temperature plot
ax1.set_xlabel('Date', fontsize = 12)
ax1.set_ylabel('Average Temperature (°F)', color = '#1f77b4', fontsize = 12)
line1 = ax1.plot(dates_test, y_test_actual, label = 'Actual Temp (°F)', color = '#1f77b4', linewidth = 1.5)
line2 = ax1.plot(dates_test, predictions, label = 'Multivariate LSTM Forecast (°F)', color = "#ff7f0e", linestyle = '--', linewidth = 1.5)
ax1.tick_params(axis = 'y', labelcolor = '#1f77b4')
ax1.grid(True, linestyle = ':', alpha = 0.6)

# Bar chart overlay
ax2 = ax1.twinx()
ax2.set_ylabel('Precipitation (Inches)', color = 'gray', fontsize = 12)
line3 = ax2.bar(dates_test, prcp_test, label = 'Precipitation (in)', color = 'gray', alpha = 0.3, width = 1.5)
ax2.tick_params(axis = 'y', labelcolor = 'gray')

# Combine legends
lines = line1+line2 + [line3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc = 'lower right', fontsize = 11)

plt.title('Charlotte, NC - Multivariate Forecast with Precipitation Overlay', fontsize = 14, fontweight = 'bold')
plt.tight_layout()

plt.savefig('charlotte_temp_and_prcp_plot.png', dpi = 300)
print("Plot saved as 'charlotte_temp_and_prcp_plot.png'")
plt.show()
