# Charlotte Weather Prediction using Multivariate LSTM

A deep learning project that uses historical weather data from Charlotte, NC (NOAA) to forecast next-day average temperatures. Built with Python, TensorFlow/Keras, Pandas, and NumPy.

---

## Background & Concepts

When dealing with time-series data like daily temperatures, standard feedforward neural networks fall short because they treat every input independently. Weather, however, depends heavily on temporal context—what happened over the last 14 days matters when predicting tomorrow. Thus, we use the following:

### 1. Recurrent Neural Networks (RNNs)
Traditional neural networks process data in a single forward pass. An **RNN** introduces a loop mechanism that carries information from previous time steps forward. This gives the network a form of "memory," making it suitable for sequential data like time series or text.

### 2. Long Short-Term Memory (LSTM)
Standard RNNs struggle with longer sequences because gradients tend to vanish or explode during backpropagation over time. **LSTM** networks solve this by introducing a complex cell architecture with three control gates:
* **Forget Gate:** Decides what information from previous days to discard.
* **Input Gate:** Decides what new information from the current day to store in memory.
* **Output Gate:** Decides what processed memory to output for the final prediction.

This gating mechanism allows LSTMs to retain critical long-term trends (like multi-week seasonal transitions) alongside short-term fluctuations.

---

## How This Model Works

Rather than relying on temperature alone, this project uses a **Multivariate LSTM** setup that evaluates **6 distinct inputs** over a **14-day lookback window**:

1. **`TAVG`**: Daily average temperature (°F)
2. **`PRCP`**: Daily precipitation (inches)
3. **`TMAX` & `TMIN`**: Daily high and low temperatures
4. **`Sin_Day` & `Cos_Day`**: Sine and cosine transformations of the day of the year

### Sine/Cosine Encoding
Standard day numbers (1 to 365) make December 31st (365) and January 1st (1) look numerically distant to a neural network, even though they sit right next to each other in real life. Mapping the day of the year onto a 2D unit circle via sine and cosine coordinates ensures the model understands seasonal continuity.

---

## Project Structure

```text
charlotte_weather_lstm/
│── charlotte-weather.csv                       # NOAA historical weather dataset
│── univariate_weather_lstm.py                  # Full training pipeline script
│── charlotte_weather_plot.png                  # Output forecast plot
│── charlotte_multivariate_weather_plot.png     # Multivariate forecast plot without rain overlay
│── charlotte_temp_and_prcp_plot.png            # Output dual-axis forecast plot with rain overlay
└── .gitignore                                  # Environment and cache ignores
```
## Results & Performance

* **Model Architecture:** 2-layer stacked LSTM (64 units → 32 units) with Dropout (0.2)
* **Forecast Accuracy:** ~91% (~4.5°F Mean Absolute Error on test data)
* **Tolerance Metrics:**
  * ~50% of predictions fall within **±3°F** of actual temperatures
  * ~75% of predictions fall within **±5°F** of actual temperatures

---

## Setup & Running

1. **Clone or download this repo** into a local folder.
2. **Create and activate a Python virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
3. Install dependencies
    ```bash
    pip install pandas numpy matplotlib scikit-learn tensorflow
    ```
4. Run the model 
    ```bash
    python multivariate_lstm.py
    ```