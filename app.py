import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime

st.title("🧠 LSTM Stock Price Predictor")
st.sidebar.header("User Input Parameters")

# Sidebar inputs
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.now().date())
look_back = st.sidebar.slider("Time Step (Days Lookback)", 30, 120, 60, step=10) # 60 is a good starting point

@st.cache_data
def load_data(ticker, start, end):
    """Fetches and processes stock data."""
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            st.error(f"Could not retrieve data for ticker: {ticker}")
            return None, None, None
        
        # We'll use the 'Close' price for prediction
        data = df['Close'].values.reshape(-1, 1)
        
        # 1. Scaling / Normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # 2. Train-Test Split (e.g., 80% train, 20% test)
        training_data_len = int(np.ceil(len(scaled_data) * 0.8))
        train_data = scaled_data[0:training_data_len, :]
        test_data = scaled_data[training_data_len - look_back:, :]
        
        return train_data, test_data, scaler, df
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return None, None, None, None

def create_sequences(data, look_back):
    """Creates the (Samples, Timesteps, Features) structure."""
    X, Y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0]) # look_back sequential prices
        Y.append(data[i, 0])            # The price to predict (next day)
    return np.array(X), np.array(Y)

@st.cache_resource
def train_model(X_train, y_train):
    """Builds and trains the Stacked LSTM model."""
    st.write("---")
    st.subheader("Building and Training Model (may take a moment...)")
    
    # Reshape input data to the required LSTM format: (Samples, Timesteps, Features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # Build the LSTM model
    model = Sequential()
    
    # Layer 1: LSTM with return_sequences=True to pass output to the next LSTM layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2)) # Prevents overfitting
    
    # Layer 2: LSTM
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Final Dense Layers
    model.add(Dense(units=25))
    model.add(Dense(units=1)) # Output layer for one price prediction
    
    # Compile
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0) 
    
    st.success("✅ Model Training Complete!")
    return model

if st.sidebar.button("Run Prediction"):
    train_data, test_data, scaler, df = load_data(ticker_symbol, start_date, end_date)

    if df is not None:
        # 1. Create Train Sequences
        X_train, y_train = create_sequences(train_data, look_back)
        
        # 2. Train Model
        model = train_model(X_train, y_train)

        # 3. Create Test Sequences and Predict
        X_test, y_test = create_sequences(test_data, look_back)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        predictions = model.predict(X_test)
        
        # 4. Inverse Transform Predictions (back to actual price scale)
        predictions = scaler.inverse_transform(predictions)
        
        # 5. Inverse Transform Actual Test Data for comparison
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        # 6. Evaluation (RMSE)
        rmse = np.sqrt(np.mean((predictions - y_test_actual)**2))
        st.subheader("Model Performance")
        st.metric(label="Root Mean Squared Error (RMSE)", value=f"${rmse:,.2f}")
        st.caption("Lower RMSE indicates a better fit to the test data.")
        
        # 7. Visualization
        st.subheader(f"Price Prediction for {ticker_symbol}")
        
        # Prepare the final plot data
        train = df.iloc[0:len(X_train)]
        valid = df.iloc[len(X_train):len(X_train) + len(y_test_actual)]
        valid['Predictions'] = predictions
        
        plt.figure(figsize=(16, 8))
        plt.title('LSTM Model Stock Price Prediction')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Close Price (USD)', fontsize=12)
        plt.plot(train['Close'], label='Training Data')
        plt.plot(valid['Close'], label='Actual Price', color='red')
        plt.plot(valid['Predictions'], label='Predicted Price', color='green')
        plt.legend(loc='lower right')
        
        st.pyplot(plt)
        
        st.subheader("Data Table (Actual vs. Predicted)")
        st.write(valid[['Close', 'Predictions']].tail(10))

else:
    st.info("👈 Enter parameters and click 'Run Prediction' to start the analysis.")
