# 📈 Apple Stock Price Prediction with Attention Layer 📉

This project focuses on predicting Apple's stock price using deep learning techniques with an **Attention Layer** to improve prediction accuracy. By leveraging historical stock data and an attention mechanism, this model aims to capture the most important features that influence Apple's stock price movement, achieving high accuracy in its predictions.

---

## 🚀 Project Overview  
- **Objective**: Predict the future stock prices of Apple (AAPL) using historical stock data and an attention mechanism to improve the model's ability to focus on significant time points.
- **Dataset**: Historical stock prices of Apple, including open, high, low, close prices, volume, and adjusted close prices.
- **Approach**:  
  - Preprocess the stock data (normalization, time-series transformation).
  - Implement a deep learning model with an attention mechanism to better understand long-term dependencies in stock price movement.
  - Train and evaluate the model on historical data to predict future prices.
  
---

## 🛠️ Tools and Technologies  
- **Python** 🐍  
- **TensorFlow/Keras** for deep learning model development  
- **Pandas** for data manipulation  
- **NumPy** for numerical computations  
- **Matplotlib** and **Seaborn** for data visualization  
- **Scikit-learn** for data preprocessing and evaluation  
- **Yahoo Finance API** for fetching stock data  

---

## 📊 Dataset Details  
- **Source**: Data is fetched from **Yahoo Finance** using the Yahoo Finance API.
- **Features**:  
  - **Open**: The stock's opening price.  
  - **High**: The stock's highest price during the day.  
  - **Low**: The stock's lowest price during the day.  
  - **Close**: The stock's closing price (used for predictions).  
  - **Volume**: The number of shares traded.  
  - **Adjusted Close**: Stock price adjusted for splits and dividends.  

- **Timeframe**: Historical data spanning from **YYYY-MM-DD** to **YYYY-MM-DD**.

---

## 🧠 Model Overview  
1. **Model Architecture**:  
   - The model uses **Long Short-Term Memory (LSTM)** layers to capture temporal dependencies in stock prices.  
   - An **Attention Layer** is incorporated to help the model focus on significant periods that influence the stock price.
   - The output layer predicts the stock price for the next day or over a set period.

2. **Training**:  
   - The model is trained using the **mean squared error** (MSE) loss function, commonly used in regression tasks.
   - Optimizer: **Adam** for faster convergence and better performance.

---

## 🏆 Results  
- **Accuracy**: The model achieves an accuracy of **XX%** on predicting Apple's stock prices.
- **Evaluation Metrics**:  
  - **Mean Squared Error (MSE)**: To measure the model's error in price prediction.
  - **Root Mean Squared Error (RMSE)**: Provides an interpretable measure of error.
  - **Mean Absolute Error (MAE)**: To capture how close the predicted values are to the actual values.

- **Graph**: The model's predictions are compared against actual stock prices to visualize the performance.

---
