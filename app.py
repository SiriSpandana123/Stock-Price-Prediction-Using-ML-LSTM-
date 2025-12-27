import numpy as np
import pandas as pd
import plotly.graph_objects as go
from flask import Flask, render_template, request, redirect, url_for
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, median_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

app = Flask(__name__)

# Define the function to calculate regression metrics
def calculate_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'evs': evs, 'medae': medae}

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        dataset = pd.read_csv(file)
        dataset['Open'] = pd.to_numeric(dataset['Open'], errors='coerce')
        dataset.dropna(subset=['Open'], inplace=True)
        data = dataset['Open'].values.reshape(-1, 1)
        sc = MinMaxScaler(feature_range=(0, 1))
        data_scaled = sc.fit_transform(data)
        X, y = [], []
        for i in range(60, len(data_scaled)):
            X.append(data_scaled[i-60:i, 0])
            y.append(data_scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        regressor = Sequential()
        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units=1))
        regressor.compile(optimizer='adam', loss='mean_squared_error')

        history = regressor.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

        predicted_stock_price_train = regressor.predict(X_train)
        predicted_stock_price_test = regressor.predict(X_test)
        predicted_stock_price_train = sc.inverse_transform(predicted_stock_price_train)
        predicted_stock_price_test = sc.inverse_transform(predicted_stock_price_test)
        y_train_rescaled = sc.inverse_transform(y_train.reshape(-1, 1))
        y_test_rescaled = sc.inverse_transform(y_test.reshape(-1, 1))

        train_metrics = calculate_regression_metrics(y_train_rescaled, predicted_stock_price_train)
        test_metrics = calculate_regression_metrics(y_test_rescaled, predicted_stock_price_test)

        next_day_price = regressor.predict(X[-1].reshape(1, X[-1].shape[0], X[-1].shape[1]))
        next_day_price = sc.inverse_transform(next_day_price)[0][0]

        # Plot 1: Raw Open Prices
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=np.arange(len(data)), y=data.flatten(), mode='lines', name='Raw Open Prices'))
        fig1.update_layout(title='Raw Open Prices', xaxis_title='Time', yaxis_title='Price')

        # Plot 2: Scaled Open Prices
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=np.arange(len(data_scaled)), y=data_scaled.flatten(), mode='lines', name='Scaled Open Prices'))
        fig2.update_layout(title='Scaled Open Prices', xaxis_title='Time', yaxis_title='Scaled Price')

        # Plot 3: Training and Validation Loss
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=np.arange(len(history.history['loss'])), y=history.history['loss'], mode='lines', name='Training Loss'))
        fig3.add_trace(go.Scatter(x=np.arange(len(history.history['val_loss'])), y=history.history['val_loss'], mode='lines', name='Validation Loss'))
        fig3.update_layout(title='Training and Validation Loss', xaxis_title='Epoch', yaxis_title='Loss')

        # Plot 4: Predicted vs Actual Stock Prices (Training Data)
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=np.arange(len(y_train_rescaled)), y=y_train_rescaled.flatten(), mode='lines', name='Real Stock Price (Training Data)', line=dict(color='red')))
        fig4.add_trace(go.Scatter(x=np.arange(len(predicted_stock_price_train)), y=predicted_stock_price_train.flatten(), mode='lines', name='Predicted Stock Price (Training Data)', line=dict(color='blue')))
        fig4.update_layout(title="Stock Price Prediction (Training Data)", xaxis_title='Time', yaxis_title="Stock Price")

        # Plot 5: Predicted vs Actual Stock Prices (Testing Data)
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=np.arange(len(y_test_rescaled)), y=y_test_rescaled.flatten(), mode='lines', name='Real Stock Price (Testing Data)', line=dict(color='red')))
        fig5.add_trace(go.Scatter(x=np.arange(len(predicted_stock_price_test)), y=predicted_stock_price_test.flatten(), mode='lines', name='Predicted Stock Price (Testing Data)', line=dict(color='blue')))
        fig5.update_layout(title="Stock Price Prediction (Testing Data)", xaxis_title='Time', yaxis_title="Stock Price")

        # Plot 6: Distribution of Prediction Errors
        errors = y_test_rescaled - predicted_stock_price_test
        fig6 = go.Figure()
        fig6.add_trace(go.Histogram(x=errors.flatten(), nbinsx=50, marker_color='blue', opacity=0.7))
        fig6.update_layout(title='Distribution of Prediction Errors', xaxis_title='Prediction Error', yaxis_title='Frequency')

        graphJSON1 = fig1.to_json()
        graphJSON2 = fig2.to_json()
        graphJSON3 = fig3.to_json()
        graphJSON4 = fig4.to_json()
        graphJSON5 = fig5.to_json()
        graphJSON6 = fig6.to_json()

        return render_template('results.html', train_metrics=train_metrics, test_metrics=test_metrics,
                               graphJSON1=graphJSON1, graphJSON2=graphJSON2,
                               graphJSON3=graphJSON3, graphJSON4=graphJSON4,
                               graphJSON5=graphJSON5, graphJSON6=graphJSON6,
                               next_day_price=next_day_price)

if __name__ == "__main__":
    app.run(debug=True)
