# 📈 Institutional Supervised Machine Learning Stock Intelligence Platform

An institutional-grade **Supervised Machine Learning Stock Price Forecasting and Intelligence Platform** built with Python, Scikit-Learn, XGBoost, Flask, and dynamic glassmorphism UI.

This application uses historical stock price data (Open, High, Low, Close, Volume) from real global corporations (Alphabet/Google, Apple, NVIDIA, Amazon, Tesla, Microsoft, Meta, Netflix, AMD, S&P 500) to train supervised algorithms that predict future stock prices, multi-day target horizons, and day-by-day month forecasts.

---

## 🌟 Key Features

- ** real Corporations & Stock Data**:
  - Predict stock prices for verified corporations: **Alphabet Inc. (Google - `GOOGL`)**, **Apple Inc. (`AAPL`)**, **NVIDIA (`NVDA`)**, **Amazon (`AMZN`)**, **Tesla (`TSLA`)**, **Microsoft (`MSFT`)**, **Meta (`META`)**, **Netflix (`NFLX`)**, **AMD (`AMD`)**, and **S&P 500 (`SPY`)**.
- ** 34 Automated Feature Engineering Indicators**:
  - Computes technical indicators: Simple Moving Averages (SMA 10, 20, 50, 200), Exponential Moving Averages (EMA 12, 26), Relative Strength Index (RSI 14), MACD, Bollinger Bands, Historical Volatility, Price Lags, and Target Shifts.
- ** Supervised Learning Algorithms**:
  - Compares 4 supervised regression and classification models: **Linear & Ridge Regression**, **Random Forest Regressor**, **XGBoost**, and **Support Vector Machine (SVR/SVC)**.
- ** Multi-Horizon Target Forecasting**:
  - Generates price projections for **1-Day (Tomorrow)**, **3-Day**, **5-Day**, and **7-Day** future price targets.
- ** Current Month Day-by-Day Forecast Table**:
  - Provides a detailed calendar forecast for every trading day of the current month with predicted prices ($), daily increase/decrease dollar changes ($), daily returns (%), and Bullish/Bearish signals.
- ** Formatted CSV Excel Report Export**:
  - Download structured `.CSV` reports compatible with Microsoft Excel and Google Sheets containing actual stock prices, predicted prices, model error, technical indicators, and trade signals.
- ** Interactive Educational Video Canvas**:
  - Includes a custom interactive video canvas player demonstrating step-by-step how supervised machine learning algorithms ingest market data, build features, train models, and backtest portfolio strategies.
- ** Dark / Light Theme Switching**:
  - Glassmorphic Bloomberg-terminal UI with instant Dark/Light theme mode toggling.

---

## 🧠 How Supervised Machine Learning Stock Prediction Works

1. **Data Collection ($X$)**:
   - The platform downloads historical daily price data from stock exchanges for the selected time period (1 to 5 years).
2. **Technical Feature Engineering**:
   - Math indicators (SMA, RSI, MACD, Volatility, Lags) are derived to build input matrix $X_t$.
3. **Supervised Target Mapping ($Y_{t+1}$)**:
   - The supervised model learns a mapping function $f(X_t) \rightarrow Y_{t+1}$, where $Y_{t+1}$ is the actual stock price on the next trading day.
4. **Time-Series Split & Evaluation**:
   - Data is split chronologically into training (80%) and testing (20%) sets to evaluate out-of-sample accuracy using **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **$R^2$ Score**.
5. **Backtesting & Equity Curve**:
   - Simulates trading strategy performance starting with $1.00 capital against a Buy-and-Hold benchmark.

---

## 🚀 Quickstart: Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/saisaran-m/StockPrediction.git
cd StockPrediction
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

*(Or install core packages directly)*:
```bash
pip install flask pandas numpy scikit-learn xgboost yfinance matplotlib chart.js
```

### 3. Launch the Application
```bash
python app.py
```

### 4. Open in Web Browser
Navigate to **`http://localhost:5050`** (or **`http://127.0.0.1:5050`**).

---

## 🌐 Deploying Live Online (Free Hosting Options)

To get a **Public Live URL** for your project, you can deploy for free using any of the following platforms:

### Option A: Render.com (Recommended - 5 Minutes)
1. Sign up at [Render.com](https://render.com).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository `saisaran-m/StockPrediction`.
4. Set:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Click **Create Web Service**. Render will generate a live URL (e.g. `https://stock-prediction.onrender.com`).

### Option B: PythonAnywhere.com
1. Create a free account at [PythonAnywhere.com](https://www.pythonanywhere.com).
2. Open a bash console and clone your repo:
   ```bash
   git clone https://github.com/saisaran-m/StockPrediction.git
   ```
3. Go to the **Web** tab, create a new Web App selecting **Flask**, point to `app.py`, and hit **Reload**.

---

## 📁 Repository Directory Structure

```
StockPrediction/
├── app.py                      # Flask Server & REST API endpoints
├── main.py                     # CLI pipeline runner
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation & User Guide
├── stock_prediction_notebook.ipynb # Jupyter Notebook walkthrough
├── src/
│   ├── data_loader.py          # Stock downloader & Ticker resolver
│   ├── features.py             # 34 Technical indicator pipeline
│   ├── models.py               # Supervised regression/classification trainers
│   └── evaluate.py             # MAE, RMSE, Sharpe ratio & backtesting
├── templates/
│   └── index.html              # Dashboard view & layout
└── static/
    ├── style.css               # Glassmorphism styling & theme design
    └── app.js                  # Client logic, Chart.js & video canvas
```

---

## 📄 License
Distributed under the MIT License. Feel free to use, modify, and build upon this project.
