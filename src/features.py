import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators and features for supervised machine learning models.
    """
    df = df.copy()
    
    # 1. Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # Distance from Moving Averages (normalized price metrics)
    df['Close_to_SMA20'] = df['Close'] / df['SMA_20'] - 1.0
    df['Close_to_SMA50'] = df['Close'] / df['SMA_50'] - 1.0
    df['Close_to_SMA200'] = df['Close'] / df['SMA_200'] - 1.0
    
    # 2. Momentum & Oscillator (RSI 14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # 3. MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 4. Bollinger Bands (20-period, 2 std dev)
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_Middle'] = df['SMA_20']
    df['BB_Upper'] = df['BB_Middle'] + (rolling_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (rolling_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_PctB'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-9)
    
    # 5. Volatility & Returns
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()
    
    df['Return_3D'] = df['Close'].pct_change(3)
    df['Return_5D'] = df['Close'].pct_change(5)
    df['Return_10D'] = df['Close'].pct_change(10)
    
    # 6. Price Lagged Features
    df['Close_Lag_1'] = df['Close'].shift(1)
    df['Close_Lag_2'] = df['Close'].shift(2)
    df['Close_Lag_3'] = df['Close'].shift(3)
    df['Close_Lag_5'] = df['Close'].shift(5)
    
    # 7. Volume Features
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA_20'] + 1e-9)
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # 8. High-Low Spread
    df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']
    df['CO_Spread'] = (df['Close'] - df['Open']) / df['Open']

    return df

def build_supervised_dataset(df: pd.DataFrame, target_type: str = 'regression', forecast_horizon: int = 1):
    """
    Constructs feature matrix X and target y for Supervised Learning.
    
    Parameters:
        df: DataFrame with OHLCV data.
        target_type: 'regression' (predict next day close) or 'classification' (predict direction: 1 for up, 0 for down).
        forecast_horizon: Shift horizon in trading days (default 1).
        
    Returns:
        X (pd.DataFrame), y (pd.Series), clean_df (pd.DataFrame)
    """
    df_feat = add_technical_indicators(df)
    
    # Create Supervised Targets
    df_feat['Target_Close'] = df_feat['Close'].shift(-forecast_horizon)
    df_feat['Target_Return'] = (df_feat['Target_Close'] - df_feat['Close']) / df_feat['Close']
    df_feat['Target_Direction'] = (df_feat['Target_Return'] > 0).astype(int)
    
    # Drop NaNs resulting from rolling metrics and shift operations
    df_clean = df_feat.dropna().copy()
    
    # Define Feature Column List
    feature_cols = [
        'Close', 'Open', 'High', 'Low', 'Volume',
        'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
        'EMA_12', 'EMA_26',
        'Close_to_SMA20', 'Close_to_SMA50', 'Close_to_SMA200',
        'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Width', 'BB_PctB',
        'Daily_Return', 'Log_Return', 'Volatility_20',
        'Return_3D', 'Return_5D', 'Return_10D',
        'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5',
        'Volume_Ratio', 'Volume_Change',
        'HL_Spread', 'CO_Spread'
    ]
    
    # Filter features that are actually in df_clean
    feature_cols = [c for c in feature_cols if c in df_clean.columns]
    
    X = df_clean[feature_cols]
    
    if target_type == 'classification':
        y = df_clean['Target_Direction']
    else:
        y = df_clean['Target_Close']
        
    logger.info(f"Constructed Supervised dataset: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y, df_clean

if __name__ == "__main__":
    from data_loader import fetch_stock_data
    df = fetch_stock_data("AAPL", period="2y")
    X, y, clean_df = build_supervised_dataset(df, target_type='regression')
    print("Features shape:", X.shape)
    print("Target head:\n", y.head())
