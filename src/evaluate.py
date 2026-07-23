import numpy as np
import pandas as pd
from typing import Dict, Any

def run_trading_backtest(df_clean: pd.DataFrame, predictions: np.ndarray, test_size: float = 0.2, target_type: str = 'regression') -> Dict[str, Any]:
    """
    Simulates out-of-sample trading performance based on Supervised ML model signals.
    Compares strategy cumulative returns against Buy & Hold benchmark.
    
    Parameters:
        df_clean: Cleaned DataFrame with Close prices and Date index.
        predictions: Out-of-sample model predictions array.
        test_size: Ratio of data used for test set.
        target_type: 'regression' or 'classification'.
        
    Returns:
        Dictionary containing cumulative returns series, Sharpe Ratio, Max Drawdown, and Win Rate.
    """
    split_idx = int(len(df_clean) * (1 - test_size))
    test_df = df_clean.iloc[split_idx:].copy()
    
    if len(test_df) != len(predictions):
        test_df = test_df.iloc[:len(predictions)].copy()
        
    test_df['Prediction'] = predictions
    
    # Generate Trading Signal (1 = Long / Buy, 0 = Hold Cash)
    if target_type == 'regression':
        test_df['Signal'] = (test_df['Prediction'] > test_df['Close']).astype(int)
    else:
        test_df['Signal'] = (test_df['Prediction'] == 1).astype(int)
        
    # Calculate Daily Returns
    test_df['Market_Return'] = test_df['Close'].pct_change().fillna(0)
    # Strategy Return = Signal (t) * Market Return (t+1)
    test_df['Strategy_Return'] = test_df['Signal'].shift(1).fillna(0) * test_df['Market_Return']
    
    # Cumulative Returns
    test_df['Buy_Hold_Equity'] = (1 + test_df['Market_Return']).cumprod()
    test_df['Strategy_Equity'] = (1 + test_df['Strategy_Return']).cumprod()
    
    # Performance Metrics
    total_benchmark_return = (test_df['Buy_Hold_Equity'].iloc[-1] - 1) * 100
    total_strategy_return = (test_df['Strategy_Equity'].iloc[-1] - 1) * 100
    
    # Annualized Sharpe Ratio (assuming risk-free rate = 0%)
    strat_daily_std = test_df['Strategy_Return'].std()
    sharpe_ratio = (test_df['Strategy_Return'].mean() / (strat_daily_std + 1e-9)) * np.sqrt(252) if strat_daily_std > 0 else 0.0
    
    # Max Drawdown
    equity_peak = test_df['Strategy_Equity'].cummax()
    drawdown = (test_df['Strategy_Equity'] - equity_peak) / equity_peak
    max_drawdown = drawdown.min() * 100
    
    # Win Rate on active trading days
    trade_days = test_df[test_df['Signal'].shift(1) == 1]
    win_rate = (trade_days['Market_Return'] > 0).mean() * 100 if len(trade_days) > 0 else 0.0
    
    dates = [str(d)[:10] for d in test_df.index]
    
    return {
        'dates': dates,
        'buy_hold_equity': test_df['Buy_Hold_Equity'].round(4).tolist(),
        'strategy_equity': test_df['Strategy_Equity'].round(4).tolist(),
        'total_benchmark_return': round(total_benchmark_return, 2),
        'total_strategy_return': round(total_strategy_return, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'max_drawdown': round(max_drawdown, 2),
        'win_rate': round(win_rate, 2),
        'test_df': test_df
    }

if __name__ == "__main__":
    from data_loader import fetch_stock_data
    from features import build_supervised_dataset
    from models import SupervisedStockPredictor
    
    df = fetch_stock_data("AAPL", period="2y")
    X, y, clean_df = build_supervised_dataset(df, target_type='regression')
    predictor = SupervisedStockPredictor(target_type='regression')
    results, summary_df = predictor.train_and_evaluate(X, y)
    
    best_name = predictor.best_model_name
    best_preds = results[best_name]['predictions']
    backtest = run_trading_backtest(clean_df, best_preds, target_type='regression')
    print("Backtest Strategy Return:", backtest['total_strategy_return'], "%")
    print("Buy & Hold Return:", backtest['total_benchmark_return'], "%")
