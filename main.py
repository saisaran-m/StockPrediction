import argparse
import sys
import os
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import fetch_stock_data
from src.features import build_supervised_dataset
from src.models import SupervisedStockPredictor
from src.evaluate import run_trading_backtest

def run_pipeline(ticker: str = "AAPL", period: str = "2y", target_type: str = "regression"):
    print("=" * 70)
    print(f" STOCK PRICE PREDICTION USING SUPERVISED MACHINE LEARNING ")
    print(f" Target Ticker: {ticker} | Period: {period} | Task: {target_type.upper()}")
    print("=" * 70)
    
    # 1. Load Data
    print("\n[Step 1/4] Fetching historical stock data...")
    df = fetch_stock_data(ticker=ticker, period=period)
    print(f"Data Loaded: {len(df)} trading rows from {df.index[0]} to {df.index[-1]}")
    
    # 2. Feature Engineering & Supervised Dataset Construction
    print("\n[Step 2/4] Engineering Technical Features & Supervised Targets...")
    X, y, clean_df = build_supervised_dataset(df, target_type=target_type)
    print(f"Supervised Feature Matrix: {X.shape[0]} samples, {X.shape[1]} technical features")
    
    # 3. Model Training & Evaluation
    print("\n[Step 3/4] Training Supervised Machine Learning Models...")
    predictor = SupervisedStockPredictor(target_type=target_type)
    results, summary_df = predictor.train_and_evaluate(X, y)
    
    print("\n" + "-" * 50)
    print(" MODEL PERFORMANCE EVALUATION MATRIX")
    print("-" * 50)
    print(summary_df.to_string(index=False))
    
    best_name = predictor.best_model_name
    print(f"\n Top Performing Supervised Model: '{best_name}'")
    
    # Feature Importances
    feat_imp = predictor.get_feature_importance(list(X.columns))
    if not feat_imp.empty:
        print("\n Top 10 Most Important Features:")
        print(feat_imp.head(10).to_string(index=False))
        
    # 4. Strategy Backtesting
    print("\n[Step 4/4] Running Financial Strategy Backtest...")
    best_preds = results[best_name]['predictions']
    backtest = run_trading_backtest(clean_df, best_preds, target_type=target_type)
    
    print("\n" + "-" * 50)
    print(" FINANCIAL BACKTEST & STRATEGY RESULTS")
    print("-" * 50)
    print(f"  Strategy Return:     {backtest['total_strategy_return']:>8.2f} %")
    print(f"  Buy & Hold Return:   {backtest['total_benchmark_return']:>8.2f} %")
    print(f"  Sharpe Ratio:        {backtest['sharpe_ratio']:>8.2f}")
    print(f"  Max Drawdown:        {backtest['max_drawdown']:>8.2f} %")
    print(f"  Signal Win Rate:     {backtest['win_rate']:>8.2f} %")
    print("=" * 70)
    
    return predictor, results, backtest, clean_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Price Prediction using Supervised ML")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (e.g. AAPL, NVDA, MSFT)")
    parser.add_argument("--period", type=str, default="2y", help="Historical data period (e.g. 1y, 2y, 5y)")
    parser.add_argument("--target", type=str, default="regression", choices=["regression", "classification"], help="Supervised task type")
    
    args = parser.parse_args()
    run_pipeline(ticker=args.ticker, period=args.period, target_type=args.target)
