import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Comprehensive Real Corporate Stock Metadata with Real Logo URLs
STOCK_METADATA = {
    "GOOGL": {
        "ticker": "GOOGL",
        "name": "Alphabet Inc. (Google)",
        "sector": "Communication Services / Internet & AI",
        "exchange": "NASDAQ",
        "base_price": 182.50,
        "logo_url": "https://logo.clearbit.com/google.com",
        "description": "Dominates global internet search, digital advertising, YouTube, Android, and Google Cloud."
    },
    "AAPL": {
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "sector": "Technology / Consumer Electronics",
        "exchange": "NASDAQ",
        "base_price": 224.80,
        "logo_url": "https://logo.clearbit.com/apple.com",
        "description": "Designs and manufactures iPhones, MacBooks, iPads, Apple Watch, and iOS software services."
    },
    "AMZN": {
        "ticker": "AMZN",
        "name": "Amazon.com, Inc.",
        "sector": "Consumer Cyclical / E-Commerce & Cloud",
        "exchange": "NASDAQ",
        "base_price": 186.20,
        "logo_url": "https://logo.clearbit.com/amazon.com",
        "description": "Global leader in e-commerce retail, Amazon Web Services (AWS) cloud, and AI digital services."
    },
    "NVDA": {
        "ticker": "NVDA",
        "name": "NVIDIA Corporation",
        "sector": "Technology / Semiconductors & AI Hardware",
        "exchange": "NASDAQ",
        "base_price": 128.50,
        "logo_url": "https://logo.clearbit.com/nvidia.com",
        "description": "World leader in GPU microchips powering artificial intelligence models and data centers."
    },
    "TSLA": {
        "ticker": "TSLA",
        "name": "Tesla, Inc.",
        "sector": "Consumer Cyclical / EV & Clean Energy",
        "exchange": "NASDAQ",
        "base_price": 248.60,
        "logo_url": "https://logo.clearbit.com/tesla.com",
        "description": "Pioneer of autonomous electric vehicles, battery energy storage, and solar technology."
    },
    "MSFT": {
        "ticker": "MSFT",
        "name": "Microsoft Corporation",
        "sector": "Technology / Enterprise Software & Cloud",
        "exchange": "NASDAQ",
        "base_price": 448.90,
        "logo_url": "https://logo.clearbit.com/microsoft.com",
        "description": "Develops Azure cloud infrastructure, Windows OS, Office 365, and Copilot AI systems."
    },
    "META": {
        "ticker": "META",
        "name": "Meta Platforms, Inc.",
        "sector": "Communication Services / Social Media & VR",
        "exchange": "NASDAQ",
        "base_price": 508.30,
        "logo_url": "https://logo.clearbit.com/meta.com",
        "description": "Operates Facebook, Instagram, WhatsApp, Threads, and Quest VR virtual reality platforms."
    },
    "NFLX": {
        "ticker": "NFLX",
        "name": "Netflix, Inc.",
        "sector": "Communication Services / Entertainment",
        "exchange": "NASDAQ",
        "base_price": 645.10,
        "logo_url": "https://logo.clearbit.com/netflix.com",
        "description": "Global streaming subscription service providing movies, TV shows, and original media content."
    },
    "AMD": {
        "ticker": "AMD",
        "name": "Advanced Micro Devices, Inc.",
        "sector": "Technology / Semiconductors",
        "exchange": "NASDAQ",
        "base_price": 158.40,
        "logo_url": "https://logo.clearbit.com/amd.com",
        "description": "Designs high-performance semiconductor processors, Instinct AI accelerators, and GPUs."
    },
    "INTC": {
        "ticker": "INTC",
        "name": "Intel Corporation",
        "sector": "Technology / Semiconductors",
        "exchange": "NASDAQ",
        "base_price": 31.50,
        "logo_url": "https://logo.clearbit.com/intel.com",
        "description": "Leading semiconductor chip manufacturer for personal computers, enterprise servers, and data centers."
    },
    "DIS": {
        "ticker": "DIS",
        "name": "The Walt Disney Company",
        "sector": "Communication Services / Entertainment",
        "exchange": "NYSE",
        "base_price": 96.80,
        "logo_url": "https://logo.clearbit.com/disney.com",
        "description": "Global entertainment icon operating media networks, theme parks, Disney+, and film studios."
    },
    "JPM": {
        "ticker": "JPM",
        "name": "JPMorgan Chase & Co.",
        "sector": "Financial / Global Investment Banking",
        "exchange": "NYSE",
        "base_price": 204.50,
        "logo_url": "https://logo.clearbit.com/jpmorganchase.com",
        "description": "Largest multinational financial services firm and investment bank in the United States."
    },
    "KO": {
        "ticker": "KO",
        "name": "The Coca-Cola Company",
        "sector": "Consumer Staples / Beverages",
        "exchange": "NYSE",
        "base_price": 64.20,
        "logo_url": "https://logo.clearbit.com/coca-colacompany.com",
        "description": "World's largest non-alcoholic beverage corporation selling products in over 200 countries."
    },
    "SPY": {
        "ticker": "SPY",
        "name": "SPDR S&P 500 ETF Trust",
        "sector": "Financial / Broad US Stock Market Index",
        "exchange": "NYSE Arca",
        "base_price": 554.20,
        "logo_url": "https://logo.clearbit.com/ssga.com",
        "description": "Tracks the benchmark S&P 500 index representing 500 largest US publicly traded corporations."
    }
}

# Alias Map to resolve spelling variations, typos, and common names
TICKER_ALIASES = {
    "GOOGLE": "GOOGL", "GOOG": "GOOGL", "ALPHABET": "GOOGL", "GOOGL": "GOOGL",
    "AMAZON": "AMZN", "AMAZN": "AMZN", "AMXN": "AMZN", "AMZN": "AMZN",
    "APPLE": "AAPL", "APPL": "AAPL", "AAPL": "AAPL",
    "TESLA": "TSLA", "TESL": "TSLA", "TSLA": "TSLA",
    "MICROSOFT": "MSFT", "MSF": "MSFT", "MSFT": "MSFT",
    "NVIDIA": "NVDA", "NVID": "NVDA", "NVDA": "NVDA",
    "META": "META", "FACEBOOK": "META", "FB": "META",
    "NETFLIX": "NFLX", "NETFLX": "NFLX", "NFLX": "NFLX",
    "INTEL": "INTC", "INTC": "INTC",
    "DISNEY": "DIS", "DIS": "DIS",
    "JPMORGAN": "JPM", "JPM": "JPM", "CHASE": "JPM",
    "COCA COLA": "KO", "COCACOLA": "KO", "KO": "KO",
    "S&P": "SPY", "SP500": "SPY", "S&P500": "SPY", "SPY": "SPY"
}

def resolve_ticker(input_str: str) -> str:
    """Resolves user search string or typo to official ticker symbol."""
    if not input_str:
        return "GOOGL"
    clean = input_str.strip().upper()
    if clean in TICKER_ALIASES:
        return TICKER_ALIASES[clean]
    for key, val in TICKER_ALIASES.items():
        if key in clean:
            return val
    return clean

def get_company_info(input_str: str) -> dict:
    """Returns official company metadata given a ticker symbol or company name."""
    ticker = resolve_ticker(input_str)
    if ticker in STOCK_METADATA:
        info = STOCK_METADATA[ticker].copy()
        return info
    
    # Generic fallback if ticker is not in metadata list
    clean_name = ticker.capitalize()
    return {
        "ticker": ticker,
        "name": f"{clean_name} Corporation",
        "sector": "Publicly Traded Stock Equity",
        "exchange": "Global Stock Market",
        "base_price": 150.0,
        "logo_url": f"https://ui-avatars.com/api/?name={ticker}&background=6366f1&color=fff&bold=true",
        "description": f"Historical equity trading data for {ticker}."
    }

def generate_synthetic_stock_data(ticker: str = "AAPL", days: int = 1250) -> pd.DataFrame:
    """
    Generates realistic synthetic stock market data using Geometric Brownian Motion (GBM).
    """
    ticker_clean = resolve_ticker(ticker)
    info = get_company_info(ticker_clean)
    start_p = info.get("base_price", 150.0)
    
    logger.info(f"Generating synthetic stock data for '{info['name']}' ({ticker_clean}) over {days} days...")
    
    np.random.seed(abs(hash(ticker_clean)) % 100000)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    
    mu = 0.0006  # Positive annual drift
    sigma = 0.018  # Daily volatility
    
    returns = np.random.normal(mu, sigma, days)
    price_path = start_p * np.exp(np.cumsum(returns))
    
    high_noise = np.abs(np.random.normal(0.005, 0.003, days))
    low_noise = np.abs(np.random.normal(0.005, 0.003, days))
    
    close = price_path
    open_p = np.roll(close, 1)
    open_p[0] = start_p * (1 + np.random.normal(0, 0.005))
    
    high = np.maximum(open_p, close) * (1 + high_noise)
    low = np.minimum(open_p, close) * (1 - low_noise)
    volume = np.random.randint(15_000_000, 85_000_000, days)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_p,
        'High': high,
        'Low': low,
        'Close': close,
        'Adj Close': close,
        'Volume': volume
    })
    
    df.set_index('Date', inplace=True)
    return df

def fetch_stock_data(ticker: str = "AAPL", period: str = "5y", start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Fetches historical OHLCV stock data via yfinance.
    """
    ticker_clean = resolve_ticker(ticker)
    logger.info(f"Attempting to fetch historical stock data for '{ticker_clean}'...")
    
    try:
        import yfinance as yf
        
        if start_date and end_date:
            df = yf.download(ticker_clean, start=start_date, end=end_date, progress=False)
        else:
            df = yf.download(ticker_clean, period=period, progress=False)
            
        if df.empty or len(df) < 50:
            logger.warning(f"yfinance returned empty data for '{ticker_clean}'. Using synthetic fallback.")
            return generate_synthetic_stock_data(ticker=ticker_clean)
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}'")
                
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']
            
        df = df.ffill().bfill()
        logger.info(f"Successfully loaded {len(df)} historical rows for '{ticker_clean}'.")
        return df

    except Exception as e:
        logger.warning(f"Error fetching data via yfinance for '{ticker_clean}': {e}. Using synthetic fallback.")
        return generate_synthetic_stock_data(ticker=ticker_clean)
