import yfinance as yf
import pandas as pd

# Fetch QQQ 1-minute data for the last 30 days
def fetch_qqq_data():
    print("Fetching QQQ 1-minute data from Yahoo Finance...")
    data = yf.download("QQQ", period="30d", interval="1m")
    if data.empty:
        raise ValueError("No data fetched. Please check the ticker or date range.")
    
    # Save data to CSV for reuse
    csv_file = "qqq_1min_data.csv"
    data.to_csv(csv_file)
    print(f"QQQ data saved to {csv_file}")
    return data

# Simulated trading function for backtesting
def simulated_trade(data):
    print("Starting simulated trading...")
    
    # Example: Simple moving average crossover strategy
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    data['Signal'] = 0
    data.loc[data['SMA_10'] > data['SMA_50'], 'Signal'] = 1
    data.loc[data['SMA_10'] <= data['SMA_50'], 'Signal'] = -1

    # Backtest results
    data['Daily_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Signal'].shift(1) * data['Daily_Return']

    total_return = (1 + data['Strategy_Return']).cumprod().iloc[-1] - 1
    print(f"Total strategy return: {total_return * 100:.2f}%")

    return data

# Main function for backtesting
if __name__ == "__main__":
    try:
        qqq_data = fetch_qqq_data()
        backtest_results = simulated_trade(qqq_data)
        backtest_results.to_csv("qqq_backtest_results.csv")
        print("Backtest completed. Results saved to qqq_backtest_results.csv")
    except Exception as e:
        print("Error during backtesting:", e)