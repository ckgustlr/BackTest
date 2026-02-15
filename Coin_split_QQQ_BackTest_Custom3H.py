import yfinance as yf
import pandas as pd
import datetime
import time

# Hyperparameters
LEVERAGE = 10
#ENTRY_GAP = 0.003  # 0.3%
SIZE_RATIO = 1 / 120  # 1%
# LONG_PROFIT_NARROW = [0.01, 0.02]  # 1% to 2%
# LONG_PROFIT_WIDE = [0.01, 0.03]  # 1% to 3%
# SHORT_PROFIT_NARROW = [0.01, 0.02]
# SHORT_PROFIT_WIDE = [0.01, 0.03]
# RISK_THRESHOLD = 0.02  # 2%

# Fetch QQQ 1-minute data for the last 30 days
def fetch_qqq_data():
    csv_file = "qqq_1h_1year_data.csv"
    try:
        # If CSV exists, load and return
        raw_data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        valid_idx = pd.to_datetime(raw_data.index, errors='coerce')
        data = raw_data[~valid_idx.isna()].copy()
        data.index = pd.to_datetime(data.index)
        data = data[pd.to_numeric(data['Close'], errors='coerce').notna()]
        print(f"Loaded QQQ data from {csv_file} (valid rows only)")
        return data
    except FileNotFoundError:
        print("CSV not found. Fetching QQQ 1-hour data in 8-day batches for 1 year...")
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=365)
        batch_days = 8
        all_data = []
        batch_start = start_date
        while batch_start < end_date:
            batch_end = min(batch_start + datetime.timedelta(days=batch_days), end_date)
            print(f"Fetching: {batch_start.strftime('%Y-%m-%d')} to {batch_end.strftime('%Y-%m-%d')}")
            batch_data = yf.download(
                "QQQ",
                start=batch_start.strftime('%Y-%m-%d'),
                end=batch_end.strftime('%Y-%m-%d'),
                interval="1h"
            )
            if not batch_data.empty:
                all_data.append(batch_data)
            batch_start = batch_end
            time.sleep(2)  # Avoid rate limits
        if all_data:
            data = pd.concat(all_data)
            data.to_csv(csv_file, index=True, header=True)
            print(f"QQQ 1h 1year data saved to {csv_file}")
            return data
        else:
            raise ValueError("No data fetched. Please check the ticker or date range.")

# Simulated trading function for backtesting
def simulated_trade(data, max_counter=3):
    print(f"Starting simulated trading with max_counter={max_counter}...")

    # Initialize variables
    free_balance = 5000  # Example starting balance
    long_positions = []
    short_positions = []
    last_entry_price_long = None
    last_entry_price_short = None
    first_entry_done = False
    equity_curve = []  # Track free_balance history

    # Remove rows where 'Close' is not numeric and convert to float
    data = data[pd.to_numeric(data['Close'], errors='coerce').notna()].copy()
    data['Close'] = data['Close'].astype(float)
    # Filter data for trading hours (Monday 10:00 to Saturday 10:00)
    data = data[(data.index.dayofweek < 5) & ((data.index.dayofweek != 5) | (data.index.hour < 10))]

    # Initialize 3-hour counter
    three_hour_counter = 1  # Start at 1
    
    # Track if any entry occurred in current cycle (1 to max_counter)
    cycle_entry_occurred = False

    # Initialize realized PnL for long and short separately
    long_realized_pnl = 0
    short_realized_pnl = 0

    # Prepare result rows
    result_rows = []
    event_rows = []
    realized_pnl = 0
    # 루프 내에서 사용할 변수 초기화
    long_avg_price = 0
    short_avg_price = 0
    # Initialize `result` before using it in the loop
    result = {}

    for i, (timestamp, row) in enumerate(data.iterrows()):
        current_price = row['Close']
        short_avg = sum(p['price'] * p['size'] for p in short_positions) / sum(p['size'] for p in short_positions) if short_positions else 0
        short_profit = round((short_avg - current_price) / current_price * 100, 2) if current_price != 0 else 0

        # Initialize profit for this row
        current_short_pnl = 0

        # 1년전 첫 가격에서 롱/숏 동시 진입
        if not first_entry_done:
            current_total_balance = free_balance + sum(p['size'] for p in long_positions) + sum(p['size'] for p in short_positions)
            entry_size = current_total_balance * SIZE_RATIO
            free_balance -= entry_size * 2
            long_positions.append({'price': current_price, 'size': entry_size, 'profit_target': None})
            short_positions.append({'price': current_price, 'size': entry_size, 'profit_target': None})
            last_entry_price_long = current_price
            last_entry_price_short = current_price
            first_entry_done = True
            # Set three_hour_counter to 1 for the initial entry
            three_hour_counter = 1
            
            result = {
                'Datetime': timestamp,
                'Close': row['Close'],
                'short_avg': short_avg,
                'short_profit': short_profit,
                'entry_short': current_price,
                'exit_short': '',
                'short_size': entry_size,
                'short_count': len(short_positions),
                'short_total_size': sum(p['size'] for p in short_positions) if short_positions else '',
                'entry_long': '',
                'exit_long': '',
                'long_size': '',
                'long_count': '',
                'long_total_size': '',
                'long_avg': '',
                'total_balance': free_balance + sum(p['size'] for p in long_positions) + sum(p['size'] for p in short_positions),
                'three_hour_counter': three_hour_counter,
                'long_realized_pnl': 0,
                'short_realized_pnl': 0
            }
            result_rows.append(result.copy())
            event_rows.append(result.copy())
            continue

        # Ensure no division by zero errors in average price calculations
        try:
            long_avg_price = sum(p['price'] * p['size'] for p in long_positions) / sum(p['size'] for p in long_positions) if long_positions else 0
        except ZeroDivisionError:
            long_avg_price = 0

        try:
            short_avg_price = sum(p['price'] * p['size'] for p in short_positions) / sum(p['size'] for p in short_positions) if short_positions else 0
        except ZeroDivisionError:
            short_avg_price = 0

        # Check if short_profit is greater than or equal to 1 to exit short positions
        if short_profit >= 1 and short_positions:
            # Calculate profit as 0.1 times the position size (1% * 10x leverage)
            for position in short_positions:
                exit_pnl = position['size'] * 0.1  # Calculate profit
                free_balance += position['size'] + exit_pnl  # Add principal + profit to free balance
                current_short_pnl += exit_pnl  # Add to current row profit
                short_realized_pnl += exit_pnl  # Add to cumulative realized PnL
            short_positions.clear()  # Clear all short positions
        
        # Create result dictionary AFTER exit logic so total_balance reflects updated free_balance
        result = {
            'Datetime': timestamp,
            'Close': row['Close'],
            'short_avg': short_avg,
            'short_profit': short_profit,
            'entry_short': '',
            'exit_short': current_price if current_short_pnl > 0 else '',
            'short_size': '',
            'short_count': len(short_positions),
            'short_total_size': sum(p['size'] for p in short_positions) if short_positions else '',
            'entry_long': '',
            'exit_long': '',
            'long_size': '',
            'long_count': '',
            'long_total_size': '',
            'long_avg': '',
            'total_balance': free_balance + sum(p['size'] for p in long_positions) + sum(p['size'] for p in short_positions),
            'three_hour_counter': three_hour_counter,
            'long_realized_pnl': 0,
            'short_realized_pnl': current_short_pnl
        }

        # Entry conditions based on counter (1 to max_counter)
        entry_flag = False
        
        # Calculate current total balance for entry size
        current_total_balance = free_balance + sum(p['size'] for p in long_positions) + sum(p['size'] for p in short_positions)
        
        # Short entry: Enter if current_price > short_avg_price
        if current_price > short_avg_price:  # 원래 시나리오 검증 
        #if current_price < short_avg_price: #주마가편 시나리오 
            entry_size = current_total_balance * SIZE_RATIO
            free_balance -= entry_size
            short_positions.append({'price': current_price, 'size': entry_size, 'profit_target': None})
            last_entry_price_short = current_price
            result['entry_short'] = current_price
            result['short_size'] = entry_size
            entry_flag = True
        
        # Long entry: Enter if current_price < long_avg_price
        if current_price < long_avg_price:
            entry_size = current_total_balance * SIZE_RATIO
            free_balance -= entry_size
            long_positions.append({'price': current_price, 'size': entry_size, 'profit_target': None})
            last_entry_price_long = current_price
            entry_flag = True

        # Force entry logic: On counter max_counter, if NO entry occurred in entire cycle (counter 1 to max_counter)
        # Force entry only if cycle_entry_occurred is False (meaning counters 1 to max_counter all failed conditions)
        if three_hour_counter == max_counter and not cycle_entry_occurred:
            entry_size = current_total_balance * SIZE_RATIO
            # Always enter short position on force entry
            free_balance -= entry_size
            short_positions.append({'price': current_price, 'size': entry_size, 'profit_target': None})
            last_entry_price_short = current_price
            result['entry_short'] = current_price
            result['short_size'] = entry_size
            entry_flag = True
        
        # Update cycle_entry_occurred flag
        if entry_flag:
            cycle_entry_occurred = True
            # Recalculate total_balance after entry to reflect updated free_balance and positions
            result['total_balance'] = free_balance + sum(p['size'] for p in long_positions) + sum(p['size'] for p in short_positions)
        
        # Append result rows
        result_rows.append(result.copy())
        if result['entry_long'] or result['entry_short'] or result['exit_long'] or result['exit_short']:
            event_rows.append(result.copy())
        
        # Increment counter and reset cycle flag
        three_hour_counter += 1
        if three_hour_counter > max_counter:
            three_hour_counter = 1
            cycle_entry_occurred = False  # Reset for new cycle

    print("Simulated trading completed.")
    # Return both DataFrames: all timestamps and only events
    return pd.DataFrame(result_rows), pd.DataFrame(event_rows)

# Main function for backtesting
if __name__ == "__main__":
    try:
        qqq_data = fetch_qqq_data()
        
        # Retry logic: increment max_counter until short_count <= 120
        max_counter = 3
        max_short_count = 0
        all_results = None
        event_results = None
        attempt = 0
        
        print("\n" + "="*60)
        print("Starting retry loop to optimize max_counter...")
        print("="*60)
        
        while (max_short_count > 120 or max_short_count == 0):
            attempt += 1
            print(f"\n[Attempt {attempt}] Running simulation with max_counter={max_counter}...")
            print(f"Expected entry pattern: 1, 2, 3, ..., {max_counter} (repeating)")
            
            all_results, event_results = simulated_trade(qqq_data, max_counter=max_counter)
            
            # Check the maximum short_count in the results
            if 'short_count' in all_results.columns:
                # Convert short_count to numeric, replacing non-numeric values with 0
                max_short_count = pd.to_numeric(all_results['short_count'], errors='coerce').fillna(0).astype(int).max()
            else:
                max_short_count = 0
            print(f"[Result] Max short_count: {max_short_count}")
            
            if max_short_count <= 120:
                print(f"\n✓ SUCCESS! Max short_count ({max_short_count}) is within limit (120).")
                print(f"✓ Optimal max_counter: {max_counter}")
                print("="*60 + "\n")
                break
            else:
                print(f"✗ Max short_count ({max_short_count}) exceeds 120.")
                print(f"→ Increasing max_counter from {max_counter} to {max_counter + 1}...")
                max_counter += 1
        
        # Set column order
        col_order = [
            'Datetime','Close','entry_short','exit_short','short_size','short_count','short_total_size','short_avg',
            'short_profit','entry_long','exit_long','long_size','long_count','long_total_size','long_avg',
            'total_balance','three_hour_counter','long_realized_pnl','short_realized_pnl'
        ]
        all_results = all_results[col_order]
        # Remove duplicate timestamps in all_results, keeping the first occurrence
        all_results = all_results.drop_duplicates(subset=['Datetime'], keep='first')
        event_results = event_results[col_order]
        # Remove duplicate timestamps in event_results, keeping the first occurrence
        event_results = event_results.drop_duplicates(subset=['Datetime'], keep='first')
        import subprocess
        from datetime import datetime
        all_results.to_csv("qqq_backtest_custom_all.csv", index=False)
        event_results.to_csv("qqq_backtest_custom_results.csv", index=False)
        # Generate timestamped filename for all_results
        now = datetime.now().strftime("_%y:%m:%d:%H:%M")
        remote_filename = f"qqq_backtest_custom_all{now}.csv"
        # rclone copy to Google Drive (qqq: is your rclone remote name)
        rclone_cmd = [
            "rclone", "copyto",
            "qqq_backtest_custom_all.csv",
            f"qqq:{remote_filename}"
        ]
        try:
            subprocess.run(rclone_cmd, check=True)
            print(f"Uploaded to Google Drive as {remote_filename}")
        except Exception as e:
            print(f"rclone upload failed: {e}")
        print("Backtest completed. Results saved to qqq_backtest_custom_results.csv (events only) and qqq_backtest_custom_all.csv (all timestamps)")

        # =====================
        # 성과 지표
        # =====================
        # 예시: equity_curve를 free_balance 변화로 가정
        # 실제로는 simulated_trade에서 equity_curve 리스트를 반환하도록 수정 필요
        # 아래는 예시 코드
        equity_curve = []
        # 예시: free_balance 변화를 equity_curve에 저장
        # 실제로는 simulated_trade에서 equity_curve를 반환하도록 구현해야 함
        # 예시 코드: backtest_results에 free_balance 컬럼이 있다고 가정
        if hasattr(backtest_results, 'free_balance'):
            equity_curve = backtest_results['free_balance'].tolist()
        # 만약 free_balance 추적이 안되어 있으면 빈 리스트

        if not equity_curve or len(equity_curve) < 2:
            print("DEBUG: equity_curve=", equity_curve)
            raise ValueError("Equity curve is empty or too short. No trades were executed or data is insufficient.")

        equity = pd.Series(equity_curve)
        returns = equity.pct_change().dropna()
        print("DEBUG: returns=", returns)

        if returns.empty:
            print("DEBUG: returns is empty. No trades or insufficient data.")
            raise ValueError("Returns are empty. No trades or insufficient data.")

        mdd = ((equity.cummax() - equity) / equity.cummax()).max()
        win_rate = (returns > 0).mean()

        print("최종 수익률:", equity.iloc[-1] / equity.iloc[0] - 1)
        print("MDD:", mdd)
        print("승률:", win_rate)
        if (returns > 0).any() and (returns < 0).any():
            profit_factor = returns[returns > 0].mean() / abs(returns[returns < 0].mean())
            print("손익비:", profit_factor)
        else:
            print("손익비 계산 불가: 손실 또는 이익이 없음")
    except Exception as e:
        print("Error during backtesting:", e)