import yfinance as yf
import pandas as pd
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import subprocess

# Hyperparameters
LEVERAGE = 10
SHORT_SIZE_RATIO = None  # Will be set in main
LONG_SIZE_RATIO = None   # Will be set in main
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
def simulated_trade(data, short_max_counter=3, long_max_counter=3, short_size_ratio=1/320, long_size_ratio=1/320):
    print(f"Starting simulated trading with short_max_counter={short_max_counter}, long_max_counter={long_max_counter}, short_ratio={short_size_ratio:.4f}, long_ratio={long_size_ratio:.4f}...")
    # Calculate size ratios for short and long
    # 이전 계산: SHORT_SIZE_RATIO = short_max_counter / 120
    #           LONG_SIZE_RATIO = long_max_counter / 120
    # 
    # 새로운 계산: 
    # 일주일 120시간, 3시간마다 진입 = 40번
    # 한달 4주 = 160번
    # 롱+숏 = 320번 총 진입
    SHORT_SIZE_RATIO = short_size_ratio
    LONG_SIZE_RATIO = long_size_ratio

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

    # Initialize separate counters for short and long
    short_hour_count = 1  # Start at 1
    long_hour_count = 1   # Start at 1

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
    # Cumulative entry amounts for rate calculations (principal basis)
    short_amount_cum = 0
    long_amount_cum = 0
    # Track exits to record in result dict
    short_exit_occurred = False
    long_exit_occurred = False

    for i, (timestamp, row) in enumerate(data.iterrows()):
        # Track if exits occurred (reset for this iteration)
        short_exit_occurred = False
        long_exit_occurred = False
        
        current_price = row['Close']
        short_avg = sum(p['price'] * p['size'] for p in short_positions) / sum(p['size'] for p in short_positions) if short_positions else 0
        short_count = sum(p.get('unit', 1) for p in short_positions) if short_positions else 0
        short_profit = round((short_avg - current_price) / current_price * 100, 2) if current_price != 0 else 0

        # Calculate long metrics
        long_avg = sum(p['price'] * p['size'] for p in long_positions) / sum(p['size'] for p in long_positions) if long_positions else 0
        long_count = sum(p.get('unit', 1) for p in long_positions) if long_positions else 0
        long_profit = round((current_price - long_avg) / long_avg * 100, 2) if long_avg != 0 else 0

        # Initialize profit for this row
        current_short_pnl = 0
        current_long_pnl = 0

        # 1년전 첫 가격에서 롱/숏 동시 진입
        if not first_entry_done:
            long_val_0 = sum(p['size'] * current_price * (1 + LEVERAGE * (current_price - p['price']) / p['price']) for p in long_positions) if long_positions else 0
            short_val_0 = sum(p['size'] * current_price * (1 + LEVERAGE * (p['price'] - current_price) / p['price']) for p in short_positions) if short_positions else 0
            current_total_balance = free_balance + long_val_0 + short_val_0
            long_entry_size = (current_total_balance * LONG_SIZE_RATIO) / current_price
            short_entry_size = (current_total_balance * SHORT_SIZE_RATIO) / current_price
            free_balance -= (long_entry_size * current_price + short_entry_size * current_price)
            long_positions.append({'price': current_price, 'size': long_entry_size, 'unit': 1.0, 'profit_target': None})
            short_positions.append({'price': current_price, 'size': short_entry_size, 'unit': 1.0, 'profit_target': None})
            # Update cumulative amounts for first entry
            long_amount_cum = long_entry_size * current_price
            short_amount_cum = short_entry_size * current_price
            last_entry_price_long = current_price
            last_entry_price_short = current_price
            first_entry_done = True
            # Set counters to 1 for the initial entry
            short_hour_count = 1
            long_hour_count = 1
            # Calculate totals and total balance after initial entries (포지션 평가 = 청산 시 받을 금액, 레버리지 반영)
            short_total_size = sum(p['size'] for p in short_positions) if short_positions else 0
            long_total_size = sum(p['size'] for p in long_positions) if long_positions else 0
            long_val = sum(p['size'] * current_price * (1 + LEVERAGE * (current_price - p['price']) / p['price']) for p in long_positions) if long_positions else 0
            short_val = sum(p['size'] * current_price * (1 + LEVERAGE * (p['price'] - current_price) / p['price']) for p in short_positions) if short_positions else 0
            total_balance = free_balance + long_val + short_val

            result = {
                'Datetime': timestamp,
                'Close': row['Close'],
                'short_avg': short_avg,
                'short_profit': short_profit,
                'entry_short': current_price,
                'exit_short': '',
                'short_size': short_entry_size,
                'short_count': 1.0,
                'short_total_size': short_total_size,
                'entry_long': current_price,
                'exit_long': '',
                'long_size': long_entry_size,
                'long_count': 1.0,
                'long_total_size': long_total_size,
                'long_avg': '',
                'long_profit': '',
                'free_balance': round(free_balance, 2),
                'long_val': round(long_val, 2),
                'short_val': round(short_val, 2),
                'total_balance': round(total_balance, 2),
                'short_hour_count': short_hour_count,
                'long_hour_count': long_hour_count,
                'long_realized_pnl': 0,
                'short_realized_pnl': 0
            }
            # Initialize cumulative principal amounts with the first entries
            short_amount_cum = result['entry_short'] * result['short_size']
            long_amount_cum = result['entry_long'] * result['long_size']

            # Calculate rate columns for initial entry based on cumulative principal
            half_balance = total_balance / 2 if total_balance != 0 else 0

            if half_balance > 0:
                result['short_entry_rate'] = round(short_amount_cum / half_balance * 100, 2)
                result['long_entry_rate'] = round(long_amount_cum / half_balance * 100, 2)
            else:
                result['short_entry_rate'] = 0
                result['long_entry_rate'] = 0

            if total_balance > 0:
                result['total_rate'] = round((short_amount_cum + long_amount_cum) / total_balance * 100, 2)
            else:
                result['total_rate'] = 0

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

        # Track if exits occurred (reset for this iteration)
        short_exit_occurred = False
        long_exit_occurred = False
        
        # Calculate total sizes for exit logic
        short_total_size = sum(p['size'] for p in short_positions) if short_positions else 0
        long_total_size = sum(p['size'] for p in long_positions) if long_positions else 0

        # Check if short_profit is greater than or equal to 1 to exit short positions
        if short_profit >= 1 and short_positions:
            # 청산 시 받을 금액(달러): 포지션별 size*current_price*(1+LEVERAGE*수익률)
            principal = sum(p['size'] * p['price'] for p in short_positions)  # 총 진입 원금
            received = sum(p['size'] * current_price * (1 + LEVERAGE * (p['price'] - current_price) / p['price']) for p in short_positions)
            
            # Realized PnL = 정확히 1% 수익에 해당하는 금액 (진입가 기준, LEVERAGE 10배 포함)
            # 1%는 각 포지션의 진입가 * 1% * size * LEVERAGE
            profit_1percent = sum(p['size'] * p['price'] * 0.01 * LEVERAGE for p in short_positions)
            
            fb_before = free_balance
            free_balance += received  # 원금 + 수익 모두 복구 (LEVERAGE 포함)
            print(f"[SHORT EXIT] {timestamp} | principal: {principal:.2f} | received: {received:.2f} | actual_pnl: {profit_1percent:.2f} | fb: {fb_before:.2f}->{free_balance:.2f}")
            current_short_pnl += profit_1percent  # 정확히 1% 수익 (LEVERAGE 포함)
            short_realized_pnl += profit_1percent  # 정확히 1% 수익 (LEVERAGE 포함)
            short_exit_occurred = True
            
            short_positions.clear()  # Clear all short positions
            # Reset cumulative short principal when fully exited
            short_amount_cum = 0

        # Check if long_profit is greater than or equal to 1 to exit long positions
        if long_profit >= 1 and long_positions:
            # 청산 시 받을 금액(달러): 포지션별 size*current_price*(1+LEVERAGE*수익률)
            principal = sum(p['size'] * p['price'] for p in long_positions)  # 총 진입 원금
            received = sum(p['size'] * current_price * (1 + LEVERAGE * (current_price - p['price']) / p['price']) for p in long_positions)
            
            # Realized PnL = 정확히 1% 수익에 해당하는 금액 (진입가 기준, LEVERAGE 10배 포함)
            # 1%는 각 포지션의 진입가 * 1% * size * LEVERAGE
            profit_1percent = sum(p['size'] * p['price'] * 0.01 * LEVERAGE for p in long_positions)
            
            fb_before = free_balance
            free_balance += received  # 원금 + 수익 모두 복구 (LEVERAGE 포함)
            current_long_pnl += profit_1percent  # 정확히 1% 수익 (LEVERAGE 포함)
            long_realized_pnl += profit_1percent  # 정확히 1% 수익 (LEVERAGE 포함)
            long_exit_occurred = True
            
            long_positions.clear()  # Clear all long positions
            # Reset cumulative long principal when fully exited
            long_amount_cum = 0

        # Create result dictionary AFTER exit logic so total_balance reflects updated free_balance
        long_val = sum(p['size'] * current_price * (1 + LEVERAGE * (current_price - p['price']) / p['price']) for p in long_positions) if long_positions else 0
        short_val = sum(p['size'] * current_price * (1 + LEVERAGE * (p['price'] - current_price) / p['price']) for p in short_positions) if short_positions else 0
        total_balance_here = free_balance + long_val + short_val
        
        # Debug: Print exit info if positions were closed
        if (short_profit >= 1 and not short_positions and short_total_size > 0) or (long_profit >= 1 and not long_positions and long_total_size > 0):
            print(f"[EXIT DEBUG] {timestamp} | free_balance: {free_balance:.2f} | long_profit: {long_profit:.2f}% | short_profit: {short_profit:.2f}%")
        result = {
            'Datetime': timestamp,
            'Close': row['Close'],
            'short_avg': short_avg,
            'short_profit': short_profit,
            'entry_short': '',
            'exit_short': current_price if short_exit_occurred else '',
            'short_size': '',
            'short_count': short_count,
            'short_total_size': short_total_size,
            'entry_long': '',
            'exit_long': current_price if long_exit_occurred else '',
            'long_size': '',
            'long_count': long_count,
            'long_total_size': long_total_size,
            'long_avg': long_avg,
            'long_profit': long_profit,
            'free_balance': round(free_balance, 2),
            'long_val': round(long_val, 2),
            'short_val': round(short_val, 2),
            'total_balance': round(total_balance_here, 2),
            'short_hour_count': short_hour_count,
            'long_hour_count': long_hour_count,
            'long_realized_pnl': current_long_pnl,
            'short_realized_pnl': current_short_pnl
        }

        # Entry conditions based on separate short and long counters
        entry_flag = False
        
        # Calculate current total balance for entry size (달러 기준)
        current_total_balance = free_balance + long_val + short_val
        short_entry_base = (current_total_balance * SHORT_SIZE_RATIO) / current_price
        long_entry_base = (current_total_balance * LONG_SIZE_RATIO) / current_price

        # Short entry logic when short_hour_count == short_max_counter
        if short_hour_count == short_max_counter:
            if current_price > short_avg_price:
                entry_size = short_entry_base  # Full size (수량)
                free_balance -= entry_size * current_price
                short_positions.append({'price': current_price, 'size': entry_size, 'unit': 1.0, 'profit_target': None})
                last_entry_price_short = current_price
                result['entry_short'] = current_price
                result['short_size'] = entry_size
                # Add to cumulative short principal
                short_amount_cum += result['entry_short'] * result['short_size']
                entry_flag = True
            elif current_price < short_avg_price:
                entry_size = short_entry_base / 2  # Half size
                free_balance -= entry_size * current_price
                short_positions.append({'price': current_price, 'size': entry_size, 'unit': 0.5, 'profit_target': None})
                last_entry_price_short = current_price
                result['entry_short'] = current_price
                result['short_size'] = entry_size
                # Add to cumulative short principal
                short_amount_cum += result['entry_short'] * result['short_size']
                entry_flag = True

        # Long entry logic when long_hour_count == long_max_counter
        if long_hour_count == long_max_counter:
            if current_price < long_avg_price:
                entry_size = long_entry_base  # Full size (수량)
                free_balance -= entry_size * current_price
                long_positions.append({'price': current_price, 'size': entry_size, 'unit': 1.0, 'profit_target': None})
                last_entry_price_long = current_price
                result['entry_long'] = current_price
                result['long_size'] = entry_size
                # Add to cumulative long principal
                long_amount_cum += result['entry_long'] * result['long_size']
                entry_flag = True
            elif current_price > long_avg_price:
                entry_size = long_entry_base / 2  # Half size
                free_balance -= entry_size * current_price
                long_positions.append({'price': current_price, 'size': entry_size, 'unit': 0.5, 'profit_target': None})
                last_entry_price_long = current_price
                result['entry_long'] = current_price
                result['long_size'] = entry_size
                # Add to cumulative long principal
                long_amount_cum += result['entry_long'] * result['long_size']
                entry_flag = True
        
        # Force entry logic is removed as per new logic
        
        # Update cycle_entry_occurred flag
        if entry_flag:
            cycle_entry_occurred = True
        
        # Recalculate totals and total_balance after potential entries (포지션 = 청산 시 받을 금액)
        short_total_size = sum(p['size'] for p in short_positions) if short_positions else 0
        long_total_size = sum(p['size'] for p in long_positions) if long_positions else 0
        long_val = sum(p['size'] * current_price * (1 + LEVERAGE * (current_price - p['price']) / p['price']) for p in long_positions) if long_positions else 0
        short_val = sum(p['size'] * current_price * (1 + LEVERAGE * (p['price'] - current_price) / p['price']) for p in short_positions) if short_positions else 0
        total_balance = free_balance + long_val + short_val
        result['short_total_size'] = short_total_size
        result['long_total_size'] = long_total_size
        result['free_balance'] = round(free_balance, 2)
        result['long_val'] = round(long_val, 2)
        result['short_val'] = round(short_val, 2)
        result['total_balance'] = round(total_balance, 2)

        # Calculate rate columns based on updated totals and cumulative principal amounts
        # entry_rate는 현재 포지션의 누적 투입금을 기반으로 계산
        half_balance = total_balance / 2 if total_balance != 0 else 0

        # 현재 포지션의 투입금액 기반으로 계산
        long_current_invested = sum(p['size'] * p['price'] for p in long_positions) if long_positions else 0
        short_current_invested = sum(p['size'] * p['price'] for p in short_positions) if short_positions else 0
        
        # 누적 투입금을 사용 (exit 후에는 0으로 초기화되어 새로운 entry부터 계산)
        short_amount = short_amount_cum
        long_amount = long_amount_cum
        
        # Debug: Print entry info if new positions were opened
        if 'entry_long' in result and result['entry_long'] != '':
            print(f"[ENTRY DEBUG] {timestamp} | Entry Long at {result['entry_long']} | Size: {result['long_size']:.4f} | long_amount_cum: {long_amount_cum:.2f}")
        if 'entry_short' in result and result['entry_short'] != '':
            print(f"[ENTRY DEBUG] {timestamp} | Entry Short at {result['entry_short']} | Size: {result['short_size']:.4f} | short_amount_cum: {short_amount_cum:.2f}")

        if half_balance > 0:
            result['short_entry_rate'] = round(short_amount / half_balance * 100, 2)
            result['long_entry_rate'] = round(long_amount / half_balance * 100, 2)
        else:
            result['short_entry_rate'] = 0
            result['long_entry_rate'] = 0

        if total_balance > 0:
            result['total_rate'] = round((short_amount + long_amount) / total_balance * 100, 2)
        else:
            result['total_rate'] = 0
        
        # Append result rows
        result_rows.append(result.copy())
        if result['entry_long'] or result['entry_short'] or result['exit_long'] or result['exit_short']:
            event_rows.append(result.copy())
        
        # Increment counters separately
        short_hour_count += 1
        if short_hour_count > short_max_counter:
            short_hour_count = 1
        
        long_hour_count += 1
        if long_hour_count > long_max_counter:
            long_hour_count = 1

    print("Simulated trading completed.")
    # Return both DataFrames: all timestamps and only events
    return pd.DataFrame(result_rows), pd.DataFrame(event_rows)

# Main function for backtesting
if __name__ == "__main__":
    try:
        qqq_data = fetch_qqq_data()
        
        # Optimization: Test SIZE_RATIO variations with fixed short_max=20, long_max=1
        print("\n" + "="*80)
        print("Testing SIZE_RATIO optimization (short=20, long=1)")
        print("SIZE_RATIO: 1/160 ~ 1/320 (10단위) = 16 values")
        print("Total combinations: 16 * 16 = 256")
        print("="*80)
        
        # Generate size_ratios: 1/160, 1/170, ..., 1/320
        size_ratio_denominators = list(range(160, 330, 10))  # 160, 170, ..., 320
        size_ratios = [1.0 / d for d in size_ratio_denominators]
        print(f"Size ratios: {[f'1/{d}' for d in size_ratio_denominators]}")
        
        short_counter = 20  # Fixed
        long_counter = 1    # Fixed
        
        results_matrix = {}  # Store results: (short_ratio_idx, long_ratio_idx) -> final_balance
        combination_count = 0
        
        for short_ratio_idx, short_ratio in enumerate(size_ratios):
            for long_ratio_idx, long_ratio in enumerate(size_ratios):
                combination_count += 1
                denom_short = size_ratio_denominators[short_ratio_idx]
                denom_long = size_ratio_denominators[long_ratio_idx]
                print(f"[{combination_count}/256] Testing: SHORT_RATIO=1/{denom_short}, LONG_RATIO=1/{denom_long}...", end=" ", flush=True)
                
                all_results, event_results = simulated_trade(
                    qqq_data, 
                    short_max_counter=short_counter, 
                    long_max_counter=long_counter,
                    short_size_ratio=short_ratio,
                    long_size_ratio=long_ratio
                )

                # Filter condition: Drop if total_rate > 100
                dropped = False
                
                # Check total_rate
                total_rate_series = pd.to_numeric(all_results['total_rate'], errors='coerce') if 'total_rate' in all_results.columns else None
                
                if total_rate_series is not None:
                    # Get the last total_rate value
                    last_total_rate = total_rate_series.iloc[-1]
                    if last_total_rate > 100:
                        print(f"DROPPED: total_rate={last_total_rate:.2f} > 100")
                        dropped = True
                
                # If drop condition triggered, skip this combination
                if dropped:
                    continue

                # Get final total_balance for valid combinations
                final_balance = all_results['total_balance'].iloc[-1]
                results_matrix[(short_ratio_idx, long_ratio_idx)] = final_balance
                print(f" Final Balance: {final_balance:.2f}")
        
        print("\n" + "="*80)
        print("Optimization complete. Finding best combination...")
        print("="*80)
        
        if not results_matrix:
            raise ValueError("All 256 combinations were dropped by constraints. No valid parameter set found.")

        # Create matrix for heatmap using only valid (non-dropped) combinations
        matrix = np.zeros((len(size_ratios), len(size_ratios)))
        for (short_idx, long_idx), balance in results_matrix.items():
            matrix[short_idx, long_idx] = balance
        
        # Mark non-tested combinations as -inf for visualization
        for i in range(len(size_ratios)):
            for j in range(len(size_ratios)):
                if (i, j) not in results_matrix:
                    matrix[i, j] = np.nan
        
        # Find best combination (ignore NaN by using nanargmax)
        best_idx = np.unravel_index(np.nanargmax(matrix), matrix.shape)
        best_short_idx = best_idx[0]
        best_long_idx = best_idx[1]
        best_short_ratio = size_ratios[best_short_idx]
        best_long_ratio = size_ratios[best_long_idx]
        best_short_denom = size_ratio_denominators[best_short_idx]
        best_long_denom = size_ratio_denominators[best_long_idx]
        best_balance = matrix[best_idx[0], best_idx[1]]
        
        print(f"\n✓ BEST COMBINATION FOUND!")
        print(f"  Short Count Max: {short_counter}")
        print(f"  Long Count Max: {long_counter}")
        print(f"  Short Size Ratio: 1/{best_short_denom} ({best_short_ratio:.6f})")
        print(f"  Long Size Ratio: 1/{best_long_denom} ({best_long_ratio:.6f})")
        print(f"  Final Balance: {best_balance:.2f}")
        print("="*80)
        
        # Run final simulation with best combination and save results
        print(f"\nGenerating final results with optimal parameters...")
        all_results, event_results = simulated_trade(
            qqq_data, 
            short_max_counter=short_counter, 
            long_max_counter=long_counter,
            short_size_ratio=best_short_ratio,
            long_size_ratio=best_long_ratio
        )
        
        # Set column order
        col_order = [
            'Datetime','Close','entry_short','exit_short','short_size','short_count','short_total_size','short_avg',
            'short_profit','entry_long','exit_long','long_size','long_count','long_total_size','long_avg','long_profit',
            'free_balance','long_val','short_val','total_balance','short_hour_count','long_hour_count','long_realized_pnl','short_realized_pnl',
            'short_entry_rate','long_entry_rate','total_rate'
        ]
        all_results = all_results[col_order]
        all_results = all_results.drop_duplicates(subset=['Datetime'], keep='first')
        event_results = event_results[col_order]
        event_results = event_results.drop_duplicates(subset=['Datetime'], keep='first')
        
        # Save optimal results with naming convention (showing short/long counter + ratio denominators)
        optimal_all_csv = f"qqq_backtest_optimal_short{short_counter}_long{long_counter}_ratio{best_short_denom}_{best_long_denom}_all.csv"
        optimal_event_csv = f"qqq_backtest_optimal_short{short_counter}_long{long_counter}_ratio{best_short_denom}_{best_long_denom}_results.csv"
        all_results.to_csv(optimal_all_csv, index=False)
        event_results.to_csv(optimal_event_csv, index=False)
        print(f"Saved optimal results: {optimal_all_csv} and {optimal_event_csv}")

        # Upload optimal ALL CSV to Google Drive with timestamp
        from datetime import datetime
        now = datetime.now().strftime("_%y%m%d_%H%M%S")
        remote_filename = f"qqq_backtest_optimal_short{short_counter}_long{long_counter}_ratio{best_short_denom}_{best_long_denom}_all{now}.csv"
        rclone_cmd = [
            "rclone", "copyto",
            optimal_all_csv,
            f"BackTest:{remote_filename}"
        ]
        try:
            subprocess.run(rclone_cmd, check=True)
            print(f"Uploaded to Google Drive as {remote_filename}")
        except Exception as e:
            print(f"rclone upload failed: {e}")
        
        # Create heatmap
        plt.figure(figsize=(16, 14))
        plt.imshow(matrix, cmap='RdYlGn', aspect='auto', origin='lower')
        plt.colorbar(label='Final Total Balance')
        plt.xlabel('Long Size Ratio Denominator')
        plt.ylabel('Short Size Ratio Denominator')
        ratio_labels = [f'1/{d}' for d in size_ratio_denominators]
        plt.title(f'SIZE_RATIO Optimization: Final Balance\nShort_Count={short_counter}, Long_Count={long_counter}\nBest: 1/{best_short_denom} / 1/{best_long_denom}, Balance={best_balance:.2f}')
        plt.xticks(range(len(size_ratio_denominators)), ratio_labels, rotation=45)
        plt.yticks(range(len(size_ratio_denominators)), ratio_labels)
        
        # Mark best point
        plt.plot(best_idx[1], best_idx[0], 'b*', markersize=30, label=f'Best: (1/{best_short_denom}, 1/{best_long_denom})')
        plt.legend(fontsize=12)
        
        # Add value annotations (only for tested combinations)
        for (short_idx, long_idx), balance in results_matrix.items():
            plt.text(long_idx, short_idx, f'{balance:.0f}', ha='center', va='center', fontsize=8, color='black', alpha=0.6)
        
        plt.tight_layout()
        heatmap_file = f'optimization_heatmap_sizeratios_s{short_counter}_l{long_counter}.png'
        plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
        print(f"\nHeatmap saved to: {heatmap_file}")
        plt.show()
        
        print("\nBacktest completed. Results saved:")
        print(f"  - {optimal_all_csv} (all timestamps)")
        print(f"  - {optimal_event_csv} (events only)")
        print(f"  - {heatmap_file} (256-combination visualization)")

        # =====================
        # 성과 지표
        # =====================
        equity_curve = all_results['total_balance'].tolist()
        # all_results에서 total_balance를 사용합니다

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