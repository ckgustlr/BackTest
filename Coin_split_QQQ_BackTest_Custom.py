import yfinance as yf
import pandas as pd
import datetime
import time

# Hyperparameters
LEVERAGE = 10
ENTRY_GAP = 0.003  # 0.3%
SIZE_RATIO = 0.01  # 1%
LONG_PROFIT_NARROW = [0.01, 0.02]  # 1% to 2%
LONG_PROFIT_WIDE = [0.01, 0.03]  # 1% to 3%
SHORT_PROFIT_NARROW = [0.01, 0.02]
SHORT_PROFIT_WIDE = [0.01, 0.03]
RISK_THRESHOLD = 0.02  # 2%

# Fetch QQQ 1-minute data for the last 30 days
def fetch_qqq_data():
    csv_file = "qqq_1min_data.csv"
    try:
        # If CSV exists, load and return
        # Read CSV and skip rows that cannot be parsed as datetime
        raw_data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        # Filter out rows where index is not a valid datetime
        valid_idx = pd.to_datetime(raw_data.index, errors='coerce')
        data = raw_data[~valid_idx.isna()].copy()
        data.index = pd.to_datetime(data.index)
        # Remove rows where 'Close' is not numeric
        data = data[pd.to_numeric(data['Close'], errors='coerce').notna()]
        print(f"Loaded QQQ data from {csv_file} (valid rows only)")
        return data
    except FileNotFoundError:
        print("CSV not found. Fetching QQQ 1-minute data in 8-day batches...")
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=30)
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
                interval="1m"
            )
            if not batch_data.empty:
                all_data.append(batch_data)
            batch_start = batch_end
            time.sleep(2)  # Avoid rate limits
        if all_data:
            data = pd.concat(all_data)
            # Save with index and header for proper loading
            data.to_csv(csv_file, index=True, header=True)
            print(f"QQQ data saved to {csv_file}")
            return data
        else:
            raise ValueError("No data fetched. Please check the ticker or date range.")

# Simulated trading function for backtesting
def simulated_trade(data):
    print("Starting simulated trading...")

    # Initialize variables
    free_balance = 10000  # Example starting balance
    long_positions = []
    short_positions = []
    last_entry_price_long = None
    last_entry_price_short = None
    equity_curve = []  # Track free_balance history

    # Remove rows where 'Close' is not numeric and convert to float
    data = data[pd.to_numeric(data['Close'], errors='coerce').notna()].copy()
    data['Close'] = data['Close'].astype(float)
    # Filter data for trading hours (Monday 10:00 to Saturday 10:00)
    data = data[(data.index.dayofweek < 5) & ((data.index.dayofweek != 5) | (data.index.hour < 10))]

    # Prepare result rows
    result_rows = []
    event_rows = []
    realized_pnl = 0
    # 루프 내에서 사용할 변수 초기화
    long_avg_price = 0
    short_avg_price = 0
    for timestamp, row in data.iterrows():
        current_price = row['Close']
        # Prepare all columns in requested order, default blank
        result = {
            'Datetime': timestamp,
            'Close': row['Close'],
            'last_entry_price_short': last_entry_price_short if last_entry_price_short is not None else '',
            'entry_short': '',
            'exit_short': '',
            'short_size': '',
            'short_count': len(short_positions),
            'short_total_size': sum(p['size'] for p in short_positions) if short_positions else '',
            'short_avg': sum(p['price'] * p['size'] for p in short_positions) / sum(p['size'] for p in short_positions) if short_positions else '',
            'last_entry_price_long': last_entry_price_long if last_entry_price_long is not None else '',
            'entry_long': '',
            'exit_long': '',
            'long_size': '',
            'long_count': len(long_positions),
            'long_total_size': sum(p['size'] for p in long_positions) if long_positions else '',
            'long_avg': sum(p['price'] * p['size'] for p in long_positions) / sum(p['size'] for p in long_positions) if long_positions else '',
            'realized_pnl': '',  # Will be set below with leverage
            'long_short_gap': '',
            'total_balance': 0  # Will be set below after realized_pnl
        }
        # long_short_gap 계산 (평단 갭이 아니라 실제 long_short_gap 공식 적용)
        if long_avg_price > 0 and short_avg_price > 0:
            try:
                result['long_short_gap'] = abs(long_avg_price - short_avg_price) / ((long_avg_price + short_avg_price) / 2)
            except Exception:
                result['long_short_gap'] = ''
        else:
            result['long_short_gap'] = ''

        # Entry conditions
        entry_flag = False
        if last_entry_price_long is None or current_price <= last_entry_price_long * (1 - ENTRY_GAP):
            entry_size = free_balance * SIZE_RATIO
            free_balance -= entry_size
            long_positions.append({'price': current_price, 'size': entry_size, 'profit_target': None})
            last_entry_price_long = current_price
            result['entry_long'] = current_price
            result['long_size'] = entry_size
            entry_flag = True

        if last_entry_price_short is None or current_price >= last_entry_price_short * (1 + ENTRY_GAP):
            entry_size = free_balance * SIZE_RATIO
            free_balance -= entry_size
            short_positions.append({'price': current_price, 'size': entry_size, 'profit_target': None})
            last_entry_price_short = current_price
            result['entry_short'] = current_price
            result['short_size'] = entry_size
            entry_flag = True

        # Exit conditions
        long_avg_price = sum(p['price'] * p['size'] for p in long_positions) / sum(p['size'] for p in long_positions) if long_positions else 0
        short_avg_price = sum(p['price'] * p['size'] for p in short_positions) / sum(p['size'] for p in short_positions) if short_positions else 0

        exit_flag_long = False
        exit_flag_short = False
        exit_long_pnl = 0
        exit_short_pnl = 0

        if long_positions and current_price >= long_avg_price * (1 + LONG_PROFIT_NARROW[0]):
            exit_flag_long = True
            exit_long_pnl = sum(p['size'] for p in long_positions) * (1 + LONG_PROFIT_NARROW[0])
            free_balance += exit_long_pnl
            realized_pnl += exit_long_pnl - sum(p['size'] for p in long_positions)
            result['exit_long'] = current_price
            result['long_size'] = sum(p['size'] for p in long_positions)
            result['long_total_size'] = 0
            # Apply leverage to realized_pnl for this event
            result['realized_pnl'] = (exit_long_pnl - sum(p['size'] for p in long_positions)) * LEVERAGE
            long_positions = []

        if short_positions and current_price <= short_avg_price * (1 - SHORT_PROFIT_NARROW[0]):
            exit_flag_short = True
            exit_short_pnl = sum(p['size'] for p in short_positions) * (1 + SHORT_PROFIT_NARROW[0])
            free_balance += exit_short_pnl
            realized_pnl += exit_short_pnl - sum(p['size'] for p in short_positions)
            result['exit_short'] = current_price
            result['short_size'] = sum(p['size'] for p in short_positions)
            result['short_total_size'] = 0
            # Apply leverage to realized_pnl for this event
            result['realized_pnl'] = (exit_short_pnl - sum(p['size'] for p in short_positions)) * LEVERAGE
            short_positions = []

        # Risk management
        if long_positions and short_positions:
            long_short_gap = abs(long_avg_price - short_avg_price) / ((long_avg_price + short_avg_price) / 2)
        else:
            long_short_gap = 0
        if long_short_gap <= RISK_THRESHOLD:
            is_normal_arrangement = True  # 정배열: True, 역배열: False
            LONG_PROFIT = LONG_PROFIT_NARROW
            SHORT_PROFIT = SHORT_PROFIT_NARROW
        else:
            is_normal_arrangement = False  # Only 정배열 for wide zones
            LONG_PROFIT = LONG_PROFIT_WIDE
            SHORT_PROFIT = SHORT_PROFIT_WIDE

        if long_positions:
            n = len(long_positions)
            for idx in range(n):
                if is_normal_arrangement:
                    profit = LONG_PROFIT[0] + (LONG_PROFIT[1] - LONG_PROFIT[0]) * idx / (n-1) if n > 1 else LONG_PROFIT[0]
                else:
                    profit = LONG_PROFIT[0] + (LONG_PROFIT[1] - LONG_PROFIT[0]) * (n-1-idx) / (n-1) if n > 1 else LONG_PROFIT[0]
                long_positions[idx]['profit_target'] = profit
            realized = []
            for pos in long_positions:
                if current_price >= pos['price'] * (1 + pos['profit_target']):
                    free_balance += pos['size'] * (1 + pos['profit_target'])
                    realized_pnl += pos['size'] * pos['profit_target']
                    realized.append(pos)
            long_positions = [p for p in long_positions if p not in realized]

        if short_positions:
            n = len(short_positions)
            for idx in range(n):
                if is_normal_arrangement:
                    profit = SHORT_PROFIT[0] + (SHORT_PROFIT[1] - SHORT_PROFIT[0]) * idx / (n-1) if n > 1 else SHORT_PROFIT[0]
                else:
                    profit = SHORT_PROFIT[0] + (SHORT_PROFIT[1] - SHORT_PROFIT[0]) * (n-1-idx) / (n-1) if n > 1 else SHORT_PROFIT[0]
                short_positions[idx]['profit_target'] = profit
            realized = []
            for pos in short_positions:
                if current_price <= pos['price'] * (1 - pos['profit_target']):
                    free_balance += pos['size'] * (1 + pos['profit_target'])
                    realized_pnl += pos['size'] * pos['profit_target']
                    realized.append(pos)
            short_positions = [p for p in short_positions if p not in realized]

        # Track free_balance for performance metrics
        equity_curve.append(free_balance)
        # realized_pnl은 exit_long 또는 exit_short가 발생한 row에만 기록, 그 외에는 ''
        if not (result['exit_long'] or result['exit_short']):
            result['realized_pnl'] = ''
        # total_balance도 realized_pnl이 있는 경우만 반영, 그 외에는 realized_pnl 없이 계산
        try:
            realized_pnl_val = float(result['realized_pnl']) if result['realized_pnl'] != '' else 0
        except Exception:
            realized_pnl_val = 0
        result['total_balance'] = free_balance + sum(p['size'] for p in long_positions) + sum(p['size'] for p in short_positions) + realized_pnl_val
        # Append row for all timestamps
        result_rows.append(result.copy())
        # Append row only if entry/exit event occurred (for event CSV)
        if result['entry_long'] or result['entry_short'] or result['exit_long'] or result['exit_short']:
            event_rows.append(result.copy())
        # long_short_gap 계산 (평단 갭이 아니라 실제 long_short_gap 공식 적용)
        if long_avg_price > 0 and short_avg_price > 0:
            try:
                result['long_short_gap'] = abs(long_avg_price - short_avg_price) / ((long_avg_price + short_avg_price) / 2)
            except Exception:
                result['long_short_gap'] = 'N/A'
        else:
            result['long_short_gap'] = 'N/A'

        # Entry conditions
        entry_flag = False
        if last_entry_price_long is None or current_price <= last_entry_price_long * (1 - ENTRY_GAP):
            entry_size = free_balance * SIZE_RATIO
            free_balance -= entry_size
            long_positions.append({'price': current_price, 'size': entry_size, 'profit_target': None})
            last_entry_price_long = current_price
            result['entry_long'] = current_price
            result['long_size'] = entry_size
            entry_flag = True

        if last_entry_price_short is None or current_price >= last_entry_price_short * (1 + ENTRY_GAP):
            entry_size = free_balance * SIZE_RATIO
            free_balance -= entry_size
            short_positions.append({'price': current_price, 'size': entry_size, 'profit_target': None})
            last_entry_price_short = current_price
            result['entry_short'] = current_price
            result['short_size'] = entry_size
            entry_flag = True

        # Exit conditions
        long_avg_price = sum(p['price'] * p['size'] for p in long_positions) / sum(p['size'] for p in long_positions) if long_positions else 0
        short_avg_price = sum(p['price'] * p['size'] for p in short_positions) / sum(p['size'] for p in short_positions) if short_positions else 0

        exit_flag_long = False
        exit_flag_short = False
        exit_long_pnl = 0
        exit_short_pnl = 0

        if long_positions and current_price >= long_avg_price * (1 + LONG_PROFIT_NARROW[0]):
            exit_flag_long = True
            exit_long_pnl = sum(p['size'] for p in long_positions) * (1 + LONG_PROFIT_NARROW[0])
            free_balance += exit_long_pnl
            realized_pnl += exit_long_pnl - sum(p['size'] for p in long_positions)
            result['exit_long'] = current_price
            result['long_size'] = sum(p['size'] for p in long_positions)
            result['long_total_size'] = 0
            result['realized_pnl'] = exit_long_pnl - sum(p['size'] for p in long_positions)
            long_positions = []

        if short_positions and current_price <= short_avg_price * (1 - SHORT_PROFIT_NARROW[0]):
            exit_flag_short = True
            exit_short_pnl = sum(p['size'] for p in short_positions) * (1 + SHORT_PROFIT_NARROW[0])
            free_balance += exit_short_pnl
            realized_pnl += exit_short_pnl - sum(p['size'] for p in short_positions)
            result['exit_short'] = current_price
            result['short_size'] = sum(p['size'] for p in short_positions)
            result['short_total_size'] = 0
            result['realized_pnl'] = exit_short_pnl - sum(p['size'] for p in short_positions)
            short_positions = []

        # Risk management
        if long_positions and short_positions:
            long_short_gap = abs(long_avg_price - short_avg_price) / ((long_avg_price + short_avg_price) / 2)
        else:
            long_short_gap = 0
        if long_short_gap <= RISK_THRESHOLD:
            is_normal_arrangement = True  # 정배열: True, 역배열: False
            LONG_PROFIT = LONG_PROFIT_NARROW
            SHORT_PROFIT = SHORT_PROFIT_NARROW
        else:
            is_normal_arrangement = False  # Only 정배열 for wide zones
            LONG_PROFIT = LONG_PROFIT_WIDE
            SHORT_PROFIT = SHORT_PROFIT_WIDE

        if long_positions:
            n = len(long_positions)
            for idx in range(n):
                if is_normal_arrangement:
                    profit = LONG_PROFIT[0] + (LONG_PROFIT[1] - LONG_PROFIT[0]) * idx / (n-1) if n > 1 else LONG_PROFIT[0]
                else:
                    profit = LONG_PROFIT[0] + (LONG_PROFIT[1] - LONG_PROFIT[0]) * (n-1-idx) / (n-1) if n > 1 else LONG_PROFIT[0]
                long_positions[idx]['profit_target'] = profit
            realized = []
            for pos in long_positions:
                if current_price >= pos['price'] * (1 + pos['profit_target']):
                    free_balance += pos['size'] * (1 + pos['profit_target'])
                    realized_pnl += pos['size'] * pos['profit_target']
                    realized.append(pos)
            long_positions = [p for p in long_positions if p not in realized]

        if short_positions:
            n = len(short_positions)
            for idx in range(n):
                if is_normal_arrangement:
                    profit = SHORT_PROFIT[0] + (SHORT_PROFIT[1] - SHORT_PROFIT[0]) * idx / (n-1) if n > 1 else SHORT_PROFIT[0]
                else:
                    profit = SHORT_PROFIT[0] + (SHORT_PROFIT[1] - SHORT_PROFIT[0]) * (n-1-idx) / (n-1) if n > 1 else SHORT_PROFIT[0]
                short_positions[idx]['profit_target'] = profit
            realized = []
            for pos in short_positions:
                if current_price <= pos['price'] * (1 - pos['profit_target']):
                    free_balance += pos['size'] * (1 + pos['profit_target'])
                    realized_pnl += pos['size'] * pos['profit_target']
                    realized.append(pos)
            short_positions = [p for p in short_positions if p not in realized]

        # Track free_balance for performance metrics
        equity_curve.append(free_balance)
        # Update total_balance after all changes
        result['total_balance'] = free_balance + sum(p['size'] for p in long_positions) + sum(p['size'] for p in short_positions)
        # Append row only if entry/exit event occurred
        if result['entry_long'] or result['entry_short'] or result['exit_long'] or result['exit_short']:
            result_rows.append(result)

    print("Simulated trading completed.")
    # Return both DataFrames: all timestamps and only events
    return pd.DataFrame(result_rows), pd.DataFrame(event_rows)

    print("Simulated trading completed.")
    # Return both data and equity_curve for performance metrics
    return pd.DataFrame({'free_balance': equity_curve})

# Main function for backtesting
if __name__ == "__main__":
    try:
        qqq_data = fetch_qqq_data()
        all_results, event_results = simulated_trade(qqq_data)
        # Set column order
        col_order = [
            'Datetime','Close','last_entry_price_short','entry_short','exit_short','short_size','short_count','short_total_size','short_avg',
            'last_entry_price_long','entry_long','exit_long','long_size','long_count','long_total_size','long_avg',
            'realized_pnl','long_short_gap','total_balance'
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