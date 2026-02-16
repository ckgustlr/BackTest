import yfinance as yf
import pandas as pd
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys 

# Jetson 등에서 빠른 확인용: True면 1케이스만 실행 (S=12, L=1, ratio 1/80)
#SINGLE_TEST_MODE = os.environ.get("QQQ_SINGLE_TEST", "").lower() in ("1", "true", "yes")
SINGLE_TEST_MODE = False


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
def simulated_trade(data, short_max_counter=3, long_max_counter=3, short_size_ratio=1/320, long_size_ratio=1/320, verbose=True):
    if verbose:
        print(f"Starting simulated trading with short_max_counter={short_max_counter}, long_max_counter={long_max_counter}, short_ratio={short_size_ratio:.2f}, long_ratio={long_size_ratio:.2f}...")
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
            long_positions.append({'price': current_price, 'size': long_entry_size, 'unit': 1.0, 'profit_target': None, 'target_price': round(current_price * 1.01, 2)})
            short_positions.append({'price': current_price, 'size': short_entry_size, 'unit': 1.0, 'profit_target': None, 'target_price': round(current_price * 0.99, 2)})
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
                'Close': round(float(row['Close']), 2),
                'short_avg': round(short_avg, 2),
                'short_profit': short_profit,
                'entry_short': round(current_price, 2),
                'exit_short': '',
                'short_size': round(short_entry_size, 2),
                'short_count': round(1.0, 2),
                'short_total_size': round(short_total_size, 2),
                'entry_long': round(current_price, 2),
                'exit_long': '',
                'long_size': round(long_entry_size, 2),
                'long_count': round(1.0, 2),
                'long_total_size': round(long_total_size, 2),
                'long_avg': '',
                'long_profit': '',
                'free_balance': round(free_balance, 2),
                'long_val': round(long_val, 2),
                'short_val': round(short_val, 2),
                'total_balance': round(total_balance, 2),
                'short_hour_count': short_hour_count,
                'long_hour_count': long_hour_count,
                'long_realized_pnl': round(0, 2),
                'short_realized_pnl': round(0, 2)
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

        # 2. 분할 매도 조건 (각 투입 건별 1% 수익 시)
        exit_short_parts = []
        remaining_short = []
        for p in short_positions:
            if current_price <= p['target_price']:
                received = p['size'] * current_price * (1 + LEVERAGE * (p['price'] - current_price) / p['price'])
                profit_1p = p['size'] * p['price'] * 0.01 * LEVERAGE
                free_balance += received
                current_short_pnl += profit_1p
                short_realized_pnl += profit_1p
                short_amount_cum = max(0, short_amount_cum - p['size'] * p['price'])
                exit_short_parts.append(f"{p['price']:.2f}->{current_price:.2f}({p['size']:.4f})")
                # print(f"분할 매도(숏) 실현: {current_price} (진입가: {p['price']}, size: {p['size']})")
            else:
                remaining_short.append(p)
        short_positions = remaining_short
        if exit_short_parts:
            short_exit_occurred = True

        exit_long_parts = []
        remaining_long = []
        for p in long_positions:
            if current_price >= p['target_price']:
                received = p['size'] * current_price * (1 + LEVERAGE * (current_price - p['price']) / p['price'])
                profit_1p = p['size'] * p['price'] * 0.01 * LEVERAGE
                free_balance += received
                current_long_pnl += profit_1p
                long_realized_pnl += profit_1p
                long_amount_cum = max(0, long_amount_cum - p['size'] * p['price'])
                exit_long_parts.append(f"{p['price']:.2f}->{current_price:.2f}({p['size']:.4f})")
                # print(f"분할 매도(롱) 실현: {current_price} (진입가: {p['price']}, size: {p['size']})")
            else:
                remaining_long.append(p)
        long_positions = remaining_long
        if exit_long_parts:
            long_exit_occurred = True

        # Create result dictionary AFTER exit logic (잔여 포지션 기준 avg/profit 재계산)
        short_avg_after = sum(p['price'] * p['size'] for p in short_positions) / sum(p['size'] for p in short_positions) if short_positions else 0
        long_avg_after = sum(p['price'] * p['size'] for p in long_positions) / sum(p['size'] for p in long_positions) if long_positions else 0
        short_profit_after = round((short_avg_after - current_price) / current_price * 100, 2) if current_price != 0 else 0
        long_profit_after = round((current_price - long_avg_after) / long_avg_after * 100, 2) if long_avg_after != 0 else 0
        long_val = sum(p['size'] * current_price * (1 + LEVERAGE * (current_price - p['price']) / p['price']) for p in long_positions) if long_positions else 0
        short_val = sum(p['size'] * current_price * (1 + LEVERAGE * (p['price'] - current_price) / p['price']) for p in short_positions) if short_positions else 0
        total_balance_here = free_balance + long_val + short_val
        
        result = {
            'Datetime': timestamp,
            'Close': round(float(row['Close']), 2),
            'short_avg': round(short_avg_after, 2),
            'short_profit': short_profit_after,
            'entry_short': '',
            'exit_short': ','.join(exit_short_parts) if exit_short_parts else '',
            'short_size': '',
            'short_count': float(len(short_positions)),
            'short_total_size': round(sum(p['size'] for p in short_positions) if short_positions else 0, 2),
            'entry_long': '',
            'exit_long': ','.join(exit_long_parts) if exit_long_parts else '',
            'long_size': '',
            'long_count': float(len(long_positions)),
            'long_total_size': round(sum(p['size'] for p in long_positions) if long_positions else 0, 2),
            'long_avg': round(long_avg_after, 2),
            'long_profit': long_profit_after,
            'free_balance': round(free_balance, 2),
            'long_val': round(long_val, 2),
            'short_val': round(short_val, 2),
            'total_balance': round(total_balance_here, 2),
            'short_hour_count': short_hour_count,
            'long_hour_count': long_hour_count,
            'long_realized_pnl': round(current_long_pnl, 2),
            'short_realized_pnl': round(current_short_pnl, 2)
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
                short_positions.append({'price': current_price, 'size': entry_size, 'unit': 1.0, 'profit_target': None, 'target_price': round(current_price * 0.99, 2)})
                last_entry_price_short = current_price
                result['entry_short'] = current_price
                result['short_size'] = entry_size
                # Add to cumulative short principal
                short_amount_cum += result['entry_short'] * result['short_size']
                entry_flag = True
            elif current_price < short_avg_price:
                entry_size = short_entry_base / 2  # Half size
                free_balance -= entry_size * current_price
                short_positions.append({'price': current_price, 'size': entry_size, 'unit': 0.5, 'profit_target': None, 'target_price': round(current_price * 0.99, 2)})
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
                long_positions.append({'price': current_price, 'size': entry_size, 'unit': 1.0, 'profit_target': None, 'target_price': round(current_price * 1.01, 4)})
                last_entry_price_long = current_price
                result['entry_long'] = current_price
                result['long_size'] = entry_size
                # Add to cumulative long principal
                long_amount_cum += result['entry_long'] * result['long_size']
                entry_flag = True
            elif current_price > long_avg_price:
                entry_size = long_entry_base / 2  # Half size
                free_balance -= entry_size * current_price
                long_positions.append({'price': current_price, 'size': entry_size, 'unit': 0.5, 'profit_target': None, 'target_price': round(current_price * 1.01, 4)})
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
        # if 'entry_long' in result and result['entry_long'] != '':
        #     print(f"[ENTRY DEBUG] {timestamp} | Entry Long at {result['entry_long']} | Size: {result['long_size']:.4f} | long_amount_cum: {long_amount_cum:.2f}")
        # if 'entry_short' in result and result['entry_short'] != '':
        #     print(f"[ENTRY DEBUG] {timestamp} | Entry Short at {result['entry_short']} | Size: {result['short_size']:.4f} | short_amount_cum: {short_amount_cum:.2f}")

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
        # ✓ DROP 조건: short_profit < -10 OR long_profit < -10
        short_p = result.get('short_profit', 0)
        long_p = result.get('long_profit', 0)
        
        # Append result rows
        # ✓ DROP 조건: short_profit < -10 OR long_profit < -10 (단, -100 제외)
        short_p = result.get('short_profit', 0)
        long_p = result.get('long_profit', 0)
        
        if (short_p < -10 and short_p != -100) or (long_p < -10 and long_p != -100):
            if verbose:
                print(f"  ! DROPPED at {timestamp}: short_profit={short_p} < -10 OR long_profit={long_p} < -10 (excluding -100)")
            return pd.DataFrame(), pd.DataFrame()
        
        result_rows.append(result.copy())

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

    if verbose:
        print("Simulated trading completed.")
    # ✓ 드랍 조건 검사 (DataFrame 생성 前)
    if result_rows:
        last_result = result_rows[-1]
        if last_result.get('total_rate', 0) > 100:
            if verbose:
                print(f"  ! DROPPED: total_rate={last_result.get('total_rate')} > 100")
            # 빈 DataFrame 반환 (또는 None)
            return pd.DataFrame(), pd.DataFrame()
    
    # Return both DataFrames: all timestamps and only events
    return pd.DataFrame(result_rows), pd.DataFrame(event_rows)

def _run_one_case(tup):
    """멀티프로세싱 워커: (data, short_counter, long_counter, short_denom, long_denom) → ((sc,lc,sd,ld), balance or None)"""
    data, short_counter, long_counter, short_denom, long_denom = tup
    short_ratio = 1.0 / short_denom
    long_ratio = 1.0 / long_denom
    try:
        all_results, _ = simulated_trade(
            data,
            short_max_counter=short_counter,
            long_max_counter=long_counter,
            short_size_ratio=short_ratio,
            long_size_ratio=long_ratio,
            verbose=False,
        )
        total_rate_series = pd.to_numeric(all_results["total_rate"], errors="coerce")
        if total_rate_series is not None and not total_rate_series.empty:
            last_total_rate = total_rate_series.iloc[-1]
            if last_total_rate > 100:
                return ((short_counter, long_counter, short_denom, long_denom), None)
        final_balance = all_results["total_balance"].iloc[-1]
        return ((short_counter, long_counter, short_denom, long_denom), final_balance)
    except Exception:
        return ((short_counter, long_counter, short_denom, long_denom), None)


# Main function for backtesting
if __name__ == "__main__":
    try:
        qqq_data = fetch_qqq_data()
        
        if SINGLE_TEST_MODE:
            # Jetson 등: 단일 케이스만 실행 (빠른 확인용). 환경변수 QQQ_SINGLE_TEST=1 로 활성화
            print("\n" + "="*80)
            print("SINGLE TEST MODE (QQQ_SINGLE_TEST=1): 1 case only — S=18, L=1, ratio 1/110 1/90")
            print("="*80)
            best_short_counter, best_long_counter = 18, 1
            best_short_denom, best_long_denom = 110, 90
        else:
            # 1~20 x 1~20 시간, 비율 1/50~1/110 (10단위 7개) 각각 → 20*20*7*7 = 19600 케이스, total_rate>100 드랍
            # CPU 멀티프로세싱으로 병렬 실행
            print("\n" + "="*80)
            print("Testing: SHORT_TIME=1~20, LONG_TIME=1~20, SHORT_RATIO=1/300~1/360(7), LONG_RATIO=1/300~1/360(7)")
            print("Total: 20*20*7*7 = 19600 cases (total_rate>100 드랍,SHORT_PROFIT<-10% 드랍, LONG_PROFIT<-10% 드랍)")
            n_workers = min(16, os.cpu_count() or 8)
            print(f"Parallel: ProcessPoolExecutor max_workers={n_workers}")
            print("="*80)
            
            short_counter_range = list(range(1, 21))   # 1~20
            long_counter_range = list(range(1, 21))    # 1~20
            ratio_denominators = list(range(300, 370, 10))  # 300,310,320,330,340,350,360 → 7개
            
            tasks = [
                (qqq_data, sc, lc, sd, ld)
                for sc in short_counter_range
                for lc in long_counter_range
                for sd in ratio_denominators
                for ld in ratio_denominators
            ]
            total_combos = len(tasks)
            results_matrix = {}
            done = 0
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for key, balance in executor.map(_run_one_case, tasks):
                    done += 1
                    if done % 500 == 0 or done == total_combos:
                        print(f"Progress: {done}/{total_combos}", flush=True)
                    if balance is not None:
                        results_matrix[key] = balance
            
            print("\n" + "="*80)
            print("Optimization complete. Finding best (short_time, long_time, short_ratio, long_ratio)...")
            print("="*80)
            
            if not results_matrix:
                raise ValueError("All combinations were dropped (total_rate>100). No valid parameter set.")
            
            best_combo = max(results_matrix, key=results_matrix.get)
            best_short_counter = best_combo[0]
            best_long_counter = best_combo[1]
            best_short_denom = best_combo[2]
            best_long_denom = best_combo[3]
            best_balance = results_matrix[best_combo]
            
            print(f"\n✓ BEST: Short={best_short_counter}, Long={best_long_counter}, Ratio 1/{best_short_denom} 1/{best_long_denom}, Balance={best_balance:.2f}")
            print("="*80)
        
        # 최적 조건으로 최종 실행 및 저장
        print(f"\nGenerating final results with optimal parameters...")
        all_results, event_results = simulated_trade(
            qqq_data,
            short_max_counter=best_short_counter,
            long_max_counter=best_long_counter,
            short_size_ratio=1.0 / best_short_denom,
            long_size_ratio=1.0 / best_long_denom
        )
        
        # Check drop condition: ANY row where total_rate > 100
        if 'total_rate' in all_results.columns:
            total_rate_col = pd.to_numeric(all_results['total_rate'], errors='coerce')
            
            # 100을 초과하는 첫 번째 행 찾기
            exceeding_mask = total_rate_col > 100
            if exceeding_mask.any():
                first_exceed_idx = exceeding_mask.idxmax()
                exceed_value = total_rate_col[first_exceed_idx]
                exceed_timestamp = all_results['Datetime'].iloc[first_exceed_idx]
                
                print(f"\n!!! DROP CONDITION MET !!!")
                print(f"Timestamp: {exceed_timestamp}")
                print(f"total_rate: {exceed_value:.2f} > 100")
                print("✗ This parameter set is DROPPED (excluded from results)")
                print("="*80)
                sys.exit(0)

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
        
        optimal_all_csv = f"qqq_backtest_optimal_s{best_short_counter}_l{best_long_counter}_ratio{best_short_denom}_{best_long_denom}_all.csv"
        optimal_event_csv = f"qqq_backtest_optimal_s{best_short_counter}_l{best_long_counter}_ratio{best_short_denom}_{best_long_denom}_results.csv"
        all_results.to_csv(optimal_all_csv, index=False)
        event_results.to_csv(optimal_event_csv, index=False)
        print(f"✓ Saved: {optimal_all_csv}, {optimal_event_csv}")

        from datetime import datetime
        now = datetime.now().strftime("_%y%m%d_%H%M%S")
        remote_filename = f"qqq_backtest_optimal_s{best_short_counter}_l{best_long_counter}_ratio{best_short_denom}_{best_long_denom}_all{now}.csv"
        try:
            subprocess.run(["rclone", "copyto", optimal_all_csv, f"BackTest:/{remote_filename}"], check=True, timeout=30)
            print(f"✓ Uploaded to Google Drive: {remote_filename}")
        except Exception as e:
            print(f"rclone upload failed: {e}")
        
        print("\n" + "="*80)
        print(f"Backtest Results: SHORT={best_short_counter}, LONG={best_long_counter}, RATIO 1/{best_short_denom} 1/{best_long_denom}")
        print("="*80)

        # =====================
        # 성과 지표
        # =====================
        equity_curve = all_results['total_balance'].tolist()
        # all_results에서 total_balance를 사용합니다

        if not equity_curve or len(equity_curve) < 2:
            raise ValueError("Equity curve is empty or too short. No trades were executed or data is insufficient.")

        equity = pd.Series(equity_curve)
        returns = equity.pct_change().dropna()

        if returns.empty:
            raise ValueError("Returns are empty. No trades or insufficient data.")

        # =====================
        # 드랍 조건 검사: total_rate > 100
        # =====================
        if 'total_rate' in all_results.columns:
            total_rate_series = pd.to_numeric(all_results['total_rate'], errors='coerce')
            if not total_rate_series.empty:
                last_total_rate = total_rate_series.iloc[-1]
                if last_total_rate > 100:
                    print(f"\n!!! DROP CONDITION MET: total_rate={last_total_rate:.2f} > 100 !!!")
                    print("✗ This parameter set is DROPPED (excluded from results)")
                    print("="*80)
                    sys.exit(0)
        
        # =====================
        # 성과 지표
        # =====================    

        mdd = ((equity.cummax() - equity) / equity.cummax()).max()
        win_rate = (returns > 0).mean()

        print(f"최종 수익률: {(equity.iloc[-1] / equity.iloc[0] - 1):.2%}")
        print(f"MDD: {mdd:.2%}")
        print(f"승률: {win_rate:.2%}")
        if (returns > 0).any() and (returns < 0).any():
            profit_factor = returns[returns > 0].mean() / abs(returns[returns < 0].mean())
            print(f"손익비: {profit_factor:.2f}")
        else:
            print("손익비 계산 불가: 손실 또는 이익이 없음")
    except Exception as e:
        print("Error during backtesting:", e)