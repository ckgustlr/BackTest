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
SINGLE_TEST_MODE = True

# Short Strategy Toggle: Set to False to test LONG-only strategy separately
# Change this to False to validate long-only performance without short positions
ENABLE_SHORT = False  # Change to False for LONG-only testing

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
def simulated_trade(data, short_max_counter=3, long_max_counter=3, short_size_ratio=1/320, long_size_ratio=1/320, rebalance_increment=0.001, enable_short=True, verbose=True, single_mode=False):
    if verbose:
        mode_str = "LONG only" if not enable_short else "SHORT + LONG"
        print(f"Starting simulated trading ({mode_str}) with short_max_counter={short_max_counter}, long_max_counter={long_max_counter}, short_ratio={short_size_ratio:.2f}, long_ratio={long_size_ratio:.2f}...")
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

    if verbose:
        print(f"  SHORT_SIZE_RATIO={SHORT_SIZE_RATIO:.4f}, LONG_SIZE_RATIO={LONG_SIZE_RATIO:.4f}")

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
            
            # enable_short가 False면 전체 금액을 long에만 사용
            if enable_short:
                long_entry_size = (current_total_balance * LONG_SIZE_RATIO) / current_price
                short_entry_size = (current_total_balance * SHORT_SIZE_RATIO) / current_price
                free_balance -= (long_entry_size * current_price + short_entry_size * current_price)
                long_positions.append({'price': current_price, 'size': long_entry_size, 'unit': 1.0, 'profit_target': None, 'target_price': round(current_price * 1.01, 2), 'active': True, 'entry_time': timestamp})
                short_positions.append({'price': current_price, 'size': short_entry_size, 'unit': 1.0, 'profit_target': None, 'target_price': round(current_price * 0.99, 2), 'active': True, 'entry_time': timestamp})
                short_amount_cum = short_entry_size * current_price
                last_entry_price_short = current_price
                short_hour_count = 1
                if verbose:
                    print(f"  [INIT] Long Entry: size={long_entry_size:.4f}, price={current_price}, capital={long_entry_size*current_price:.2f}")
                    print(f"  [INIT] Short Entry: size={short_entry_size:.4f}, price={current_price}, capital={short_entry_size*current_price:.2f}")
            else:
                # Short을 disable하는 경우 long비율만 사용
                long_entry_size = (current_total_balance * LONG_SIZE_RATIO) / current_price
                short_entry_size = 0  # No short position when disabled
                free_balance -= (long_entry_size * current_price)
                long_positions.append({'price': current_price, 'size': long_entry_size, 'unit': 1.0, 'profit_target': None, 'target_price': round(current_price * 1.01, 2), 'active': True, 'entry_time': timestamp})
                short_amount_cum = 0
                if verbose:
                    print(f"  [INIT LONG-ONLY] Long Entry: size={long_entry_size:.4f}, price={current_price}, capital={long_entry_size*current_price:.2f}")
            
            # Update cumulative amounts for first entry
            long_amount_cum = long_entry_size * current_price
            last_entry_price_long = current_price
            first_entry_done = True
            # Set counters to 1 for the initial entry
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
                'entry_short': round(current_price, 2) if enable_short else '',
                'exit_short': '',
                'short_size': round(short_entry_size, 2),
                'short_count': round(1.0, 2) if enable_short else 0,
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
                'short_hour_count': short_hour_count if enable_short else 0,
                'long_hour_count': long_hour_count,
                'long_realized_pnl': round(0, 2),
                'short_realized_pnl': round(0, 2)
            }
            # Initialize cumulative principal amounts with the first entries
            short_amount_cum = result['entry_short'] * result['short_size'] if enable_short else 0
            long_amount_cum = result['entry_long'] * result['long_size']

            # Calculate rate columns for initial entry based on cumulative principal
            half_balance = total_balance / 2 if total_balance != 0 else 0

            if half_balance > 0:
                result['short_entry_rate'] = round(short_amount_cum / half_balance * 100, 2) if enable_short else 0
                result['long_entry_rate'] = round(long_amount_cum / half_balance * 100, 2)
            else:
                result['short_entry_rate'] = 0
                result['long_entry_rate'] = 0

            if total_balance > 0:
                if enable_short:
                    result['total_rate'] = round((short_amount_cum + long_amount_cum) / total_balance * 100, 2)
                else:
                    result['total_rate'] = round(long_amount_cum / total_balance * 100, 2)
            else:
                result['total_rate'] = 0

            result_rows.append(result.copy())
            event_rows.append(result.copy())
            continue

        # Ensure no division by zero errors in average price calculations (active only)
        try:
            active_long = [p for p in long_positions if p.get('active', True)]
            long_avg_price = sum(p['price'] * p['size'] for p in active_long) / sum(p['size'] for p in active_long) if active_long else 0
        except ZeroDivisionError:
            long_avg_price = 0

        try:
            active_short = [p for p in short_positions if p.get('active', True)]
            short_avg_price = sum(p['price'] * p['size'] for p in active_short) / sum(p['size'] for p in active_short) if active_short else 0
        except ZeroDivisionError:
            short_avg_price = 0

        # Track if exits occurred (reset for this iteration)
        short_exit_occurred = False
        long_exit_occurred = False
        
        # Save average prices BEFORE exit/entry processing for accurate profit calculations
        short_avg_price_at_exit = short_avg_price
        long_avg_price_at_exit = long_avg_price
        
        # Calculate total sizes for exit logic
        short_total_size = sum(p['size'] for p in short_positions) if short_positions else 0
        long_total_size = sum(p['size'] for p in long_positions) if long_positions else 0

        # 2. 분할 매도 조건 (각 투입 건별 1% 수익 시)
        exit_short_parts = []
        if enable_short:  # Only process short exits when enabled
            for p in short_positions:
                if p.get('active', True) and current_price <= p['target_price']:
                    received = p['size'] * current_price * (1 + LEVERAGE * (p['price'] - current_price) / p['price'])
                    # 평단 기준 손익 계산 (트리거된 target_price 기준으로, exit 시점의 평단 사용)
                    actual_profit = p['size'] * (short_avg_price_at_exit - p['target_price']) * LEVERAGE
                    free_balance += received
                    current_short_pnl += actual_profit
                    short_realized_pnl += actual_profit
                    short_amount_cum = max(0, short_amount_cum - p['size'] * p['price'])
                    exit_short_parts.append(f"{p['price']:.2f}->{current_price:.2f}({p['size']:.4f})")
                    p['active'] = False  # Mark as closed, don't remove from list
                    # print(f"분할 매도(숏) 실현: {current_price} (진입가: {p['price']}, size: {p['size']})")
            if exit_short_parts:
                short_exit_occurred = True

        exit_long_parts = []
        for p in long_positions:
            if p.get('active', True) and current_price >= p['target_price']:
                received = p['size'] * current_price * (1 + LEVERAGE * (current_price - p['price']) / p['price'])
                # 평단 기준 손익 계산 (트리거된 target_price 기준으로, exit 시점의 평단 사용)
                actual_profit = p['size'] * (p['target_price'] - long_avg_price_at_exit) * LEVERAGE
                free_balance += received
                current_long_pnl += actual_profit
                long_realized_pnl += actual_profit
                long_amount_cum = max(0, long_amount_cum - p['size'] * p['price'])
                exit_long_parts.append(f"{p['price']:.2f}->{current_price:.2f}({p['size']:.4f})")
                p['active'] = False  # Mark as closed, don't remove from list
                # print(f"분할 매도(롱) 실현: {current_price} (진입가: {p['price']}, size: {p['size']})")
        if exit_long_parts:
            long_exit_occurred = True

        # Rebalance logic every 2 hours
        # This should be placed after exit logic and before entry logic

        # Initialize rebalancing trackers if they don't exist
        if 'last_rebalance_time_long' not in locals():
            last_rebalance_time_long = None
        if 'last_rebalance_time_short' not in locals():
            last_rebalance_time_short = None

        # --- Long Position Rebalancing ---
        # 1시간 이상 전에 진입한 활성 포지션만 재조정 대상
        active_long_positions = [p for p in long_positions if p.get('active', True) and (p.get('entry_time') is None or (timestamp - p.get('entry_time', timestamp)) >= datetime.timedelta(hours=1))]
        if active_long_positions:
            # Check if 1 hour has passed since the last rebalance
            if last_rebalance_time_long is None or (timestamp - last_rebalance_time_long) >= datetime.timedelta(hours=1):
                long_avg_price_before = sum(p['price'] * p['size'] for p in active_long_positions) / sum(p['size'] for p in active_long_positions)
                
                current_profit_percent = (current_price - long_avg_price_before) / long_avg_price_before

                # Sort positions by their original entry price to apply new targets systematically
                active_long_positions.sort(key=lambda p: p['price'])
                
                # Rebalance strategy: Set target prices based on current profit + incremental steps
                # If current profit is 1.5%, then:
                # 1st position: 1.5% + 0.1% = 1.6%
                # 2nd position: 1.5% + 0.6% = 2.1% (1.5% + 0.1% * 6)
                # 3rd position: 1.5% + 1.1% = 2.6% (1.5% + 0.1% * 11)
                # Pattern: current_profit_percent + rebalance_increment * (1 + 5*i)

                if verbose:
                    print(f"  [REBALANCE LONG] time={timestamp}, current_price={current_price:.2f}, long_avg={long_avg_price_before:.2f}, current_profit%={current_profit_percent*100:.2f}%, count={len(active_long_positions)}")

                # Assign new target prices to remaining long positions
                for i, p in enumerate(active_long_positions):
                    new_target_profit_pct = current_profit_percent + rebalance_increment * (1 + 5 * i)
                    p['target_price'] = long_avg_price_before * (1 + new_target_profit_pct)
                    if verbose:
                        print(f"    [{i}] target_profit%={new_target_profit_pct*100:.2f}%, target_price={p['target_price']:.2f}")
                
                last_rebalance_time_long = timestamp

        # --- Short Position Rebalancing ---
        # 1시간 이상 전에 진입한 활성 포지션만 재조정 대상
        if enable_short:  # Only rebalance shorts when enabled
            active_short_positions = [p for p in short_positions if p.get('active', True) and (p.get('entry_time') is None or (timestamp - p.get('entry_time', timestamp)) >= datetime.timedelta(hours=1))]
            if active_short_positions:
                # Check if 1 hour has passed since the last rebalance
                if last_rebalance_time_short is None or (timestamp - last_rebalance_time_short) >= datetime.timedelta(hours=1):
                    # Rebalance target 계산용 (평단 손상 방지)
                    short_rebalance_avg = sum(p['price'] * p['size'] for p in active_short_positions) / sum(p['size'] for p in active_short_positions)
                    current_profit_percent = (short_rebalance_avg - current_price) / short_rebalance_avg

                    # Sort positions by their original entry price
                    active_short_positions.sort(key=lambda p: p['price'], reverse=True) # Higher price first
                    
                    # Rebalance strategy: Set target prices based on current profit + incremental steps
                    # Pattern: current_profit_percent - rebalance_increment * (1 + 5*i) (inverted for short)

                    if verbose:
                        print(f"  [REBALANCE SHORT] time={timestamp}, current_price={current_price:.2f}, short_avg={short_rebalance_avg:.2f}, current_profit%={current_profit_percent*100:.2f}%, count={len(active_short_positions)}")

                    # Assign new target prices to remaining short positions
                    for i, p in enumerate(active_short_positions):
                        new_target_profit_pct = current_profit_percent + rebalance_increment * (1 + 5 * i)
                        p['target_price'] = short_rebalance_avg * (1 - new_target_profit_pct)
                        if verbose:
                            print(f"    [{i}] target_profit%={new_target_profit_pct*100:.2f}%, target_price={p['target_price']:.2f}")

                    last_rebalance_time_short = timestamp
                    
        # Create result dictionary AFTER exit logic
        # Exit 후에는 평단을 재계산하지 않음 - entry 시에만 변경
        # 활성 포지션만 따로 추출 (포지션 카운트, 사이즈, 평가용)
        active_long_calc = [p for p in long_positions if p.get('active', True)]
        active_short_calc = [p for p in short_positions if p.get('active', True)]
        
        # Exit 후 활성 포지션이 없으면 리스트를 완전히 비우고 평단 초기화
        if not active_long_calc:
            long_positions.clear()
            long_avg_price = 0  # 평단 초기화
        if not active_short_calc:
            short_positions.clear()
            short_avg_price = 0  # 평단 초기화
        
        # 현재 활성 포지션의 평단 (exit 후에도 유지, entry 후에만 변경)
        short_profit_after = round((short_avg_price - current_price) / current_price * 100, 2) if current_price != 0 else 0
        long_profit_after = round((current_price - long_avg_price) / long_avg_price * 100, 2) if long_avg_price != 0 else 0
        long_val = sum(p['size'] * current_price * (1 + LEVERAGE * (current_price - p['price']) / p['price']) for p in active_long_calc) if active_long_calc else 0
        short_val = sum(p['size'] * current_price * (1 + LEVERAGE * (p['price'] - current_price) / p['price']) for p in active_short_calc) if active_short_calc else 0
        total_balance_here = free_balance + long_val + short_val

        # 현재 보유 포지션의 exit 준비 가격 표시 (active만 표시)
        exit_ready_list = []
        for p in active_long_calc:
            exit_ready_list.append(f"L:{p['target_price']:.2f}({p['size']:.4f})")
        for p in active_short_calc:
            exit_ready_list.append(f"S:{p['target_price']:.2f}({p['size']:.4f})")
        current_ready_to_exit = ", ".join(exit_ready_list) if exit_ready_list else ""
        
        result = {
            'Datetime': timestamp,
            'Close': round(float(row['Close']), 2),
            'short_avg': round(short_avg_price, 2),
            'short_profit': short_profit_after,
            'entry_short': '',
            'exit_short': ','.join(exit_short_parts) if exit_short_parts else '',
            'short_size': '',
            'short_count': float(len(active_short_calc)),
            'short_total_size': round(sum(p['size'] for p in active_short_calc) if active_short_calc else 0, 2),
            'entry_long': '',
            'exit_long': ','.join(exit_long_parts) if exit_long_parts else '',
            'long_size': '',
            'long_count': float(len(active_long_calc)),
            'long_total_size': round(sum(p['size'] for p in active_long_calc) if active_long_calc else 0, 2),
            'long_avg': round(long_avg_price, 2),
            'long_profit': long_profit_after,
            'free_balance': round(free_balance, 2),
            'long_val': round(long_val, 2),
            'short_val': round(short_val, 2),
            'total_balance': round(total_balance_here, 2),
            'short_hour_count': short_hour_count,
            'long_hour_count': long_hour_count,
            'long_realized_pnl': round(current_long_pnl, 2),
            'short_realized_pnl': round(current_short_pnl, 2),
            'Current_readyto_Exit': current_ready_to_exit  # ← 마지막에 추가
        }

        # Entry conditions based on separate short and long counters
        entry_flag = False
        
        # Calculate current total balance for entry size (달러 기준)
        current_total_balance = free_balance + long_val + short_val
        short_entry_base = (current_total_balance * SHORT_SIZE_RATIO) / current_price
        long_entry_base = (current_total_balance * LONG_SIZE_RATIO) / current_price

        # Entry decision making - active positions만 기반으로 (평단 손상 방지 위해 별도 변수 사용)
        # 진입 시점에 활성 포지션만으로 평균가 계산 (모든 inactive는 무시)
        short_avg_for_entry = sum(p['price'] * p['size'] for p in active_short_calc) / sum(p['size'] for p in active_short_calc) if active_short_calc else 0
        long_avg_for_entry = sum(p['price'] * p['size'] for p in active_long_calc) / sum(p['size'] for p in active_long_calc) if active_long_calc else 0

        # Short entry logic when short_hour_count == short_max_counter
        if enable_short and short_hour_count == short_max_counter:  # Only when short is enabled
            if current_price > short_avg_for_entry:
                entry_size = short_entry_base  # Full size (수량)
                free_balance -= entry_size * current_price
                short_positions.append({'price': current_price, 'size': entry_size, 'unit': 1.0, 'profit_target': None, 'target_price': round(current_price * 0.99, 2), 'active': True, 'entry_time': timestamp})
                last_entry_price_short = current_price
                result['entry_short'] = current_price
                result['short_size'] = entry_size
                # Add to cumulative short principal
                short_amount_cum += result['entry_short'] * result['short_size']
                entry_flag = True
            elif current_price < short_avg_for_entry:
                entry_size = short_entry_base / 2  # Half size
                free_balance -= entry_size * current_price
                short_positions.append({'price': current_price, 'size': entry_size, 'unit': 0.5, 'profit_target': None, 'target_price': round(current_price * 0.99, 2), 'active': True, 'entry_time': timestamp})
                last_entry_price_short = current_price
                result['entry_short'] = current_price
                result['short_size'] = entry_size
                # Add to cumulative short principal
                short_amount_cum += result['entry_short'] * result['short_size']
                entry_flag = True

        # Long entry logic when long_hour_count == long_max_counter
        if long_hour_count == long_max_counter:
            if current_price < long_avg_for_entry:
                entry_size = long_entry_base  # Full size (수량)
                free_balance -= entry_size * current_price
                long_positions.append({'price': current_price, 'size': entry_size, 'unit': 1.0, 'profit_target': None, 'target_price': round(current_price * 1.01, 4), 'active': True, 'entry_time': timestamp})
                last_entry_price_long = current_price
                result['entry_long'] = current_price
                result['long_size'] = entry_size
                # Add to cumulative long principal
                long_amount_cum += result['entry_long'] * result['long_size']
                entry_flag = True
            elif current_price > long_avg_for_entry:
                entry_size = long_entry_base / 2  # Half size
                free_balance -= entry_size * current_price
                long_positions.append({'price': current_price, 'size': entry_size, 'unit': 0.5, 'profit_target': None, 'target_price': round(current_price * 1.01, 4), 'active': True, 'entry_time': timestamp})
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
        active_long_final = [p for p in long_positions if p.get('active', True)]
        active_short_final = [p for p in short_positions if p.get('active', True)]
        
        # Entry 후에 평단 업데이트 (entry가 발생했을 때만)
        if entry_flag:
            # Entry가 발생했으므로 평단 재계산 및 업데이트
            short_avg_price = sum(p['price'] * p['size'] for p in active_short_final) / sum(p['size'] for p in active_short_final) if active_short_final else 0
            long_avg_price = sum(p['price'] * p['size'] for p in active_long_final) / sum(p['size'] for p in active_long_final) if active_long_final else 0
            
            # Entry 직후 각 포지션의 target_price를 재조정 간격으로 설정 (1%, 1+ri%, 1+2*ri%, ...)
            # 각 포지션이 다른 레벨에서 차례로 exit되도록
            for i, p in enumerate(active_long_final):
                target_profit_pct = 0.01 + i * rebalance_increment
                p['target_price'] = round(long_avg_price * (1 + target_profit_pct), 4)
            for i, p in enumerate(active_short_final):
                target_profit_pct = 0.01 + i * rebalance_increment
                p['target_price'] = round(short_avg_price * (1 - target_profit_pct), 2)
            
            # Result에 새로운 평단 반영
            result['short_avg'] = round(short_avg_price, 2)
            result['long_avg'] = round(long_avg_price, 2)
            
            short_profit_after_entry = round((short_avg_price - current_price) / current_price * 100, 2) if current_price != 0 else 0
            long_profit_after_entry = round((current_price - long_avg_price) / long_avg_price * 100, 2) if long_avg_price != 0 else 0
            
            result['short_profit'] = short_profit_after_entry
            result['long_profit'] = long_profit_after_entry
        
        short_total_size = sum(p['size'] for p in active_short_final) if active_short_final else 0
        long_total_size = sum(p['size'] for p in active_long_final) if active_long_final else 0
        
        # Entry 후 count도 재계산 (이전 값을 덮어쓰기)
        result['short_count'] = float(len(active_short_final))
        result['long_count'] = float(len(active_long_final))
        long_val = sum(p['size'] * current_price * (1 + LEVERAGE * (current_price - p['price']) / p['price']) for p in active_long_final) if active_long_final else 0
        short_val = sum(p['size'] * current_price * (1 + LEVERAGE * (p['price'] - current_price) / p['price']) for p in active_short_final) if active_short_final else 0
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
            result['short_entry_rate'] = round(short_amount / half_balance * 100, 2) if enable_short else 0
            result['long_entry_rate'] = round(long_amount / half_balance * 100, 2)
        else:
            result['short_entry_rate'] = 0
            result['long_entry_rate'] = 0

        if total_balance > 0:
            # When short is disabled, calculate total_rate using only long_amount
            if enable_short:
                result['total_rate'] = round((short_amount + long_amount) / total_balance * 100, 2)
            else:
                result['total_rate'] = round(long_amount / total_balance * 100, 2)
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

    if verbose:
        print("Simulated trading completed.")
    
    # ✓ 드랍 조건 검사 (DataFrame 생성 前) - 싱글모드는 계속, 멀티모드는 중단
    if result_rows:
        last_result = result_rows[-1]
        if last_result.get('total_rate', 0) > 100:
            if single_mode:
                # 싱글 모드: 로그만 출력하고 계속 진행
                if verbose:
                    print(f"  ! DROPPED (LOG ONLY): total_rate={last_result.get('total_rate')} > 100")
            else:
                # 멀티 모드: 즉시 중단
                if verbose:
                    print(f"  ! DROPPED: total_rate={last_result.get('total_rate')} > 100")
                return pd.DataFrame(), pd.DataFrame()
    
    # Return both DataFrames: all timestamps and only events
    return pd.DataFrame(result_rows), pd.DataFrame(event_rows)

def _run_one_case(tup):
    """멀티프로세싱 워커: (data, short_counter, long_counter, short_denom, long_denom, rebalance_increment) → ((sc,lc,sd,ld,ri), balance or None)"""
    data, short_counter, long_counter, short_denom, long_denom, rebalance_increment = tup
    short_ratio = 1.0 / short_denom
    long_ratio = 1.0 / long_denom
    try:
        all_results, _ = simulated_trade(
            data,
            short_max_counter=short_counter,
            long_max_counter=long_counter,
            short_size_ratio=short_ratio,
            long_size_ratio=long_ratio,
            rebalance_increment=rebalance_increment,
            enable_short=ENABLE_SHORT,  # Use the global ENABLE_SHORT setting
            verbose=False,  # 멀티 모드에서는 상세 출력 비활성화
            single_mode=False
        )

        # 드랍 조건: 결과가 비어있거나, total_rate가 한 번이라도 100을 초과한 경우
        if all_results.empty:
            return ((short_counter, long_counter, short_denom, long_denom, rebalance_increment), None)

        total_rate_series = pd.to_numeric(all_results["total_rate"], errors="coerce")
        if (total_rate_series > 100).any():
            return ((short_counter, long_counter, short_denom, long_denom, rebalance_increment), None)

        final_balance = all_results["total_balance"].iloc[-1]
        return ((short_counter, long_counter, short_denom, long_denom, rebalance_increment), final_balance)
    except Exception:
        return ((short_counter, long_counter, short_denom, long_denom, rebalance_increment), None)


# Main function for backtesting
if __name__ == "__main__":
    try:
        qqq_data = fetch_qqq_data()
        
        if SINGLE_TEST_MODE:
            # Jetson 등: 단일 케이스만 실행 (빠른 확인용). 환경변수 QQQ_SINGLE_TEST=1 로 활성화
            print("\n" + "="*80)
            print("SINGLE TEST MODE (QQQ_SINGLE_TEST=1): 1 case only — S=12, L=12, ratio 1/100 1/100")
            print("="*80)
            best_short_counter, best_long_counter = 12, 12
            best_short_denom, best_long_denom = 100, 100
            best_rebalance_increment = 0.001 
        else:
            # 1~20 x 1~20 시간, 비율 1/50~1/110 (10단위 7개) 각각 → 20*20*7*7 = 19600 케이스, total_rate>100 드랍
            # CPU 멀티프로세싱으로 병렬 실행
            print("\n" + "="*80)
            print("Testing: SHORT_TIME=1~20, LONG_TIME=1~20, SHORT_RATIO=1/100~1/150(6), LONG_RATIO=1/100~1/150(6), REBALANCE_INC=0.1~0.5%")
            print("Total: 20*20*6*6*5 = 72000 cases (total_rate>100 드랍)")
            n_workers = min(16, os.cpu_count() or 8)
            print(f"Parallel: ProcessPoolExecutor max_workers={n_workers}")
            print("="*80)
            
            short_counter_range = list(range(1, 22, 3))   # 1~20
            long_counter_range = list(range(1, 22, 3))    # 1~20
            ratio_denominators = [100]
            #ratio_denominators = list(range(100, 100, 10))  # 100,110,120,130,140,150 → 6개
            rebalance_increments = [0.001, 0.002, 0.003, 0.004, 0.005]  # 0.1% ~ 0.5% 재조정 간격
            
            tasks = [
                (qqq_data, sc, lc, sd, ld, ri)
                for sc in short_counter_range
                for lc in long_counter_range
                for sd in ratio_denominators
                for ld in ratio_denominators
                for ri in rebalance_increments
            ]
            total_combos = len(tasks)
            results_matrix = {}
            done = 0
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for key, balance in executor.map(_run_one_case, tasks):
                    done += 1
                    print(f"Progress: {done}/{total_combos} cases", end='\r', flush=True)
                    if balance is not None:
                        results_matrix[key] = balance
            print()  # Move to the next line after progress is done
            
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
            best_rebalance_increment = best_combo[4]
            best_balance = results_matrix[best_combo]
            
            print(f"\n✓ BEST: Short={best_short_counter}, Long={best_long_counter}, Ratio 1/{best_short_denom} 1/{best_long_denom}, Rebalance Inc={best_rebalance_increment:.1%}, Balance={best_balance:.2f}")
            print("="*80)
        
        # 최적 조건으로 최종 실행 및 저장
        print(f"\nGenerating final results with optimal parameters...")
        # SINGLE_TEST_MODE에서는 best_rebalance_increment가 없을 수 있으니 기본값 0.001 사용
        final_rebalance_increment = best_rebalance_increment if 'best_rebalance_increment' in locals() else 0.001
        all_results, event_results = simulated_trade(
            qqq_data,
            short_max_counter=best_short_counter,
            long_max_counter=best_long_counter,
            short_size_ratio=1.0 / best_short_denom,
            long_size_ratio=1.0 / best_long_denom,
            rebalance_increment=final_rebalance_increment,
            enable_short=ENABLE_SHORT,  # Use the global ENABLE_SHORT setting
            single_mode=SINGLE_TEST_MODE  # ← 이 줄 추가!!
        )

        # 빈 DataFrame 체크 (드랍된 경우 - 싱글 모드에서만 계속)
        if all_results.empty:
            if not SINGLE_TEST_MODE:
                print("✗ This parameter set was DROPPED")
                sys.exit(0)
            else:
                print("\n✗ DROPPED (Single mode: continuing with empty results)")
                print("="*80)
                # 빈 데이터로도 계속 진행

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
                if not SINGLE_TEST_MODE:
                    print("✗ This parameter set is DROPPED (excluded from results)")
                    print("="*80)
                    sys.exit(0)
                else:
                    print("! SINGLE MODE: Continuing despite drop condition for debugging/upload.")

        # Conditionally filter out short-related columns when ENABLE_SHORT=False
        if ENABLE_SHORT:
            col_order = [
                'Datetime','Close','entry_short','exit_short','short_size','short_count','short_total_size','short_avg',
                'short_profit','entry_long','exit_long','long_size','long_count','long_total_size','long_avg','long_profit',
                'free_balance','long_val','short_val','total_balance','short_hour_count','long_hour_count','long_realized_pnl','short_realized_pnl',
                'short_entry_rate','long_entry_rate','total_rate','Current_readyto_Exit'  # ← 추가
            ]
        else:
            # When short is disabled, exclude short-related columns
            col_order = [
                'Datetime','Close','entry_long','exit_long','long_size','long_count','long_total_size','long_avg','long_profit',
                'free_balance','long_val','total_balance','long_hour_count','long_realized_pnl',
                'long_entry_rate','total_rate','Current_readyto_Exit'
            ]
        all_results = all_results[col_order]
        all_results = all_results.drop_duplicates(subset=['Datetime'], keep='first')
        event_results = event_results[col_order]
        event_results = event_results.drop_duplicates(subset=['Datetime'], keep='first')
        
        # Rebalance increment를 파일 이름에 포함 (0.1% ~ 0.5%)
        ri_percent = int(final_rebalance_increment * 1000)  # 0.001 -> 1, 0.005 -> 5
        optimal_all_csv = f"qqq_backtest_optimal_s{best_short_counter}_l{best_long_counter}_ratio{best_short_denom}_{best_long_denom}_ri{ri_percent}_all.csv"
        optimal_event_csv = f"qqq_backtest_optimal_s{best_short_counter}_l{best_long_counter}_ratio{best_short_denom}_{best_long_denom}_ri{ri_percent}_results.csv"
        all_results.to_csv(optimal_all_csv, index=False)
        event_results.to_csv(optimal_event_csv, index=False)
        print(f"✓ Saved: {optimal_all_csv}, {optimal_event_csv}")

        from datetime import datetime
        now = datetime.now().strftime("_%y%m%d_%H%M%S")
        remote_filename = f"qqq_backtest_optimal_s{best_short_counter}_l{best_long_counter}_ratio{best_short_denom}_{best_long_denom}_ri{ri_percent}_all{now}.csv"
        try:
            subprocess.run(["rclone", "copyto", optimal_all_csv, f"BackTest:/{remote_filename}"], check=True, timeout=30)
            print(f"✓ Uploaded to Google Drive: {remote_filename}")
        except Exception as e:
            print(f"rclone upload failed: {e}")
        
        print("\n" + "="*80)
        print(f"Backtest Results: SHORT={best_short_counter}, LONG={best_long_counter}, RATIO 1/{best_short_denom} 1/{best_long_denom}, RI={final_rebalance_increment:.1%}")
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
                    if not SINGLE_TEST_MODE:
                        print("✗ This parameter set is DROPPED (excluded from results)")
                        print("="*80)
                        sys.exit(0)
                    else:
                        print("! SINGLE MODE: Continuing...")
        
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