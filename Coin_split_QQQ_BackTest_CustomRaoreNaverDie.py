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
def simulated_trade(data, short_max_counter=3, long_max_counter=3):

    print(f"Starting simulated trading with price step entry/exit...")

    # Bi-directional swing trading: long on 0.1 drop, short on 0.1 rise
    total_balance = 5000
    free_balance = total_balance
    long_positions = []
    short_positions = []
    result_rows = []
    event_rows = []
    last_long_entry = None
    last_short_entry = None
    # 진입 비율 계산
    # 각 포지션(롱/숏)은 전체 자산의 절반만 사용
    half_balance = total_balance / 2
    # 10% / 0.1% = 100 구간으로 나눔
    entry_count = 100
    # 각 진입마다 투입할 금액
    entry_amount = half_balance / entry_count
    # 평균 계산 변수 초기화 (루프 내에서 계산)
    long_avg = 0.0
    short_avg = 0.0
    avg_gap_pct = 0.0
    # 실현 손익 변수 초기화 (누적)
    realized_long = 0.0
    realized_short = 0.0
    realized_total = 0.0
    
    for i, (timestamp, row) in enumerate(data.iterrows()):
        current_price = round(float(row['Close']), 1)
        entry_long = False
        entry_short = False
        exit_long = False
        exit_short = False
        # 롱 진입: 시간 간격으로 분할 매수 (매번 진입)
        buy_volume = round(entry_amount / current_price, 6)  # 소수점 6자리까지
        target = round(current_price * 1.01, 1)
        long_positions.append({'entry_price': current_price, 'volume': buy_volume, 'target_price': target})
        free_balance -= buy_volume * current_price
        last_long_entry = current_price
        entry_long = True
        
        # 숏 진입: 시간 간격으로 분할 매도 (매번 진입)
        sell_volume = round(entry_amount / current_price, 6)
        target = round(current_price * 0.99, 1)
        short_positions.append({'entry_price': current_price, 'volume': sell_volume, 'target_price': target})
        free_balance -= sell_volume * current_price
        last_short_entry = current_price
        entry_short = True
        # 롱 청산: 각 롱 진입건별 1% 수익 시
        exit_long_prices = []
        remaining_longs = []
        long_size_before_exit = sum(p['volume'] for p in long_positions)
        for p in long_positions:
            if current_price >= p['target_price']:
                free_balance += p['volume'] * current_price
                exit_long = True
                exit_long_prices.append(f"{round(p['entry_price'],1)}->{current_price}")
            else:
                remaining_longs.append(p)
        long_size_after_exit = sum(p['volume'] for p in remaining_longs)
        long_exited_size = long_size_before_exit - long_size_after_exit
        if long_exited_size > 0:
            realized_long += (long_exited_size * current_price / 10)
        long_positions = remaining_longs
        # 숏 청산: 각 숏 진입건별 1% 수익 시 (즉, 1% 하락 시)
        exit_short_prices = []
        remaining_shorts = []
        short_size_before_exit = sum(p['volume'] for p in short_positions)
        for p in short_positions:
            if current_price <= p['target_price']:
                free_balance -= p['volume'] * current_price
                exit_short = True
                exit_short_prices.append(f"{round(p['entry_price'],1)}->{current_price}")
            else:
                remaining_shorts.append(p)
        short_size_after_exit = sum(p['volume'] for p in remaining_shorts)
        short_exited_size = short_size_before_exit - short_size_after_exit
        if short_exited_size > 0:
            realized_short += (short_exited_size * current_price / 10)
        short_positions = remaining_shorts
        # 평균 계산 (포지션 업데이트 후)
        long_avg = round(sum(p['entry_price'] * p['volume'] for p in long_positions) / sum(p['volume'] for p in long_positions), 1) if long_positions else 0.0
        short_avg = round(sum(p['entry_price'] * p['volume'] for p in short_positions) / sum(p['volume'] for p in short_positions), 1) if short_positions else 0.0
        avg_gap_pct = round(((long_avg - short_avg) / ((long_avg+short_avg)/ 2) * 100) if long_avg else 0.0, 1)
        
        # 롱/숏 현재 투입 금액 계산 (현재 가격 기준)
        long_invested = sum(current_price * p['volume'] for p in long_positions)
        short_invested = sum(current_price * p['volume'] for p in short_positions)
        # 롱/숏 진입 수량 (각 진입 시점의 가격에 따른 사이즈를 합산)
        long_entry_size = sum(p['volume'] for p in long_positions)
        short_entry_size = sum(p['volume'] for p in short_positions)
        
        Long_entry_rate = round(long_invested / half_balance * 100, 2) if half_balance else 0.0
        Short_entry_rate = round(short_invested / half_balance * 100, 2) if half_balance else 0.0
        Total_entry_rate = round((long_invested + short_invested) / total_balance * 100, 2) if total_balance else 0.0
        
        # 전체 자산 = free_balance + 롱 투입액 + 숏 투입액 + 실현 손익 누적
        total_balance_calc = round(free_balance + long_invested + short_invested + realized_total, 1)
        
        # 실현 손익 누적값
        realized_total = round(realized_long + realized_short, 2)
        
        # 기록
        result = {
            'Datetime': timestamp,
            'Close': current_price,
            'entry_long': f"{current_price}" if entry_long else '',
            'exit_long': ','.join(exit_long_prices) if exit_long_prices else '',
            'entry_short': f"{current_price}" if entry_short else '',
            'exit_short': ','.join(exit_short_prices) if exit_short_prices else '',
            'long_count': len(long_positions),
            'short_count': len(short_positions),
            'long_entry_size': round(long_entry_size, 6),
            'short_entry_size': round(short_entry_size, 6),
            'long_avg': long_avg,
            'short_avg': short_avg,
            'avg_gap_pct': avg_gap_pct,
            'Long_entry_rate': Long_entry_rate,
            'Short_entry_rate': Short_entry_rate,
            'Total_entry_rate': Total_entry_rate,
            'realized_long': round(realized_long, 2),
            'realized_short': round(realized_short, 2),
            'realized_total': realized_total,
            'free_balance': round(free_balance, 1),
            'total_balance': total_balance_calc
        }
        result_rows.append(result.copy())
        if entry_long or exit_long or entry_short or exit_short:
            event_rows.append(result.copy())
    print("Simulated trading completed.")
    return pd.DataFrame(result_rows), pd.DataFrame(event_rows)

# Main function for backtesting
if __name__ == "__main__":
    try:
        qqq_data = fetch_qqq_data()

        # ====== 12,12 조건만 테스트 (임시) ======
        print("\n=== 가격 간격(0.1) 진입/청산 결과 확인 ===")
        all_results, event_results = simulated_trade(qqq_data)
        col_order = [
            'Datetime','Close','entry_long','exit_long','entry_short','exit_short',
            'long_count','short_count','long_entry_size','short_entry_size',
            'long_avg','short_avg','avg_gap_pct',
            'Long_entry_rate','Short_entry_rate','Total_entry_rate',
            'realized_long','realized_short','realized_total',
            'free_balance','total_balance'
        ]
        all_results = all_results[col_order]
        all_results = all_results.drop_duplicates(subset=['Datetime'], keep='first')
        event_results = event_results[col_order]
        event_results = event_results.drop_duplicates(subset=['Datetime'], keep='first')
        all_results.to_csv("qqq_backtest_pricegap_all.csv", index=False)
        event_results.to_csv("qqq_backtest_pricegap_events.csv", index=False)
        print("가격 간격 전체 결과: qqq_backtest_pricegap_all.csv")
        print("가격 간격 이벤트 결과: qqq_backtest_pricegap_events.csv")

        # === 드라이브 업로드: 시간 붙여서 ===
        from datetime import datetime
        now = datetime.now().strftime("_%y%m%d_%H%M%S")
        remote_filename = f"qqq_backtest_pricegap_all{now}.csv"
        rclone_cmd = [
            "rclone", "copyto",
            "qqq_backtest_pricegap_all.csv",
            f"qqq:{remote_filename}"
        ]
        try:
            subprocess.run(rclone_cmd, check=True)
            print(f"업로드 완료: {remote_filename} (Google Drive)")
        except Exception as e:
            print(f"rclone 업로드 실패: {e}")
        # ====== END ======
    except Exception as e:
        print("Error during price step test:", e)