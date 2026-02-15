import yfinance as yf
import pandas as pd
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import subprocess

# Hyperparameters
LEVERAGE = 10
ENTRY_SIZE_RATIO = 1 / 160  # 고정된 투입 비율

# Fetch QQQ 1-hour data for the last year
def fetch_qqq_data():
    csv_file = "qqq_1h_1year_data.csv"
    try:
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
            time.sleep(2)
        if all_data:
            data = pd.concat(all_data)
            data.to_csv(csv_file, index=True, header=True)
            print(f"QQQ 1h 1year data saved to {csv_file}")
            return data
        else:
            raise ValueError("No data fetched.")

# Simulated trading function
def simulated_trade(data, long_entry_gap=0.001, short_entry_gap=0.001):
    """
    long_entry_gap: 롱 진입 간격 (0.001 = 0.1%)
    short_entry_gap: 숏 진입 간격
    """
    print(f"[TRADE] long_gap={long_entry_gap*100:.1f}%, short_gap={short_entry_gap*100:.1f}%")
    
    free_balance = 5000
    long_positions = []
    short_positions = []
    last_entry_price_long = None
    last_entry_price_short = None
    first_entry_done = False

    data = data[pd.to_numeric(data['Close'], errors='coerce').notna()].copy()
    data['Close'] = data['Close'].astype(float)
    data = data[(data.index.dayofweek < 5) & ((data.index.dayofweek != 5) | (data.index.hour < 10))]

    long_realized_pnl = 0
    short_realized_pnl = 0
    result_rows = []
    event_rows = []
    short_amount_cum = 0
    long_amount_cum = 0

    for timestamp, row in data.iterrows():
        current_price = row['Close']
        
        short_avg = sum(p['price'] * p['size'] for p in short_positions) / sum(p['size'] for p in short_positions) if short_positions else 0
        short_profit = round((short_avg - current_price) / current_price * 100, 2) if current_price != 0 else 0

        long_avg = sum(p['price'] * p['size'] for p in long_positions) / sum(p['size'] for p in long_positions) if long_positions else 0
        long_profit = round((current_price - long_avg) / long_avg * 100, 2) if long_avg != 0 else 0

        current_short_pnl = 0
        current_long_pnl = 0

        # === INITIAL ENTRY ===
        if not first_entry_done:
            current_total_balance = free_balance
            long_entry_size = (current_total_balance * ENTRY_SIZE_RATIO) / current_price
            short_entry_size = (current_total_balance * ENTRY_SIZE_RATIO) / current_price
            
            free_balance -= (long_entry_size * current_price + short_entry_size * current_price)
            long_positions.append({'price': current_price, 'size': long_entry_size})
            short_positions.append({'price': current_price, 'size': short_entry_size})
            
            long_amount_cum = long_entry_size * current_price
            short_amount_cum = short_entry_size * current_price
            last_entry_price_long = current_price
            last_entry_price_short = current_price
            first_entry_done = True
            
            long_val = sum(p['size'] * current_price * (1 + LEVERAGE * (current_price - p['price']) / p['price']) for p in long_positions)
            short_val = sum(p['size'] * current_price * (1 + LEVERAGE * (p['price'] - current_price) / p['price']) for p in short_positions)
            total_balance = free_balance + long_val + short_val

            result = {
                'Datetime': timestamp,
                'Close': current_price,
                'entry_long': current_price,
                'entry_short': current_price,
                'long_count': 1,
                'short_count': 1,
                'long_profit': '',
                'short_profit': '',
                'free_balance': round(free_balance, 2),
                'long_val': round(long_val, 2),
                'short_val': round(short_val, 2),
                'total_balance': round(total_balance, 2),
                'long_realized_pnl': 0,
                'short_realized_pnl': 0,
                'total_entry_amount': round(long_amount_cum + short_amount_cum, 2)
            }
            result_rows.append(result.copy())
            event_rows.append(result.copy())
            continue

        # === EXIT LOGIC: 1% 이익 ===
        short_exit_occurred = False
        long_exit_occurred = False

        if short_profit >= 1 and short_positions:
            received = sum(p['size'] * current_price * (1 + LEVERAGE * (p['price'] - current_price) / p['price']) for p in short_positions)
            profit_1percent = sum(p['size'] * p['price'] * 0.01 * LEVERAGE for p in short_positions)
            
            free_balance += received
            current_short_pnl += profit_1percent
            short_realized_pnl += profit_1percent
            short_exit_occurred = True
            
            short_positions.clear()
            short_amount_cum = 0
            last_entry_price_short = None

        if long_profit >= 1 and long_positions:
            received = sum(p['size'] * current_price * (1 + LEVERAGE * (current_price - p['price']) / p['price']) for p in long_positions)
            profit_1percent = sum(p['size'] * p['price'] * 0.01 * LEVERAGE for p in long_positions)
            
            free_balance += received
            current_long_pnl += profit_1percent
            long_realized_pnl += profit_1percent
            long_exit_occurred = True
            
            long_positions.clear()
            long_amount_cum = 0
            last_entry_price_long = None

        # === ADDITIONAL ENTRY: 가격 기반 진입 ===
        if last_entry_price_long is not None:
            entry_threshold_long = last_entry_price_long * (1 - long_entry_gap)
            if current_price <= entry_threshold_long:
                current_total_balance = free_balance + sum(p['size'] * current_price * (1 + LEVERAGE * (current_price - p['price']) / p['price']) for p in long_positions) + sum(p['size'] * current_price * (1 + LEVERAGE * (p['price'] - current_price) / p['price']) for p in short_positions)
                long_entry_size = (current_total_balance * ENTRY_SIZE_RATIO) / current_price
                free_balance -= long_entry_size * current_price
                long_positions.append({'price': current_price, 'size': long_entry_size})
                long_amount_cum += long_entry_size * current_price
                last_entry_price_long = current_price

        if last_entry_price_short is not None:
            entry_threshold_short = last_entry_price_short * (1 + short_entry_gap)
            if current_price >= entry_threshold_short:
                current_total_balance = free_balance + sum(p['size'] * current_price * (1 + LEVERAGE * (current_price - p['price']) / p['price']) for p in long_positions) + sum(p['size'] * current_price * (1 + LEVERAGE * (p['price'] - current_price) / p['price']) for p in short_positions)
                short_entry_size = (current_total_balance * ENTRY_SIZE_RATIO) / current_price
                free_balance -= short_entry_size * current_price
                short_positions.append({'price': current_price, 'size': short_entry_size})
                short_amount_cum += short_entry_size * current_price
                last_entry_price_short = current_price

        # Calculate values
        long_val = sum(p['size'] * current_price * (1 + LEVERAGE * (current_price - p['price']) / p['price']) for p in long_positions) if long_positions else 0
        short_val = sum(p['size'] * current_price * (1 + LEVERAGE * (p['price'] - current_price) / p['price']) for p in short_positions) if short_positions else 0
        total_balance = free_balance + long_val + short_val

        result = {
            'Datetime': timestamp,
            'Close': current_price,
            'entry_long': '',
            'entry_short': '',
            'exit_long': current_price if long_exit_occurred else '',
            'exit_short': current_price if short_exit_occurred else '',
            'long_count': len(long_positions),
            'short_count': len(short_positions),
            'long_profit': long_profit,
            'short_profit': short_profit,
            'free_balance': round(free_balance, 2),
            'long_val': round(long_val, 2),
            'short_val': round(short_val, 2),
            'total_balance': round(total_balance, 2),
            'long_realized_pnl': current_long_pnl,
            'short_realized_pnl': current_short_pnl,
            'total_entry_amount': round(long_amount_cum + short_amount_cum, 2)
        }

        result_rows.append(result.copy())
        if result['entry_long'] or result['entry_short'] or result['exit_long'] or result['exit_short']:
            event_rows.append(result.copy())

    print("Trade completed.")
    return pd.DataFrame(result_rows), pd.DataFrame(event_rows)

# Main function
if __name__ == "__main__":
    try:
        qqq_data = fetch_qqq_data()
        
        print("\n" + "="*80)
        print("Testing PRICE GAP optimization (0.1% ~ 0.5% gap)")
        print("Total combinations: 5 * 5 = 25")
        print("="*80)
        
        gap_values = [0.001, 0.002, 0.003, 0.004, 0.005]
        gap_labels = ['0.1%', '0.2%', '0.3%', '0.4%', '0.5%']
        
        results_matrix = {}
        combination_count = 0
        
        for long_gap_idx, long_gap in enumerate(gap_values):
            for short_gap_idx, short_gap in enumerate(gap_values):
                combination_count += 1
                print(f"[{combination_count}/25] LONG_GAP={gap_labels[long_gap_idx]}, SHORT_GAP={gap_labels[short_gap_idx]}...", end=" ", flush=True)
                
                all_results, event_results = simulated_trade(qqq_data, long_entry_gap=long_gap, short_entry_gap=short_gap)

                # Filter: Drop if total_entry_amount > 100
                dropped = False
                total_entry_series = pd.to_numeric(all_results['total_entry_amount'], errors='coerce')
                if not total_entry_series.empty:
                    last_total_entry = total_entry_series.iloc[-1]
                    if last_total_entry > 100:
                        print(f"DROPPED: total_entry={last_total_entry:.2f} > 100")
                        dropped = True
                
                if dropped:
                    continue

                final_balance = all_results['total_balance'].iloc[-1]
                results_matrix[(long_gap_idx, short_gap_idx)] = final_balance
                print(f"Final Balance: {final_balance:.2f}")
        
        print("\n" + "="*80)
        print("Finding best combination...")
        print("="*80)
        
        if not results_matrix:
            raise ValueError("All combinations dropped!")

        matrix = np.zeros((len(gap_values), len(gap_values)))
        for (long_idx, short_idx), balance in results_matrix.items():
            matrix[long_idx, short_idx] = balance
        
        for i in range(len(gap_values)):
            for j in range(len(gap_values)):
                if (i, j) not in results_matrix:
                    matrix[i, j] = np.nan
        
        best_idx = np.unravel_index(np.nanargmax(matrix), matrix.shape)
        best_long_idx = best_idx[0]
        best_short_idx = best_idx[1]
        best_long_gap = gap_values[best_long_idx]
        best_short_gap = gap_values[best_short_idx]
        best_balance = matrix[best_idx[0], best_idx[1]]
        
        print(f"\n✓ BEST COMBINATION FOUND!")
        print(f"  Long Entry Gap: {gap_labels[best_long_idx]}")
        print(f"  Short Entry Gap: {gap_labels[best_short_idx]}")
        print(f"  Final Balance: {best_balance:.2f}")
        print("="*80)
        
        # Final run
        print(f"\nGenerating final results...")
        all_results, event_results = simulated_trade(qqq_data, long_entry_gap=best_long_gap, short_entry_gap=best_short_gap)
        
        all_results = all_results.drop_duplicates(subset=['Datetime'], keep='first')
        event_results = event_results.drop_duplicates(subset=['Datetime'], keep='first')
        
        optimal_all_csv = f"qqq_backtest_optimal_longgap{gap_labels[best_long_idx]}_shortgap{gap_labels[best_short_idx]}_all.csv"
        optimal_event_csv = f"qqq_backtest_optimal_longgap{gap_labels[best_long_idx]}_shortgap{gap_labels[best_short_idx]}_results.csv"
        all_results.to_csv(optimal_all_csv, index=False)
        event_results.to_csv(optimal_event_csv, index=False)
        print(f"Saved: {optimal_all_csv}")

        from datetime import datetime
        now = datetime.now().strftime("_%y%m%d_%H%M%S")
        remote_filename = f"qqq_backtest_optimal_longgap{gap_labels[best_long_idx]}_shortgap{gap_labels[best_short_idx]}_all{now}.csv"
        try:
            subprocess.run(["rclone", "copyto", optimal_all_csv, f"BackTest:{remote_filename}"], check=True)
            print(f"Uploaded: {remote_filename}")
        except:
            print("rclone upload skipped")
        
        # Heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap='RdYlGn', aspect='auto', origin='lower')
        plt.colorbar(label='Final Balance')
        plt.xlabel('Short Entry Gap')
        plt.ylabel('Long Entry Gap')
        plt.title(f'Price Gap Optimization\nBest: Long={gap_labels[best_long_idx]}, Short={gap_labels[best_short_idx]}, Balance={best_balance:.2f}')
        plt.xticks(range(len(gap_labels)), gap_labels)
        plt.yticks(range(len(gap_labels)), gap_labels)
        plt.plot(best_idx[1], best_idx[0], 'b*', markersize=25)
        
        for (long_idx, short_idx), balance in results_matrix.items():
            plt.text(short_idx, long_idx, f'{balance:.0f}', ha='center', va='center', fontsize=9, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('optimization_heatmap_pricegap.png', dpi=150)
        print("Heatmap saved: optimization_heatmap_pricegap.png")
        plt.show()
        
        # Performance metrics
        equity_curve = all_results['total_balance'].tolist()
        equity = pd.Series(equity_curve)
        returns = equity.pct_change().dropna()
        
        mdd = ((equity.cummax() - equity) / equity.cummax()).max()
        win_rate = (returns > 0).mean()
        
        print("\n최종 수익률:", f"{(equity.iloc[-1] / equity.iloc[0] - 1) * 100:.2f}%")
        print("MDD:", f"{mdd * 100:.2f}%")
        print("승률:", f"{win_rate * 100:.2f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
