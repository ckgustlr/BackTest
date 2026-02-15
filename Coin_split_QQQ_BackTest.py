#-*-coding:utf-8-*-
from pickle import TRUE
import subprocess
import time
import datetime
from datetime import datetime, timezone, timedelta 
import requests
import json
import ccxt
from telegram import Bot
import asyncio
import random
import hashlib
import hmac
from urllib.parse import urlparse
import numpy as np
from pandas import DataFrame
import sqlite3
#import dbinsert
import myutil2
import base64, hashlib, hmac, json, requests, time
import sys
import pandas as pd
import client
import math
import pprint
import bitget.mix.market_api as market
import bitget.mix.account_api as accounts
import bitget.mix.position_api as position
import bitget.mix.order_api as order
import bitget.mix.plan_api as plan
import bitget.mix.trace_api as trace
import bitget.mix.plan_api as plan
import bitget.spot.wallet_api as wallet
import json
import os
import math
from currency_converter import CurrencyConverter
import threading
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Load hyperparameters from environment variables
LEVERAGE_MIN = int(os.getenv('LEVERAGE_MIN', 10))
LEVERAGE_MAX = int(os.getenv('LEVERAGE_MAX', 50))
SHORT_SAFE_MARGIN = float(os.getenv('SHORT_SAFE_MARGIN', 0.9))
LONG_SAFE_MARGIN = float(os.getenv('LONG_SAFE_MARGIN', 1.1))
BET_SIZE_FACTOR = float(os.getenv('BET_SIZE_FACTOR', 0.01))

position_side = sys.argv[1]
account = sys.argv[2]
coin = sys.argv[3]

# Replace hardcoded API credentials with environment variables
api_key = os.getenv('API_KEY')
secret_key = os.getenv('SECRET_KEY')
passphrase = os.getenv('PASSPHRASE')

# Replace hardcoded Telegram credentials with environment variables
my_token = os.getenv('TELEGRAM_BOT_TOKEN')
chat_id = os.getenv('TELEGRAM_CHAT_ID')

# Ensure the API credentials are loaded
if not api_key or not secret_key or not passphrase:
    raise ValueError("Missing API_KEY, SECRET_KEY, or PASSPHRASE in .env file.")

# Ensure the Telegram credentials are loaded
if not my_token or not chat_id:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env file.")

TOKEN = my_token
CHAT_ID = chat_id
bot = Bot(token=TOKEN)

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.302 Safari/537.36'}

def tg_send(text):
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": str(text)}, timeout=10)
    except Exception as e:
        # Avoid crashing trading logic if Telegram fails
        print('Telegram send failed:', type(e).__name__, e)

def exit_alarm_enable(avg_price,close_price,side):
    if close_price != avg_price and avg_price != None:
        if side == "long":
            if avg_price !=0:
                profit=(close_price - avg_price)/avg_price*100
                return profit
            else:
                return 0
        elif side == "short":
            if avg_price !=0:
                profit=(avg_price - close_price)/avg_price*100
                return profit
            else:
                return 0
    else:
        return 0

def read_json(filename):
    lock = threading.Lock()
    lock.acquire()
    with open(filename) as f:
        try:
            live24data = json.load(f)
        except:
            return 0
    lock.release()
    return live24data

def get_pos_index(live24,symbol,position_side):
    for i in range(len(live24['data'])+1):
    #for i in range(20):
        if live24['data'][i]['symbol'] == symbol and live24['data'][i]['holdSide'] == position_side:
            return i
            break;

def return_true_after_minutes(minute,timestamp):
    target_timestamp = timestamp + (minute * 60)  # n분 후의 타임스탬프 계산
    if time.time() > target_timestamp:
        ret=1
        diff= target_timestamp-time.time()
    else:
        ret=0
        diff= target_timestamp-time.time()
    return ret,diff

def calculate_leverage(absamount_gap, free_lev):
    set_lev = float(absamount_gap) * float(leverage) / free_lev
    if set_lev > LEVERAGE_MAX:
        set_lev = LEVERAGE_MAX
    elif set_lev < LEVERAGE_MIN:
        set_lev = LEVERAGE_MIN
    return int(set_lev)

# Refactor entry logic
def entry_logic(position_side, liquidation_price, close_price, margin, accountApi, symbol, marginC):
    safe_margin = SHORT_SAFE_MARGIN if position_side == 'short' else LONG_SAFE_MARGIN
    if liquidation_price * safe_margin < close_price:
        try:
            cal_amount = round(margin * 0.08)
            message = f"entry>>[{account}][{position_side}]liquidationPrice*{safe_margin}:{liquidation_price * safe_margin}<close_price:{close_price}+8%:{cal_amount}"
            result = accountApi.margin(symbol, marginCoin=marginC, productType='USDT-FUTURES', amount=cal_amount, holdSide=position_side)
            print(message)
        except:
            pass  # Handle other margin levels similarly

# Refactor exit logic
def exit_logic(position_side, liquidation_price, close_price, margin, accountApi, symbol, marginC):
    safe_margin = SHORT_SAFE_MARGIN if position_side == 'short' else LONG_SAFE_MARGIN
    if liquidation_price * safe_margin > close_price:
        try:
            cal_amount = round(margin * 0.08)
            message = f"exit>>[{account}][{position_side}]liquidationPrice*{safe_margin}:{liquidation_price * safe_margin}>close_price:{close_price}-8%:{cal_amount}"
            result = accountApi.margin(symbol, marginCoin=marginC, productType='USDT-FUTURES', amount=cal_amount * -1, holdSide=position_side)
            print(message)
        except:
            pass  # Handle other margin levels similarly

if __name__ == "__main__":
    cnt=0
    cntm=0
    minute=0
    minutem=0
    pre_count=0
    pre_set_lev = 0
    long_flag = False
    short_flag = False
    if coin == 'QQQUSDT':
        symbol = 'QQQUSDT'
        productType = 'USDT-FUTURES'
        marginC = 'USDT'
        productT='umcbl'

    filename  = account+'.json'

    filename2  = coin+'_'+account+'.json'
    filename6  = 'sanfran6_main_sol.json'

    c = ccxt.bitget({'apiKey' : api_key,'secret' : secret_key, 'password' : passphrase, 'options':{'defaultType':'swap'}})
    marketApi = market.MarketApi(api_key, secret_key, passphrase, use_server_time=False, first=False)
    positionApi = position.PositionApi(api_key, secret_key, passphrase, use_server_time=False, first=False)
    orderApi = order.OrderApi(api_key, secret_key, passphrase, use_server_time=False, first=False)
    accountApi = accounts.AccountApi(api_key, secret_key, passphrase, use_server_time=False, first=False)
    planApi = plan.PlanApi(api_key, secret_key, passphrase, use_server_time=False, first=False)
    indexname = account+'_'+coin+'_'+position_side

    # api_key = "bg_fea299f16fcf4b76d7c996a8577e09ae"
    # secret_key = "abd8c395d31dcbb2a5cf59ff9909e33f5ec7fa58145e729b07e07f3445359e6e"
    # passphrase = "1qazxsw2"  # Password
    walletApi = wallet.WalletApi(api_key, secret_key, passphrase, use_server_time=False, first=False)

    while True:
        close_price = float(marketApi.ticker('QQQUSDT','USDT-FUTURES')['data'][0]['lastPr'])
        chgUtc = float(marketApi.ticker('QQQUSDT','USDT-FUTURES')['data'][0]['changeUtc24h'])*100
        chgUtcWoAbs = chgUtc
        balances = accountApi.account('QQQUSDT','USDT-FUTURES', marginCoin=marginC)
        free = float(balances['data']['available'])
        total = float(balances['data']['usdtEquity'])
        total_div4 = round(float(balances['data']['usdtEquity'])/2,1)  # temporary 이더 최소 분해능이 안나온다 테스트후 2->4 원복 예정
        Forty_percent = round(total_div4/close_price/40,8)
        Twenty_percent = round(total_div4/close_price/20,8)
        amount=round(total/close_price,8)
        freeamount=round(free/close_price,8)
        cnt=cnt+1
        cntm=cntm+1
        live24data_condition = read_json(filename2)
        if live24data_condition:
            live24data = live24data_condition
            live24data_backup=live24data
        else:
            #pass
            live24data=live24data_backup
        condition = read_json(filename)
        if condition:
            live24 = condition
            live24_backup=live24
        else:
            live24 = live24_backup
        condition = read_json('sanfran6.json')
        if condition:
            live = condition
            live_backup=live
        else:
            live = live_backup
        condition = read_json(filename6)
        if condition:
            livea = condition
            live_backup=livea
        position = positionApi.all_position(marginCoin='USDT', productType='USDT-FUTURES')
        short_profit = live24data['short_profit'] #1.001 #live24data['long_take_profit']
        long_profit = live24data['long_profit'] #0.999 #live24data['short_take_profit']
        long_take_profit = live24data['long_take_profit'] #1.001 #live24data['long_take_profit']
        short_take_profit = live24data['short_take_profit'] #0.999 #live24data['short_take_profit']
        try:
            idx = get_pos_index(position,'QQQUSDT',position_side)
        except:
             pass

        position = positionApi.all_position(marginCoin='USDT', productType='USDT-FUTURES')['data'][idx]
        liquidationPrice=round(float(position['liquidationPrice']),1)
        breakeven = round(float(position['breakEvenPrice']),1)

        unrealizedPnl=round(float(position['unrealizedPL']),1)
        #print(position)
        achievedProfits=round(float(position['achievedProfits']),1)
        avg_price = round(float(position['openPriceAvg']),3)
 #       print("close_price:{}/avg_price:{}".format(close_price,avg_price))
        absamount = float(position['available'])

        short_gap = abs(close_price-live24data['short_avg_price'])
        long_gap = abs(close_price-live24data['long_avg_price'])

        symbol2 = 'qqqusdt'
        leverage = float(position["leverage"])
        absamount_gap = abs(live24data['short_absamount']-live24data['long_absamount'])
        free_lev = float(freeamount) * float(leverage)
        short_lev = float(live24data['short_absamount']) * float(leverage)
        long_lev = float(live24data['long_absamount']) * float(leverage)
        #print("freeamount:{}/short:{}/long:{}/leverage:{}".format(freeamount,short_lev,long_lev,leverage))
        set_lev = calculate_leverage(absamount_gap, free_lev)
        if set_lev > 50:
           set_lev = 50
        elif set_lev < 10:
           set_lev = 10
        set_lev = int(set_lev)
        #print("set_lev:{}".format(set_lev))
#        set_lev = 49
        if live24data['short_absamount'] > live24data['long_absamount']:
#            print("short_absamount:{}/long_absamount:{}/gap:{}/free:{}-> set_lev:{}".format(live24data['short_absamount'],live24data['long_absamount'],absamount_gap,free_lev,set_lev))
#            print("short_gap:{} > long_gap:{}".format(short_gap,long_gap))
            if pre_set_lev != set_lev:
                message = "[{}][{}]set_leverage long {}x/short 10x".format(account,symbol2,set_lev)
                try:
                    result = accountApi.leverage_v3(symbol2,  productType = "USDT-FUTURES", marginCoin='USDT', longLeverage=set_lev, shortLeverage='10')
                except:
                    pass
#                tg_send(message)
                pre_set_lev = set_lev

        else:
            if pre_set_lev != set_lev:
                message = "[{}][{}]set_leverage short {}x/long 10x".format(account,symbol2,set_lev)
                try:
                    result = accountApi.leverage_v3(symbol2,  productType = "USDT-FUTURES", marginCoin='USDT', longLeverage='10', shortLeverage=set_lev)
                except:
                    pass
 #               tg_send(message)
                pre_set_lev = set_lev

        avg_break_gap= exit_alarm_enable(breakeven,avg_price,position_side)  #temporary Failure(8/10)
        profit=exit_alarm_enable(avg_price,close_price,position_side)
        margin = round(float(position['marginSize']),1)
        utilzation = round(livea['Main_SOLUSDT_long']['utilrate'],1)
        default_size = round(total_div4*2/(close_price/4)/20,1)
        margin_rate = margin/total_div4*100
        utilzation2 = round(margin/total_div4*100,1)
        breakeven = round(float(position['breakEvenPrice']),5)

        if live24data_condition !=0:
            total_absamount = (live24data['short_absamount']+live24data['long_absamount']+freeamount*float(leverage))
            half_absamount = total_absamount/2
            short_rate = (live24data['short_absamount']/half_absamount)*100
            long_rate = (live24data['long_absamount']/half_absamount)*100
            total_absamount_100unit = round(total_absamount/100,1)
            total_absamount_rate = (live24data['short_absamount']+live24data['long_absamount'])/total_absamount
            gap_cal = 0.001+0.003*total_absamount_rate
            if total_absamount_100unit < 0.01:
                total_absamount_100unit = 0.01
            try:
                margin_rate2=round(margin/total_div4*100,1)
            except:
                margin_rate2=0

            avg_break_gap= exit_alarm_enable(breakeven,avg_price,position_side)  #temporary Failure(8/10)
            profit=exit_alarm_enable(avg_price,close_price,position_side)
            close_price = float(marketApi.ticker('QQQUSDT','USDT-FUTURES')['data'][0]['lastPr'])
            currency =CurrencyConverter('http://www.ecb.europa.eu/stats/eurofxref/eurofxref.zip')
            usd_krw=round(currency.convert(1,'USD','KRW'),1)

            chgUtc = abs(chgUtc)
            bet_sizex = round(freeamount*leverage)

            avg_gap = live24data['long_avg_price']-live24data['short_avg_price']
            gap_middle = (live24data['long_avg_price']+live24data['short_avg_price'])/2
            gap_percent = avg_gap/(close_price/100)
            set_pull = (gap_percent+0.5)*(-1)
            free_cnt = round(freeamount*leverage*10000)
            ShortSafeMargin = 0.9
            LongSafeMargin = 1.1

            if position_side == 'short':
                if liquidationPrice*ShortSafeMargin<close_price:
                    if free >1:
                        entry_logic('short', liquidationPrice, close_price, margin, accountApi, symbol, marginC)
                    else:  # 마진이 없으면 일단, 손절후, 마진 투입
                        try:
                            print("entry>>liquidationPrice*0.9:{}<close:{}".format(liquidationPrice*0.9,close_price))
                            result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")
                            sell_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'sell' and entry['symbol'] == 'QQQUSDT']
                            sorted_sell_orders = sorted(sell_orders, key=lambda x: float(x['triggerPrice']),reverse=True)[0]
                            trg_price = round(close_price*0.9998,1) # 2025-08-31
                            result = planApi.modify_tpsl_plan_v2(symbol="qqqusdt", marginCoin="USDT", productType="USDT-FUTURES", orderId=sorted_sell_orders['orderId'], triggerPrice=trg_price,executePrice=trg_price,size=sorted_sell_orders['size'])
                            myutil2.live24flag('highest_short_price',filename2,trg_price)
                            message = "[{}][liquidationPrice*0.98:{}<close_price:{}][short:{}/free:{}USD/long:{}][{}][achievedProfits:{}>0]/lowest_triggerPrice:{}/avg_price:{} -> modifytpsl:{}/size:{}".format(account,liquidationPrice*0.98,close_price,live24data['short_absamount'],free,live24data['long_absamount'],position_side,achievedProfits,sorted_sell_orders['triggerPrice'],avg_price,trg_price,sorted_sell_orders['size'])
                            time.sleep(8)
                        except:
                            pass
                else:
                    pass
                bet_size = 0.01*bet_sizex
                print("absamount:{} <= bet_size{}".format(absamount,bet_size))          

                if absamount <= bet_size and long_profit > 0.5:
                    orderApi.place_order(symbol, marginCoin=marginC, size=bet_size,side='sell', tradeSide='open', marginMode='isolated',  productType = "USDT-FUTURES", orderType='market', price=close_price, clientOrderId='sanfran6@'+str(int(time.time()*100)), presetStopSurplusPrice=round(close_price*0.95,1), timeInForceValue='normal')
                    message="[{}]Market Short Entry/absamount:{}/bet_size:{}/achievedProfits:{}".format(account,absamount,bet_size,achievedProfits)
                    tg_send(message)
                    time.sleep(30)

                print("highest_short_price*(1+{}:{}):{}<close_price:{}".format(live24data['short_gap_rate'],1+live24data['short_gap_rate'],live24data['highest_short_price']*(1+live24data['short_gap_rate']),close_price))
                if float(live24data['highest_short_price'])*(1+live24data['short_gap_rate'])<close_price:
                    if long_profit > 0.5 or short_profit < -1:
                        try:
                            orderApi.place_order(symbol, marginCoin=marginC, size=bet_size,side='sell', tradeSide='open', marginMode='isolated',  productType = "USDT-FUTURES", orderType='limit', price=close_price, clientOrderId='sanfran6@'+str(int(time.time()*100)), presetStopSurplusPrice=round(close_price*short_take_profit,1), timeInForceValue='normal')
                            myutil2.live24flag('short_recovery_enable',filename2,True)
                            myutil2.live24flag('short_recovery_size',filename2,bet_size)
                            myutil2.live24flag('short_entry_price',filename2,close_price)
                            time.sleep(5)
                            result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")
                            sell_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'sell' and entry['symbol'] == 'QQQUSDT']
                            sorted_sell_orders = sorted(sell_orders, key=lambda x: float(x['triggerPrice']),reverse=True)[0]
                            print("short entry/triggerPrice:{}".format(sorted_sell_orders['triggerPrice']))
                            myutil2.live24flag('highest_short_price',filename2,float(close_price))
                            profit=exit_alarm_enable(avg_price,close_price,position_side)
                        except:
                            message="[free:{}][{}_{}_{}][{}][{}][size:{}]물량투입 실패:{}USD->cancel all orders".format(free,account,coin,position_side,close_price,profit,round(bet_size,8),total_div4)
                        minutem=0
                    if live24data['short_entry_price'] > close_price:  # 최고점이 낮아지면 진입 기준점이 낮아진다
                        message="short[{}][entry_price:{} > close_price :{}]".format(account,live24data['short_entry_price'],close_price)
                        myutil2.live24flag('short_entry_price',filename2,close_price)
                        time.sleep(1)

            elif position_side == 'long':  # 포지션 롱일때
                if liquidationPrice*LongSafeMargin>close_price:
                    if free >1:
                        entry_logic('long', liquidationPrice, close_price, margin, accountApi, symbol, marginC)
                    else:
                        try:
                            print("[freeless]entry>>liquidationPrice:{}*1.1:{}>close:{}".format(liquidationPrice,liquidationPrice*1.1,close_price))
                            result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")
                            sell_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'buy' and entry['symbol'] == 'QQQUSDT']
                            sorted_sell_orders = sorted(sell_orders, key=lambda x: float(x['triggerPrice']),reverse=False)[0]
                            trg_price = round(close_price*1.0002,1) #2025-08-31
                            result = planApi.modify_tpsl_plan_v2(symbol="qqqusdt", marginCoin="USDT", productType="USDT-FUTURES", orderId=sorted_sell_orders['orderId'], triggerPrice=trg_price,executePrice=trg_price,size=sorted_sell_orders['size'])
                            myutil2.live24flag('lowest_long_price',filename2,trg_price)
                            message = "[freeless][{}][liquidationPrice*1.1:{}>close_price:{}][short:{}/free:{}USD/long:{}][{}][achievedProfits:{}>0]/lowest_triggerPrice:{}/avg_price:{} -> modifytpsl:{}/size:{}".format(account,liquidationPrice*1.1,close_price,live24data['short_absamount'],free,live24data['long_absamount'],position_side,achievedProfits,sorted_sell_orders['triggerPrice'],avg_price,trg_price,sorted_sell_orders['size'])
                            tg_send(message)
                            time.sleep(8)
                        except:
                            pass
                else:
                    pass
                bet_size = 0.01*bet_sizex
             

                if absamount <= bet_size and short_profit > 0.5:
                    orderApi.place_order(symbol, marginCoin=marginC, size=bet_size,side='buy', tradeSide='open', marginMode='isolated', productType = "USDT-FUTURES", orderType='market', price=close_price, clientOrderId='sanfran6@'+str(int(time.time()*100)), presetStopSurplusPrice=round(close_price*(1.01-(divxfactor*x)),1), timeInForceValue='normal')
                    message="[{}]Market Long Entry/absamount:{}/bet_size:{}/presetTakeProfitPrice:{}".format(account,absamount,bet_size,round(close_price*1.05,1))
                    tg_send(message)
                    time.sleep(30)

                print("lowest_long_price*(1-{}:{}):{}>close_price:{}".format(live24data['long_gap_rate'],1-live24data['long_gap_rate'],live24data['lowest_long_price']*(1-live24data['long_gap_rate']),close_price))
                if float(live24data['lowest_long_price'])*(1-live24data['long_gap_rate'])>close_price:
                    if free > 1:
                        if (live24data['buy_orders_count']<2 and long_profit > 0)  or short_profit > 0.5 or long_profit < -1:
                            try:
                                orderApi.place_order(symbol, marginCoin=marginC, size=bet_size,side='buy', tradeSide='open', marginMode='isolated',  productType = "USDT-FUTURES", orderType='limit', price=close_price, clientOrderId='sanfran6@'+str(int(time.time()*100)), timeInForceValue='normal',presetStopSurplusPrice=round(close_price*long_take_profit,1))
                                myutil2.live24flag('long_recovery_enable',filename2,True)
                                myutil2.live24flag('long_recovery_size',filename2,bet_size)
                                myutil2.live24flag('long_entry_price',filename2,close_price)
                                myutil2.live24flag('low_entry_price',filename2,close_price)
                                time.sleep(5)
                                result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")
                                buy_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'buy' and entry['symbol'] == 'QQQUSDT']
                                sorted_buy_orders = sorted(buy_orders, key=lambda x: float(x['triggerPrice']))[0]
                                myutil2.live24flag('lowest_long_price',filename2,float(close_price))
                                profit=exit_alarm_enable(avg_price,close_price,position_side)
                            except:
                                message="[free:{}][{}_{}_{}][{}][{}][size:{}]물량투입 실패:{}USD->cancel all orders".format(free,account,coin,position_side,close_price,profit,round(bet_size,8),total_div4)
                                minutem=0
                            if live24data['long_entry_price'] < close_price:   # 최저점이 높아지면 진입 기준점이 높아진다
                                message="long[entry_price:{} < close_price :{}]".format(account,live24data['long_entry_price'],close_price)
                                myutil2.live24flag('long_entry_price',filename2,close_price)
                                time.sleep(1)


            if cnt%10 ==0:
                print("cnt:{}/chgUtc:{}/long:{} vs short:{} / free:{} < margin * 0.01:{}".format(cnt,chgUtc,long_take_profit,short_take_profit,free,margin * 0.01))
                if position_side == 'short':
                    if liquidationPrice*0.95<close_price:
                        entry_logic('short', liquidationPrice, close_price, margin, accountApi, symbol, marginC)
                        message = "liquidationPrice:{}*0.95={}<close_price:{}".format(liquidationPrice,liquidationPrice*0.95,close_price)
                        print(message)
                        tg_send(message)
                elif position_side == 'long':
                    if liquidationPrice*1.05>close_price:
                        entry_logic('long', liquidationPrice, close_price, margin, accountApi, symbol, marginC)
                        message = "liquidationPrice:{}*1.05={}>close_price:{}".format(liquidationPrice,liquidationPrice*1.05,close_price)
                        print(message)
                        tg_send(message)

            if cnt%30 ==0:
                if position_side == 'short':
                    myutil2.live24flag('short_absamount',filename2,absamount)
                    myutil2.live24flag('short_avg_price',filename2,avg_price)
                    myutil2.live24flag('short_profit',filename2,profit)
                    result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")
                    sell_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'sell' and entry['symbol'] == 'QQQUSDT']
                    myutil2.live24flag('sell_orders_count',filename2,len(sell_orders))
                    sorted_sell_orders = sorted(sell_orders, key=lambda x: float(x['triggerPrice']))[0]
                    myutil2.live24flag('lowest_short_price',filename2,float(sorted_sell_orders['triggerPrice']))
                    sorted_sell_orders = sorted(sell_orders, key=lambda x: float(x['triggerPrice']),reverse=True)[0]
                    myutil2.live24flag('highest_short_price',filename2,float(sorted_sell_orders['triggerPrice']))
                    if liquidationPrice*ShortSafeMargin>close_price:
                        exit_logic('short', liquidationPrice, close_price, margin, accountApi, symbol, marginC)
                elif position_side == 'long':
                    myutil2.live24flag('long_absamount',filename2,absamount)
                    myutil2.live24flag('long_avg_price',filename2,avg_price)
                    myutil2.live24flag('long_profit',filename2,profit)
                    result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")
                    buy_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'buy' and entry['symbol'] == 'QQQUSDT']
                    myutil2.live24flag('buy_orders_count',filename2,len(buy_orders))
                    sorted_buy_orders = sorted(buy_orders, key=lambda x: float(x['triggerPrice']),reverse=True)[0]
                    myutil2.live24flag('highest_long_price',filename2,float(sorted_buy_orders['triggerPrice']))
                    sorted_buy_orders = sorted(buy_orders, key=lambda x: float(x['triggerPrice']))[0]
                    myutil2.live24flag('lowest_long_price',filename2,float(sorted_buy_orders['triggerPrice']))
                    print(message)
                    if liquidationPrice*LongSafeMargin<close_price:
                        exit_logic('long', liquidationPrice, close_price, margin, accountApi, symbol, marginC)
            if cnt%60 ==0:
                print("long_avg_price:{}-short_avg_price:{}={}/{}%".format(live24data['long_avg_price'],live24data['short_avg_price'],avg_gap,gap_percent))
                print("iquidationPrice:{}".format(liquidationPrice))
                available_absamount_cnt = abs(live24data['short_absamount']-live24data['long_absamount'])/0.01
                if available_absamount_cnt < free_cnt:
                    available_absamount_cnt = free_cnt
                elif available_absamount_cnt == 0:
                    if position_side == 'short':
                        available_absamount_cnt = live24data['short_absamount']/0.01
                    elif position_side == 'long':
                        available_absamount_cnt = live24data['long_absamount']/0.01
                split_cnt = round(available_absamount_cnt/bet_sizex)
                if position_side == 'short':
                    price_gap = live24data['highest_long_price'] - close_price
                    unit_gap = price_gap / split_cnt
                    gap_rate = unit_gap / close_price
                    print("split_cnt:{}/highest_long_price:{}-close_price:{}=price_gap:{}/unit_gap:{}/gap_rate:{}".format(split_cnt,live24data['highest_long_price'],close_price,price_gap,unit_gap,gap_rate))
                    base_rate = 0.0025
                    short_take_profit0= 1-base_rate
                    myutil2.live24flag('short_take_profit',filename2,short_take_profit0)
                    gap_rate = 0.0001*live24data['sell_orders_count']
                    gap_rate = base_rate+gap_rate
                    print("sell_orders_count(21부터):{}/gap_rate:{}".format(live24data['sell_orders_count'],gap_rate))
                    #print("profit:{}/chgUTC:{}/base_rate:{}/short_gap_rate:{}".format(profit,chgUtc,base_rate,gap_rate))
                    myutil2.live24flag('short_gap_rate',filename2,gap_rate)
                    myutil2.live24flag('short_liquidationPrice',filename2,liquidationPrice)
                elif position_side == 'long':
                    price_gap = close_price - live24data['lowest_short_price']
                    unit_gap = price_gap / split_cnt
                    gap_rate = unit_gap / close_price
                    print("split_cnt:{}/close_price:{}-lowest_short_price:{}=price_gap:{}/unit_gap:{}/gap_rate:{}".format(split_cnt,close_price,live24data['lowest_short_price'],price_gap,unit_gap,gap_rate))
                    base_rate = 0.0025                        
                    long_take_profit0= 1+base_rate
                    myutil2.live24flag('long_take_profit',filename2,long_take_profit0)
                    gap_rate = 0.0001*live24data['buy_orders_count']
                    gap_rate = base_rate+gap_rate
                    print("buy_orders_count(21부터):{}/gap_rate:{}".format(live24data['buy_orders_count'],gap_rate))
                    #print("profit:{}/chgUTC:{}/base_rate:{}/long_gap_rate:{}".format(profit,chgUtc,base_rate,gap_rate))
                    print("long_gap_rate:{}".format(gap_rate))
                    myutil2.live24flag('long_gap_rate',filename2,gap_rate)
                    myutil2.live24flag('long_liquidationPrice',filename2,liquidationPrice)


                print("total_absamount write here")
                myutil2.flagwithfile('SOLUSDT.json',account,'total_absamount',total_absamount)
                minutem=minutem+1
                result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")
                orders = [entry for entry in  result['data']['entrustedList']]
                sorted_orders = sorted(orders, key=lambda x: float(x['triggerPrice']))

            if position_side == 'short':
                timeout = 9

                if return_true_after_minutes(timeout,live24data['short_entry_time'])[0]:   # 30분 마다 1% 이익 세팅
                    myutil2.live24flag('short_entry_time',filename2,time.time())
                    result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")

                    sell_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'sell' and entry['symbol'] == 'QQQUSDT']
                    sorted_sell_orders_last = sorted(sell_orders, key=lambda x: float(x['triggerPrice']),reverse=False)[0]
                    first_size=sorted(sell_orders, key=lambda x: float(x['size']),reverse=False)[0]
                    last_size=sorted(sell_orders, key=lambda x: float(x['size']),reverse=False)[-1]


                    sorted_sell_orders_last_price = round(avg_price*0.98,1) #라오어 패치, 최소 이익실현은 10% 부터 시작(10배레버리지의 1% 이익지점)
                    sorted_sell_orders_last_price_1 = avg_price*0.98 #라오어 패치, 최소 이익실현은 10% 부터 시작(10배레버리지의 1% 이익지점)

                    if live24data['short_absamount'] <= live24data['long_absamount']:
                        profit_line = 0.99
                    else:
                        profit_line = 0.995
                    if avg_price*profit_line < close_price:
                        trigger_price0 = avg_price*profit_line
                    else:
                        trigger_price0 = close_price*0.9998  #0.998 <- 0.9998 2025-11-26

                    sell_orders_gap = abs(trigger_price0-sorted_sell_orders_last_price_1)  #0.998 -> 0.9998 2025-11-19
                    sell_orders_unitgap = sell_orders_gap/(live24data['sell_orders_count']+1)

                    for i in range(len(sell_orders)):
                        sorted_sell_orders = sorted(sell_orders, key=lambda x: float(x['size']),reverse=False)[i]
                        sorted_price_sell_orders = sorted(sell_orders, key=lambda x: float(x['triggerPrice']),reverse=True)[i]
                        if profit > 0:
                            if i == 0:
                                trigger_price = sorted_sell_orders_last_price
                            else:
                                trigger_price = str(round(trigger_price0 - (sell_orders_unitgap*(i-1)),1))                               
                            print("[{}/{}][{}/{}][{}/{}]".format(i,sorted_sell_orders['size'],trigger_price,type(trigger_price).__name__,sorted_sell_orders['triggerPrice'],type(sorted_sell_orders['triggerPrice']).__name__))
                        else:
                            if i == 0:
                                trigger_price = sorted_price_sell_orders['triggerPrice']
                            else:
                                trigger_price = str(round(trigger_price0 - (sell_orders_unitgap*(i-1)),1))
                            print("[{}/{}][{}/{}][{}]".format(i,sorted_sell_orders['size'],trigger_price,type(trigger_price).__name__,sorted_sell_orders['triggerPrice']))
                        result = planApi.modify_tpsl_plan_v2(symbol="qqqusdt", marginCoin="USDT", productType="USDT-FUTURES", orderId=sorted_sell_orders['orderId'], triggerPrice=trigger_price,executePrice=trigger_price,size=sorted_sell_orders['size'])
                        time.sleep(1)
                    if pre_count != live24data['sell_orders_count']:
                        if profit > 0:
                            message = "[Short timeout:{}/count:{}][{}/{}]trigger_price:{}/gap:{}/last:{}]".format(timeout,i,live24data['short_absamount'],achievedProfits,round(trigger_price0,1),round(sell_orders_unitgap),round(sorted_sell_orders_last_price_1))
                            tg_send(message)
                            pre_count = live24data['sell_orders_count']
                else:
                    print("short 최저점 조정 남은 시간:{}".format(return_true_after_minutes(timeout,live24data['short_entry_time'])[1]))


            elif position_side == 'long':
                timeout = 9
                if return_true_after_minutes(timeout,live24data['long_entry_time'])[0]:
                    myutil2.live24flag('long_entry_time',filename2,time.time())
                    result = planApi.current_plan_v2(planType="profit_loss", productType="USDT-FUTURES")

                    buy_orders = [entry for entry in  result['data']['entrustedList'] if entry['side'] == 'buy' and entry['symbol'] == 'QQQUSDT']
                    sorted_buy_orders_last = sorted(buy_orders, key=lambda x: float(x['triggerPrice']),reverse=True)[0]
                    
                    sorted_buy_orders_last_price = round(avg_price*1.033,1) #라오어 패치, 최소 이익실현은 10% 부터 시작(10배레버리지의 1% 이익지점)
                    sorted_buy_orders_last_price_1 = avg_price*1.033 #라오어 패치, 최소 이익실현은 10% 부터 시작(10배레버리지의 1% 이익지점)

                    if live24data['short_absamount'] >= live24data['long_absamount']:
                        profit_line = 1.01
                    else:
                        profit_line = 1.005
                    if avg_price*profit_line > close_price:
                        trigger_price0 = avg_price*profit_line
                    else:
                        trigger_price0 = close_price*1.0002  #0.998 <- 0.9998 2025-11-26

                    buy_orders_gap = abs(trigger_price0-sorted_buy_orders_last_price_1)  #1.002 2025-11-19
                    buy_orders_unitgap = buy_orders_gap/(live24data['buy_orders_count']+1)

                    for i in range(len(buy_orders)):
                        sorted_buy_orders = sorted(buy_orders, key=lambda x: float(x['size']),reverse=False)[i]
                        sorted_price_buy_orders = sorted(buy_orders, key=lambda x: float(x['triggerPrice']),reverse=False)[i]
                        if profit > 0:
                            if i ==0:
                                trigger_price = sorted_buy_orders_last_price
                            else:
                                trigger_price = str(round(trigger_price0 + (buy_orders_unitgap*(i-1)),1))
                            print("[{}/{}][{}/{}][{}/{}]".format(i,sorted_buy_orders['size'],trigger_price,type(trigger_price).__name__,sorted_buy_orders['triggerPrice'],type(sorted_buy_orders['triggerPrice']).__name__))
                        else:
                            if i ==0:
                                trigger_price = sorted_price_buy_orders['triggerPrice']
                            else:
                                trigger_price = str(round(trigger_price0 + (buy_orders_unitgap*(i-1)),1))                           
                            print("[{}/{}][{}/{}][{}]".format(i,sorted_buy_orders['size'],trigger_price,type(trigger_price).__name__,sorted_buy_orders['triggerPrice']))
                        result = planApi.modify_tpsl_plan_v2(symbol="qqqusdt", marginCoin="USDT", productType="USDT-FUTURES", orderId=sorted_buy_orders['orderId'], triggerPrice=trigger_price,executePrice=trigger_price,size=sorted_buy_orders['size'])
                        time.sleep(1)
                    if profit > 0:
                        message = "[Long timeout:{}/count:{}][{}/{}]trigger_price:{}/gap:{}/last:{}]".format(timeout,i,live24data['long_absamount'],achievedProfits,round(trigger_price0,1),round(buy_orders_unitgap),round(sorted_buy_orders_last_price_1))
                        tg_send(message)
                else:
                    print("long 최고점 조정 남은 시간:{}".format(return_true_after_minutes(timeout,live24data['long_entry_time'])[1]))


