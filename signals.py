# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 00:28:36 2021

@author: tu madre es una foca
"""

# from selectors import EpollSelector
# import tradingview_ta
# import os
from tradingview_ta import TA_Handler, Interval
import time
from datetime import datetime
import config
from binance.client import Client
from binance.enums import *
import talib as ta
import asyncio
from binance import AsyncClient, BinanceSocketManager
import pandas as pd
import telegram_send

now = datetime.now()
symbolo = "TLMUSDT"
symbol = symbolo
leve = 20
comision = 0.00036

idatei = now.strftime("%d-%m-%y %H:%M:%S")
mylista1 = []
mylista1.insert(0, 1)
mylista1.insert(1, 2)
mylista1.insert(2, 3)
tpl = None
tps = None
sll = None
sls = None
ii = 1

while ii != 0:
    client = Client(config.API_KEY, config.API_SECRET, tld='com')

    print(symbolo)
    markPrice = [None]
    bingo = [None]
    entrada = [None]
    amount = [None]
    leverage = [None]
    # balance=[None]
    cuenta = [None]
    cuenta = client.futures_account_balance()
    print(cuenta)
    balance = cuenta[6]['balance']
    balanceusdt = float(balance)
    print(balanceusdt)
    cantidad = 50
    # print(client.get_symbol_info(symbol=symbolo))
    # elsimbolo=client.get_symbol_info(symbol=symbolo)
    # mincantidad=elsimbolo[0]['minQty']
    # print(mincantidad)

    """
    async def order_book(client, symbol):
        order_book = await client.get_order_book(symbol=symbolo)
        print(order_book)


    async def kline_listener(client):
        bm = BinanceSocketManager(client)
        symbol = symbolo
        res_count = 0
        async with bm.kline_socket(symbol=symbol) as stream:
            while True:
                res = await stream.recv()
                res_count += 1
                print(res)
                if res_count == 5:
                    res_count = 0
                    loop.call_soon(asyncio.create_task, order_book(client, symbol))


    #INTERVALOS: 1HOUR 1MINUTE

    klines = client.get_klines(symbol=symbolo,interval=Client.KLINE_INTERVAL_1MINUTE)
    #start = ["month_1"].values.astype('datetime64[D]')

    df = pd.DataFrame(klines,  columns=['Date',
                                        'Open',
                                        'High',
                                        'Low',
                                        'Close',
                                        'Volume',
                                        'Close time',
                                        'Quote asset volume',
                                        'Number of trades',
                                        'Taker buy base asset volume',
                                        'Taker buy quote asset volume',
                                        'Ignore'])

    ###########################################################

    ########################################################

    ################################################################
    df = df.drop(df.columns[[ 6, 7, 8, 9, 10, 11]], axis=1)
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True, drop=True)
    #data = data.drop(data.columns[0],axis=1)

    o=df['Open']   = df['Open'].astype(float)
    h=df['High']   = df['High'].astype(float)
    l=df['Low']    = df['Low'].astype(float)
    c=df['Close']  = df['Close'].astype(float)
    v=df['Volume'] = df['Volume'].astype(float)

    rATR = ta.ATR(h,l,c,14)
    #print("primer ATR")
    #print(rATR)
    v=rATR.shape
    r=v[0]-1
    datr1m=rATR[r]
    print("atr1m es:", datr1m)


    async def order_book(client, symbol):
        order_book = await client.get_order_book(symbol=symbolo)
        print(order_book)


    async def kline_listener(client):
        bm = BinanceSocketManager(client)
        symbol = symbolo
        res_count = 0
        async with bm.kline_socket(symbol=symbol) as stream:
            while True:
                res = await stream.recv()
                res_count += 1
                print(res)
                if res_count == 5:
                    res_count = 0
                    loop.call_soon(asyncio.create_task, order_book(client, symbol))


    #INTERVALOS: 1HOUR 1MINUTE

    klines = client.get_klines(symbol=symbolo,interval=Client.KLINE_INTERVAL_5MINUTE)
    #start = ["month_1"].values.astype('datetime64[D]')

    df = pd.DataFrame(klines,  columns=['Date',
                                        'Open',
                                        'High',
                                        'Low',
                                        'Close',
                                        'Volume',
                                        'Close time',
                                        'Quote asset volume',
                                        'Number of trades',
                                        'Taker buy base asset volume',
                                        'Taker buy quote asset volume',
                                        'Ignore'])

    ###########################################################

    ########################################################

    ################################################################
    df = df.drop(df.columns[[ 6, 7, 8, 9, 10, 11]], axis=1)
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True, drop=True)
    #data = data.drop(data.columns[0],axis=1)

    o=df['Open']   = df['Open'].astype(float)
    h=df['High']   = df['High'].astype(float)
    l=df['Low']    = df['Low'].astype(float)
    c=df['Close']  = df['Close'].astype(float)
    v=df['Volume'] = df['Volume'].astype(float)

    rATR = ta.ATR(h,l,c,14)
    #print("primer ATR")
    #print(rATR)
    v=rATR.shape
    r=v[0]-1
    datr5m=rATR[r]
    print("atr5m es:", datr5m)




    async def order_book(client, symbol):
        order_book = await client.get_order_book(symbol=symbolo)
        print(order_book)


    async def kline_listener(client):
        bm = BinanceSocketManager(client)
        symbol = symbolo
        res_count = 0
        async with bm.kline_socket(symbol=symbol) as stream:
            while True:
                res = await stream.recv()
                res_count += 1
                print(res)
                if res_count == 5:
                    res_count = 0
                    loop.call_soon(asyncio.create_task, order_book(client, symbol))


    #INTERVALOS: 1HOUR 1MINUTE

    klines = client.get_klines(symbol=symbolo,interval=Client.KLINE_INTERVAL_15MINUTE)
    #start = ["month_1"].values.astype('datetime64[D]')

    df = pd.DataFrame(klines,  columns=['Date',
                                        'Open',
                                        'High',
                                        'Low',
                                        'Close',
                                        'Volume',
                                        'Close time',
                                        'Quote asset volume',
                                        'Number of trades',
                                        'Taker buy base asset volume',
                                        'Taker buy quote asset volume',
                                        'Ignore'])

    ###########################################################

    ########################################################

    ################################################################
    df = df.drop(df.columns[[ 6, 7, 8, 9, 10, 11]], axis=1)
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True, drop=True)
    #data = data.drop(data.columns[0],axis=1)

    o=df['Open']   = df['Open'].astype(float)
    h=df['High']   = df['High'].astype(float)
    l=df['Low']    = df['Low'].astype(float)
    c=df['Close']  = df['Close'].astype(float)
    v=df['Volume'] = df['Volume'].astype(float)

    rATR = ta.ATR(h,l,c,14)
    #print("primer ATR")
    #print(rATR)
    v=rATR.shape
    r=v[0]-1
    datr15m=rATR[r]
    print("atr15m es:", datr15m)



    async def order_book(client, symbol):
        order_book = await client.get_order_book(symbol=symbolo)
        print(order_book)


    async def kline_listener(client):
        bm = BinanceSocketManager(client)
        symbol = symbolo
        res_count = 0
        async with bm.kline_socket(symbol=symbol) as stream:
            while True:
                res = await stream.recv()
                res_count += 1
                print(res)
                if res_count == 5:
                    res_count = 0
                    loop.call_soon(asyncio.create_task, order_book(client, symbol))


    #INTERVALOS: 1HOUR 1MINUTE

    klines = client.get_klines(symbol=symbolo,interval=Client.KLINE_INTERVAL_30MINUTE)
    #start = ["month_1"].values.astype('datetime64[D]')

    df = pd.DataFrame(klines,  columns=['Date',
                                        'Open',
                                        'High',
                                        'Low',
                                        'Close',
                                        'Volume',
                                        'Close time',
                                        'Quote asset volume',
                                        'Number of trades',
                                        'Taker buy base asset volume',
                                        'Taker buy quote asset volume',
                                        'Ignore'])

    ###########################################################

    ########################################################

    ################################################################
    df = df.drop(df.columns[[ 6, 7, 8, 9, 10, 11]], axis=1)
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True, drop=True)
    #data = data.drop(data.columns[0],axis=1)

    o=df['Open']   = df['Open'].astype(float)
    h=df['High']   = df['High'].astype(float)
    l=df['Low']    = df['Low'].astype(float)
    c=df['Close']  = df['Close'].astype(float)
    v=df['Volume'] = df['Volume'].astype(float)

    rATR = ta.ATR(h,l,c,14)
    #print("primer ATR")
    #print(rATR)
    v=rATR.shape
    r=v[0]-1
    datr30m=rATR[r]
    print("atr30m es:", datr30m)
    """


    async def order_book(client, symbol):
        order_book = await client.get_order_book(symbol=symbolo)
        print(order_book)


    async def kline_listener(client):
        bm = BinanceSocketManager(client)
        symbol = symbolo
        res_count = 0
        async with bm.kline_socket(symbol=symbol) as stream:
            while True:
                res = await stream.recv()
                res_count += 1
                print(res)
                if res_count == 5:
                    res_count = 0
                    loop.call_soon(asyncio.create_task, order_book(client, symbol))


    # INTERVALOS: 1HOUR 1MINUTE

    klines = client.get_klines(symbol=symbolo, interval=Client.KLINE_INTERVAL_1HOUR)
    # start = ["month_1"].values.astype('datetime64[D]')

    df = pd.DataFrame(klines, columns=['Date',
                                       'Open',
                                       'High',
                                       'Low',
                                       'Close',
                                       'Volume',
                                       'Close time',
                                       'Quote asset volume',
                                       'Number of trades',
                                       'Taker buy base asset volume',
                                       'Taker buy quote asset volume',
                                       'Ignore'])

    ###########################################################

    ########################################################

    ################################################################
    df = df.drop(df.columns[[6, 7, 8, 9, 10, 11]], axis=1)
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True, drop=True)
    # data = data.drop(data.columns[0],axis=1)

    o = df['Open'] = df['Open'].astype(float)
    h = df['High'] = df['High'].astype(float)
    l = df['Low'] = df['Low'].astype(float)
    c = df['Close'] = df['Close'].astype(float)
    v = df['Volume'] = df['Volume'].astype(float)

    rATR = ta.ATR(h, l, c, 14)
    # print("primer ATR")
    # print(rATR)
    v = rATR.shape
    r = v[0] - 1
    datr1h = rATR[r]
    print("atr1h es:", datr1h)


    async def order_book(client, symbol):
        order_book = await client.get_order_book(symbol=symbolo)
        print(order_book)


    async def kline_listener(client):
        bm = BinanceSocketManager(client)
        symbol = symbolo
        res_count = 0
        async with bm.kline_socket(symbol=symbol) as stream:
            while True:
                res = await stream.recv()
                res_count += 1
                print(res)
                if res_count == 5:
                    res_count = 0
                    loop.call_soon(asyncio.create_task, order_book(client, symbol))


    # INTERVALOS: 1HOUR 1MINUTE

    klines = client.get_klines(symbol=symbolo, interval=Client.KLINE_INTERVAL_2HOUR)
    # start = ["month_1"].values.astype('datetime64[D]')

    df = pd.DataFrame(klines, columns=['Date',
                                       'Open',
                                       'High',
                                       'Low',
                                       'Close',
                                       'Volume',
                                       'Close time',
                                       'Quote asset volume',
                                       'Number of trades',
                                       'Taker buy base asset volume',
                                       'Taker buy quote asset volume',
                                       'Ignore'])

    ###########################################################

    ########################################################

    ################################################################
    df = df.drop(df.columns[[6, 7, 8, 9, 10, 11]], axis=1)
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True, drop=True)
    # data = data.drop(data.columns[0],axis=1)

    o = df['Open'] = df['Open'].astype(float)
    h = df['High'] = df['High'].astype(float)
    l = df['Low'] = df['Low'].astype(float)
    c = df['Close'] = df['Close'].astype(float)
    v = df['Volume'] = df['Volume'].astype(float)

    rATR = ta.ATR(h, l, c, 14)
    # print("primer ATR")
    # print(rATR)
    v = rATR.shape
    r = v[0] - 1
    datr2h = rATR[r]
    print("atr2h es:", datr2h)


    async def order_book(client, symbol):
        order_book = await client.get_order_book(symbol=symbolo)
        print(order_book)


    async def kline_listener(client):
        bm = BinanceSocketManager(client)
        symbol = symbolo
        res_count = 0
        async with bm.kline_socket(symbol=symbol) as stream:
            while True:
                res = await stream.recv()
                res_count += 1
                print(res)
                if res_count == 5:
                    res_count = 0
                    loop.call_soon(asyncio.create_task, order_book(client, symbol))


    # INTERVALOS: 1HOUR 1MINUTE

    klines = client.get_klines(symbol=symbolo, interval=Client.KLINE_INTERVAL_4HOUR)
    # start = ["month_1"].values.astype('datetime64[D]')

    df = pd.DataFrame(klines, columns=['Date',
                                       'Open',
                                       'High',
                                       'Low',
                                       'Close',
                                       'Volume',
                                       'Close time',
                                       'Quote asset volume',
                                       'Number of trades',
                                       'Taker buy base asset volume',
                                       'Taker buy quote asset volume',
                                       'Ignore'])

    ###########################################################

    ########################################################

    ################################################################
    df = df.drop(df.columns[[6, 7, 8, 9, 10, 11]], axis=1)
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True, drop=True)
    # data = data.drop(data.columns[0],axis=1)

    o = df['Open'] = df['Open'].astype(float)
    h = df['High'] = df['High'].astype(float)
    l = df['Low'] = df['Low'].astype(float)
    c = df['Close'] = df['Close'].astype(float)
    v = df['Volume'] = df['Volume'].astype(float)

    rATR = ta.ATR(h, l, c, 14)
    # print("primer ATR")
    # print(rATR)
    v = rATR.shape
    r = v[0] - 1
    datr4h = rATR[r]
    print("atr4h es:", datr4h)


    async def order_book(client, symbol):
        order_book = await client.get_order_book(symbol=symbolo)
        print(order_book)


    async def kline_listener(client):
        bm = BinanceSocketManager(client)
        symbol = symbolo
        res_count = 0
        async with bm.kline_socket(symbol=symbol) as stream:
            while True:
                res = await stream.recv()
                res_count += 1
                print(res)
                if res_count == 5:
                    res_count = 0
                    loop.call_soon(asyncio.create_task, order_book(client, symbol))


    # INTERVALOS: 1HOUR 1MINUTE

    klines = client.get_klines(symbol=symbolo, interval=Client.KLINE_INTERVAL_1DAY)
    # start = ["month_1"].values.astype('datetime64[D]')

    df = pd.DataFrame(klines, columns=['Date',
                                       'Open',
                                       'High',
                                       'Low',
                                       'Close',
                                       'Volume',
                                       'Close time',
                                       'Quote asset volume',
                                       'Number of trades',
                                       'Taker buy base asset volume',
                                       'Taker buy quote asset volume',
                                       'Ignore'])

    ###########################################################

    ########################################################

    ################################################################
    df = df.drop(df.columns[[6, 7, 8, 9, 10, 11]], axis=1)
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True, drop=True)
    # data = data.drop(data.columns[0],axis=1)

    o = df['Open'] = df['Open'].astype(float)
    h = df['High'] = df['High'].astype(float)
    l = df['Low'] = df['Low'].astype(float)
    c = df['Close'] = df['Close'].astype(float)
    v = df['Volume'] = df['Volume'].astype(float)

    rATR = ta.ATR(h, l, c, 14)
    # print("primer ATR")
    # print(rATR)
    v = rATR.shape
    r = v[0] - 1
    datr1d = rATR[r]
    print("atr1d es:", datr1d)

    f = c.shape
    g = f[0] - 1
    cc = c[g]
    CC = float(round((cc), 4))

    now = datetime.now()
    fecha = now.strftime("%d-%m-%y %H:%M:%S")
    lista = [symbolo]

    ##########################################################################

    now = datetime.now()
    fecha = now.strftime("%d-%m-%y %H:%M:%S")
    lista = [symbolo]
    # quantityOrders = 0.000001

    strongBuy_list1 = []
    strongSell_list1 = []
    for i in lista:
        tesla = TA_Handler()
        tesla.set_symbol_as(i)
        tesla.set_exchange_as_crypto_or_stock("BINANCE")
        tesla.set_screener_as_crypto()
        tesla.set_interval_as(Interval.INTERVAL_1_DAY)
        #    tesla.get_analysis().summary
        #    print(i)

        tesla.get_analysis().summary
        #    except Exception as e:
        #      print("No Data")
        #      continue
        if ((tesla.get_analysis().summary)["RECOMMENDATION"]) == "STRONG_BUY":
            #   print(f" Compar más fuerte {i}", fecha)
            strongBuy_list1.append(i)
        elif ((tesla.get_analysis().summary)["RECOMMENDATION"]) == "STRONG_SELL":
            #   print(f" Compar más fuerte {i}", fecha)
            strongSell_list1.append(i)

    strongBuy_list2 = []
    strongSell_list2 = []
    for i in lista:
        tesla = TA_Handler()
        tesla.set_symbol_as(i)
        tesla.set_exchange_as_crypto_or_stock("BINANCE")
        tesla.set_screener_as_crypto()
        tesla.set_interval_as(Interval.INTERVAL_4_HOURS)
        #    print(i)
        #    try:
        tesla.get_analysis().summary
        #    except Exception as e:
        #      print("No Data")
        #      continue
        if ((tesla.get_analysis().summary)["RECOMMENDATION"]) == "STRONG_BUY":
            #        print(f" Compar más fuerte {i}", fecha)
            strongBuy_list2.append(i)
        elif ((tesla.get_analysis().summary)["RECOMMENDATION"]) == "STRONG_SELL":
            #        print(f" Compar más fuerte {i}", fecha)
            strongSell_list2.append(i)

    strongBuy_list3 = []
    strongSell_list3 = []
    for i in lista:
        tesla = TA_Handler()
        tesla.set_symbol_as(i)
        tesla.set_exchange_as_crypto_or_stock("BINANCE")
        tesla.set_screener_as_crypto()
        tesla.set_interval_as(Interval.INTERVAL_1_HOUR)
        #    print(i)
        #    try:
        tesla.get_analysis().summary
        #    except Exception as e:
        #      print("No Data")
        #      continue
        if ((tesla.get_analysis().summary)["RECOMMENDATION"]) == "STRONG_BUY":
            #        print(f" Compar más fuerte {i}", fecha)
            strongBuy_list3.append(i)
        elif ((tesla.get_analysis().summary)["RECOMMENDATION"]) == "STRONG_SELL":
            #        print(f" Compar más fuerte {i}", fecha)
            strongSell_list3.append(i)
    """
    strongBuy_list4 = []
    strongSell_list4 = []

    for i in lista:
        tesla = TA_Handler()
        tesla.set_symbol_as(i)
        tesla.set_exchange_as_crypto_or_stock("BINANCE")
        tesla.set_screener_as_crypto()
        tesla.set_interval_as(Interval.INTERVAL_30_MINUTES)
    #    print(i)
    #    try:
        tesla.get_analysis().summary
    #    except Exception as e:
    #      print("No Data")
    #      continue
        if((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_BUY":
    #        print(f" Compar más fuerte {i}", fecha)
            strongBuy_list4.append(i)
        elif((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_SELL":
    #        print(f" Compar más fuerte {i}", fecha)
            strongSell_list4.append(i)

    strongBuy_list5 = []
    strongSell_list5 = []

    for i in lista:
        tesla = TA_Handler()
        tesla.set_symbol_as(i)
        tesla.set_exchange_as_crypto_or_stock("BINANCE")
        tesla.set_screener_as_crypto()
        tesla.set_interval_as(Interval.INTERVAL_15_MINUTES)
    #    print(i)
    #    try:
        tesla.get_analysis().summary
    #    except Exception as e:
    #      print("No Data")
    #      continue
        if((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_BUY":
    #        print(f" Compar más fuerte {i}", fecha)
            strongBuy_list5.append(i)
        elif((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_SELL":
    #        print(f" Compar más fuerte {i}", fecha)
            strongSell_list5.append(i)

    strongBuy_list6 = []
    strongSell_list6 = []

    for i in lista:
        tesla = TA_Handler()
        tesla.set_symbol_as(i)
        tesla.set_exchange_as_crypto_or_stock("BINANCE")
        tesla.set_screener_as_crypto()
        tesla.set_interval_as(Interval.INTERVAL_5_MINUTES)
    #    print(i)
    #    try:
        tesla.get_analysis().summary
    #    except Exception as e:
    #      print("No Data")
    #      continue
        if((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_BUY":
    #        print(f" Compar más fuerte {i}", fecha)
            strongBuy_list6.append(i)
        elif((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_SELL":
    #        print(f" Compar más fuerte {i}", fecha)
            strongSell_list6.append(i)
    """
    strongBuy_list7 = []
    strongSell_list7 = []

    for i in lista:
        tesla = TA_Handler()
        tesla.set_symbol_as(i)
        tesla.set_exchange_as_crypto_or_stock("BINANCE")
        tesla.set_screener_as_crypto()
        tesla.set_interval_as(Interval.INTERVAL_2_HOURS)
        #    print(i)
        #    try:
        tesla.get_analysis().summary
        #    except Exception as e:
        #      print("No Data")
        #      continue
        if ((tesla.get_analysis().summary)["RECOMMENDATION"]) == "STRONG_BUY":
            #        print(f" Compar más fuerte {i}", fecha)
            strongBuy_list7.append(i)
        elif ((tesla.get_analysis().summary)["RECOMMENDATION"]) == "STRONG_SELL":
            #        print(f" Compar más fuerte {i}", fecha)
            strongSell_list7.append(i)
    """
    strongBuy_list8 = []
    strongSell_list8 = []

    for i in lista:
        tesla = TA_Handler()
        tesla.set_symbol_as(i)
        tesla.set_exchange_as_crypto_or_stock("BINANCE")
        tesla.set_screener_as_crypto()
        tesla.set_interval_as(Interval.INTERVAL_1_MINUTE)
    #    print(i)
    #    try:
        tesla.get_analysis().summary
    #    except Exception as e:
    #      print("No Data")
    #      continue
        if((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_BUY":
    #        print(f" Compar más fuerte {i}", fecha)
            strongBuy_list8.append(i)
        elif((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_SELL":
    #        print(f" Compar más fuerte {i}", fecha)
            strongSell_list8.append(i)








    print("*** STRONG BUY LIST 1 MIN ***")

    print(strongBuy_list8)

    print("*** STRONG SELL LIST 1 MIN ***")

    print(strongSell_list8)


    print("*** STRONG BUY LIST 5 MIN ***")

    print(strongBuy_list6)

    print("*** STRONG SELL LIST 5 MIN ***")

    print(strongSell_list6)

    print("*** STRONG BUY LIST    15 MIN***")

    print(strongBuy_list5)

    print("*** STRONG SELL LIST    15 MIN***")

    print(strongSell_list5)


    print("*** STRONG BUY LIST          30 MIN***")

    print(strongBuy_list4)

    print("*** STRONG SELL LIST         30 MIN***")

    print(strongSell_list4)

    """

    print("*** STRONG BUY LIST              1 HORA***")

    print(strongBuy_list3)

    print("*** STRONG SELL LIST             1 HORA***")

    print(strongSell_list3)

    print("*** STRONG BUY LIST              2 HORAs***")

    print(strongBuy_list7)

    print("*** STRONG SELL LIST             2 HORAs***")

    print(strongSell_list7)

    print("*** STRONG BUY LIST               4 HORAS***")

    print(strongBuy_list2)

    print("*** STRONG SELL LIST              4 HORAS***")

    print(strongSell_list2)

    print("*** STRONG BUY LIST               1 DIA***")

    print(strongBuy_list1)

    print("*** STRONG SELL LIST              1 DIA***")

    print(strongSell_list1)

    try:
        Orders = client.futures_get_open_orders(symbol=symbolo)
    except Exception as e:
        print(e)
        client = Client(config.API_KEY, config.API_SECRET, tld='com')
        continue
    if len(Orders) != 0:
        print("ya hay ordenes")
        if client.futures_position_information(symbol=symbolo)[-1]['positionAmt'] != '0.00000':  # or time.time()>abc:
            telegram_send.send(messages=["hay ordenes sin tp y sl", symbolo])
            client.futures_cancel_order(symbol=symbolo)

    else:
        if len(strongBuy_list1) != 0:
            f = c.shape
            g = f[0] - 1
            cc = c[g]
            CC = float(round((cc), 4))
            print("el close es:", CC)
            tpl = cc + ((1.5) * datr1d)
            # sls=cc+((1.5)*datr1d)
            # tps=cc-((1.5)*datr1d)
            sll = cc - ((1.5) * datr1d)
            TPL = float(round((tpl), 4))
            SLL = float(round((sll), 4))
            # TPS = float(round((tps) , 4))
            # SLS = float(round((sls) , 4))
            print("TPL:", TPL)
            print("SLL:", SLL)
            # print("TPS:",TPS)
            # print("SLS:",SLS)
            client.futures_change_leverage(symbol=symbolo, leverage=leve)

            order_buy = client.futures_create_order(
                symbol=symbolo,
                side='BUY',
                type='MARKET',
                quantity=cantidad,
                # reduceOnly=True
            )
            order_Buy_ID = order_buy['orderId']
            print(order_buy)
            precio = order_buy['price']
            print("Order ID: " + str(order_Buy_ID))

            order_buysl = client.futures_create_order(
                symbol=symbolo,
                # quantity=cantidad,
                side='SELL',
                type='STOP_MARKET',
                stopPrice=SLL,
                closePosition=True

                # activationPrice=cc*1.01,
                # callbackRate=1,
                ##reduceOnly='True'
            )
            order_Buysl_ID = order_buysl['orderId']
            print(order_buysl)
            print("Or der IDsl: " + str(order_Buysl_ID))
            order_buytp = client.futures_create_order(
                symbol=symbolo,
                side='SELL',
                type='TAKE_PROFIT_MARKET',
                stopPrice=TPL,
                closePosition=True
            )
            order_Buytp_ID = order_buytp['orderId']
            print(order_buytp)
            print("Or der IDtp: " + str(order_Buytp_ID))
            telegram_send.send(messages=["Buy order!", symbolo])
            print("Order ID: " + str(order_Buy_ID))
            print("Order IDtp: " + str(order_Buytp_ID))
            print("Order IDsl: " + str(order_Buysl_ID))
            mylista1[0] = (order_Buy_ID)
            mylista1[1] = (order_Buytp_ID)
            mylista1[2] = (order_Buysl_ID)
            print(str(mylista1))
            print("primeras órdenes")
            print(client.futures_position_information(symbol=symbolo))
            abc = time.time() + (60 * 60 * 24)
            while True:
                print(client.futures_position_information(symbol=symbolo))
                orders = client.futures_get_open_orders(symbol=symbolo)
                now = datetime.now()
                fecha = now.strftime("%d-%m-%y %H:%M:%S")
                lista = [symbolo]
                print("length lista es:")
                print(len(lista))
                print("length simbolo es:")
                print(len(symbolo))
                print(now)
                print(order_Buy_ID)

                strongBuy_list1 = []
                strongSell_list1 = []
                for i in lista:
                    tesla = TA_Handler()
                    tesla.set_symbol_as(i)
                    tesla.set_exchange_as_crypto_or_stock("BINANCE")
                    tesla.set_screener_as_crypto()
                    tesla.set_interval_as(Interval.INTERVAL_1_DAY)
                    # openOrder = client.get_order(symbol=symbolo)
                    # print(openOrder)
                    # orderId = openOrder[0]['orderId']
                    # print(orderId)
                    print(i)
                    try:
                        print(tesla.get_analysis().summary)
                    except Exception as e:
                        print("No Data")
                        continue
                    if ((tesla.get_analysis().summary)["RECOMMENDATION"]) == "STRONG_BUY":
                        print(f" Compar más fuerte {i}", fecha)
                        strongBuy_list1.append(i)

                print("*** STRONG BUY LIST               1 DIA***")

                print(strongBuy_list1)
                entrada[0] = float(client.futures_position_information(symbol=symbolo)[-1]['entryPrice'])
                print("precio entrada es", entrada)
                amount[0] = float(client.futures_position_information(symbol=symbolo)[-1]['positionAmt'])
                print("el lotaje es: ", amount)
                leverage[0] = float(client.futures_position_information(symbol=symbolo)[-1]['leverage'])
                print("el leverage es: ", leverage)
                bingo[-1] = float(entrada[0] + (comision * amount[0] * entrada[0] * 3.0))
                print("bingo es: ", bingo)
                markPrice[0] = float(client.futures_position_information(symbol=symbolo)[-1]['markPrice'])
                print("el precio mark es:", markPrice)
                if bingo[-1] < markPrice[-1]:
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    client.futures_create_order(symbol=symbolo, side='SELL', type="MARKET", quantity=cantidad,
                                                redueOnly='True')
                    telegram_send.send(messages=["Profit!", symbolo])
                    break

                # order_ids = [order['clientOrderId'] for order in orders]
                if client.futures_position_information(symbol=symbolo)[-1]['positionAmt'] == '0.00000':
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    telegram_send.send(messages=["Buy cerrado y ordenes cerradas!", symbolo])
                    break
                if len(strongBuy_list1) != 1:
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    client.futures_create_order(symbol=symbolo, side='SELL', type="MARKET", quantity=cantidad,
                                                redueOnly='True')
                    telegram_send.send(messages=["Buy cerrado y ordenes cerradas!", symbolo])
                    break
                # else: print("na de na")
        elif len(strongSell_list1) != 0:
            f = c.shape
            g = f[0] - 1
            cc = c[g]
            CC = float(round((cc), 4))
            print("el close es:", CC)
            # tpl=cc+((1.5)*datr1d)
            sls = cc + ((1.5) * datr1d)
            tps = cc - ((1.5) * datr1d)
            # sll=cc-((1.5)*datr1d)
            # TPL = float(round((tpl) , 4))
            # SLL = float(round((sll) , 4))
            TPS = float(round((tps), 4))
            SLS = float(round((sls), 4))
            # print("TPL:",TPL)
            # print("SLL:",SLL)
            print("TPS:", TPS)
            print("SLS:", SLS)
            client.futures_change_leverage(symbol=symbolo, leverage=leve)
            order_sell = client.futures_create_order(
                symbol=symbolo,
                side='SELL',
                type='MARKET',
                quantity=cantidad,
                # reduceOnly=True
            )

            order_Sell_ID = order_sell['orderId']
            print(order_sell)
            precio = order_sell['price']
            print("Order ID: " + str(order_Sell_ID))
            order_sellsl = client.futures_create_order(
                symbol=symbolo,
                # quantity=cantidad,
                side='BUY',
                type='STOP_MARKET',
                stopPrice=SLS,
                closePosition=True
                # activationPrice=cc*1.01,
                # callbackRate=1,
                ##reduceOnly='True'
            )
            order_Sellsl_ID = order_sellsl['orderId']
            print(order_sellsl)
            print("Or der IDsl: " + str(order_Sellsl_ID))
            order_selltp = client.futures_create_order(
                symbol=symbolo,
                side='BUY',
                type='TAKE_PROFIT_MARKET',
                stopPrice=TPS,
                closePosition=True
            )

            order_Selltp_ID = order_selltp['orderId']
            print(order_selltp)
            print("Order ID: " + str(order_Selltp_ID))
            telegram_send.send(messages=["Sell order!", symbolo])
            print("Order ID: " + str(order_Sell_ID))
            print("Order IDtp: " + str(order_Selltp_ID))
            print("Order IDsl: " + str(order_Sellsl_ID))
            mylista1[0] = (order_Sell_ID)
            mylista1[1] = (order_Selltp_ID)
            mylista1[2] = (order_Sellsl_ID)
            print(str(mylista1))
            print("primeras órdenes")
            print(client.futures_position_information(symbol=symbolo))
            abc = time.time() + (60 * 60 * 24)
            while True:
                print(client.futures_position_information(symbol=symbolo))
                orders = client.futures_get_open_orders(symbol=symbolo)
                now = datetime.now()
                fecha = now.strftime("%d-%m-%y %H:%M:%S")
                lista = [symbolo]
                print("length lista es:")
                print(len(lista))
                print("length simbolo es:")
                print(len(symbolo))
                print(now)
                print(order_Sell_ID)

                strongBuy_list1 = []
                strongSell_list1 = []
                for i in lista:
                    tesla = TA_Handler()
                    tesla.set_symbol_as(i)
                    tesla.set_exchange_as_crypto_or_stock("BINANCE")
                    tesla.set_screener_as_crypto()
                    tesla.set_interval_as(Interval.INTERVAL_1_DAY)
                    # openOrder = client.get_order(symbol=symbolo)
                    # print(openOrder)
                    # orderId = openOrder[0]['orderId']
                    # print(orderId)
                    print(i)
                    try:
                        print(tesla.get_analysis().summary)
                    except Exception as e:
                        print("No Data")
                        continue

                    if ((tesla.get_analysis().summary)["RECOMMENDATION"]) == "STRONG_SELL":
                        print(f" Compar más fuerte {i}", fecha)
                        strongSell_list1.append(i)

                print("*** STRONG SELL LIST              1 DIA***")

                print(strongSell_list1)

                entrada[0] = float(client.futures_position_information(symbol=symbolo)[-1]['entryPrice'])
                print("precio entrada es", entrada)
                amount[0] = float(client.futures_position_information(symbol=symbolo)[-1]['positionAmt'])
                print("el lotaje es: ", amount)
                leverage[0] = float(client.futures_position_information(symbol=symbolo)[-1]['leverage'])
                print("el leverage es: ", leverage)
                bingo[-1] = float(entrada[0] + (comision * amount[0] * entrada[0] * 3.0))
                print("bingo es: ", bingo)
                markPrice[0] = float(client.futures_position_information(symbol=symbolo)[-1]['markPrice'])
                print("el precio mark es:", markPrice)
                if bingo[-1] > markPrice[-1]:
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    client.futures_create_order(symbol=symbolo, side='BUY', type="MARKET", quantity=cantidad,
                                                redueOnly='True')
                    telegram_send.send(messages=["Profit!", symbolo])
                    break
                if client.futures_position_information(symbol=symbolo)[-1]['positionAmt'] == '0.00000':
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    telegram_send.send(messages=["Sell order closed y ordenes cerradas!", symbolo])
                    break
                if len(strongSell_list1) != 1:
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    client.futures_create_order(symbol=symbolo, side='BUY', type="MARKET", quantity=cantidad,
                                                redueOnly='True')
                    telegram_send.send(messages=["Buy cerrado y ordenes cerradas!", symbolo])
                    break
                # else: print("na de na")
        elif len(strongBuy_list2) != 0:
            f = c.shape
            g = f[0] - 1
            cc = c[g]
            CC = float(round((cc), 4))
            print("el close es:", CC)
            tpl = cc + ((1.5) * datr4h)
            # sls=cc+((1.5)*datr1d)
            # tps=cc-((1.5)*datr1d)
            sll = cc - ((1.5) * datr4h)
            TPL = float(round((tpl), 4))
            SLL = float(round((sll), 4))
            # TPS = float(round((tps) , 4))
            # SLS = float(round((sls) , 4))
            print("TPL:", TPL)
            print("SLL:", SLL)
            # print("TPS:",TPS)
            # print("SLS:",SLS)
            client.futures_change_leverage(symbol=symbolo, leverage=leve)

            order_buy = client.futures_create_order(
                symbol=symbolo,
                side='BUY',
                type='MARKET',
                quantity=cantidad,
                # reduceOnly=True
            )
            order_Buy_ID = order_buy['orderId']
            print(order_buy)
            precio = order_buy['price']
            print("Order ID: " + str(order_Buy_ID))

            order_buysl = client.futures_create_order(
                symbol=symbolo,
                # quantity=cantidad,
                side='SELL',
                type='STOP_MARKET',
                stopPrice=SLL,
                closePosition=True
                # activationPrice=cc*1.01,
                # callbackRate=1,
                ##reduceOnly='True'
            )
            order_Buysl_ID = order_buysl['orderId']
            print(order_buysl)
            print("Or der IDsl: " + str(order_Buysl_ID))
            order_buytp = client.futures_create_order(
                symbol=symbolo,
                side='SELL',
                type='TAKE_PROFIT_MARKET',
                stopPrice=TPL,
                closePosition=True
            )
            order_Buytp_ID = order_buytp['orderId']
            print(order_buytp)
            print("Or der IDtp: " + str(order_Buytp_ID))
            telegram_send.send(messages=["Buy order!", symbolo])
            print("Order ID: " + str(order_Buy_ID))
            print("Order IDtp: " + str(order_Buytp_ID))
            print("Order IDsl: " + str(order_Buysl_ID))
            mylista1[0] = (order_Buy_ID)
            mylista1[1] = (order_Buytp_ID)
            mylista1[2] = (order_Buysl_ID)
            print(str(mylista1))
            print("primeras órdenes")
            print(client.futures_position_information(symbol=symbolo))
            abc = time.time() + (60 * 60 * 4)
            while True:
                print(client.futures_position_information(symbol=symbolo))
                orders = client.futures_get_open_orders(symbol=symbolo)
                now = datetime.now()
                fecha = now.strftime("%d-%m-%y %H:%M:%S")
                lista = [symbolo]
                print("length lista es:")
                print(len(lista))
                print("length simbolo es:")
                print(len(symbolo))
                print(now)
                print(order_Buy_ID)
                # openOrder = client.get_order(symbol=symbolo)
                # print(openOrder)
                # orderId = openOrder[0]['orderId']
                # print(orderId)
                strongBuy_list2 = []
                strongSell_list2 = []
                for i in lista:
                    tesla = TA_Handler()
                    tesla.set_symbol_as(i)
                    tesla.set_exchange_as_crypto_or_stock("BINANCE")
                    tesla.set_screener_as_crypto()
                    tesla.set_interval_as(Interval.INTERVAL_4_HOURS)
                    print(i)
                    try:
                        print(tesla.get_analysis().summary)
                    except Exception as e:
                        print("No Data")
                        continue
                    if ((tesla.get_analysis().summary)["RECOMMENDATION"]) == "STRONG_BUY":
                        print(f" Compar más fuerte {i}", fecha)
                        strongBuy_list2.append(i)
                print("*** STRONG BUY LIST               4 horas***")

                print(strongBuy_list2)
                entrada[0] = float(client.futures_position_information(symbol=symbolo)[-1]['entryPrice'])
                print("precio entrada es", entrada)
                amount[0] = float(client.futures_position_information(symbol=symbolo)[-1]['positionAmt'])
                print("el lotaje es: ", amount)
                leverage[0] = float(client.futures_position_information(symbol=symbolo)[-1]['leverage'])
                print("el leverage es: ", leverage)
                bingo[-1] = float(entrada[0] + (comision * amount[0] * entrada[0] * 3.0))
                print("bingo es: ", bingo)
                markPrice[0] = float(client.futures_position_information(symbol=symbolo)[-1]['markPrice'])
                print("el precio mark es:", markPrice)
                if bingo[-1] < markPrice[-1]:
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    client.futures_create_order(symbol=symbolo, side='SELL', type="MARKET", quantity=cantidad,
                                                redueOnly='True')
                    telegram_send.send(messages=["Profit!", symbolo])
                    break
                if client.futures_position_information(symbol=symbolo)[-1]['positionAmt'] == '0.00000':
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    telegram_send.send(messages=["Buy cerrado y ordenes cerradas!", symbolo])
                    break
                if len(strongBuy_list2) != 1:
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    client.futures_create_order(symbol=symbolo, side='SELL', type="MARKET", quantity=cantidad,
                                                redueOnly='True')
                    telegram_send.send(messages=["Buy cerrado y ordenes cerradas!", symbolo])
                    break
                # else: print("na de na")

        elif len(strongSell_list2) != 0:
            f = c.shape
            g = f[0] - 1
            cc = c[g]
            CC = float(round((cc), 4))
            print("el close es:", CC)
            # tpl=cc+((1.5)*datr1d)
            sls = cc + ((1.5) * datr4h)
            tps = cc - ((1.5) * datr4h)
            # sll=cc-((1.5)*datr1d)
            # TPL = float(round((tpl) , 4))
            # SLL = float(round((sll) , 4))
            TPS = float(round((tps), 4))
            SLS = float(round((sls), 4))
            # print("TPL:",TPL)
            # print("SLL:",SLL)
            print("TPS:", TPS)
            print("SLS:", SLS)
            client.futures_change_leverage(symbol=symbolo, leverage=leve)
            order_sell = client.futures_create_order(
                symbol=symbolo,
                side='SELL',
                type='MARKET',
                quantity=cantidad,
                # reduceOnly=True
            )

            order_Sell_ID = order_sell['orderId']
            print(order_sell)
            precio = order_sell['price']
            print("Order ID: " + str(order_Sell_ID))
            order_sellsl = client.futures_create_order(
                symbol=symbolo,
                # quantity=cantidad,
                side='BUY',
                type='STOP_MARKET',
                stopPrice=SLS,
                closePosition=True
                # activationPrice=cc*1.01,
                # callbackRate=1,
                ##reduceOnly='True'
            )
            order_Sellsl_ID = order_sellsl['orderId']
            print(order_sellsl)
            print("Or der IDsl: " + str(order_Sellsl_ID))
            order_selltp = client.futures_create_order(
                symbol=symbolo,
                side='BUY',
                type='TAKE_PROFIT_MARKET',
                stopPrice=TPS,
                closePosition=True
            )

            order_Selltp_ID = order_selltp['orderId']
            print(order_selltp)
            print("Order ID: " + str(order_Selltp_ID))
            telegram_send.send(messages=["Sell order!", symbolo])
            print("Order ID: " + str(order_Sell_ID))
            print("Order IDtp: " + str(order_Selltp_ID))
            print("Order IDsl: " + str(order_Sellsl_ID))
            mylista1[0] = (order_Sell_ID)
            mylista1[1] = (order_Selltp_ID)
            mylista1[2] = (order_Sellsl_ID)
            print(str(mylista1))
            print("primeras órdenes")
            print(client.futures_position_information(symbol=symbolo))
            abc = time.time() + (60 * 60 * 4)
            while True:
                print(client.futures_position_information(symbol=symbolo))
                orders = client.futures_get_open_orders(symbol=symbolo)
                now = datetime.now()
                fecha = now.strftime("%d-%m-%y %H:%M:%S")
                lista = [symbolo]
                print("length lista es:")
                print(len(lista))
                print("length simbolo es:")
                print(len(symbolo))
                print(now)
                print(order_Sell_ID)
                # openOrder = client.get_order(symbol=symbolo)
                # print(openOrder)
                # orderId = openOrder[0]['orderId']
                # print(orderId)
                strongBuy_list2 = []
                strongSell_list2 = []
                for i in lista:
                    tesla = TA_Handler()
                    tesla.set_symbol_as(i)
                    tesla.set_exchange_as_crypto_or_stock("BINANCE")
                    tesla.set_screener_as_crypto()
                    tesla.set_interval_as(Interval.INTERVAL_4_HOURS)
                    print(i)
                    try:
                        print(tesla.get_analysis().summary)
                    except Exception as e:
                        print("No Data")
                        continue

                    if ((tesla.get_analysis().summary)["RECOMMENDATION"]) == "STRONG_SELL":
                        print(f" Compar más fuerte {i}", fecha)
                        strongSell_list2.append(i)

                print("*** STRONG SELL LIST              4 horas***")

                print(strongSell_list2)

                entrada[0] = float(client.futures_position_information(symbol=symbolo)[-1]['entryPrice'])
                print("precio entrada es", entrada)
                amount[0] = float(client.futures_position_information(symbol=symbolo)[-1]['positionAmt'])
                print("el lotaje es: ", amount)
                leverage[0] = float(client.futures_position_information(symbol=symbolo)[-1]['leverage'])
                print("el leverage es: ", leverage)
                bingo[-1] = float(entrada[0] + (comision * amount[0] * entrada[0] * 3.0))
                print("bingo es: ", bingo)
                markPrice[0] = float(client.futures_position_information(symbol=symbolo)[-1]['markPrice'])
                print("el precio mark es:", markPrice)
                if bingo[-1] > markPrice[-1]:
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    client.futures_create_order(symbol=symbolo, side='BUY', type="MARKET", quantity=cantidad,
                                                redueOnly='True')
                    telegram_send.send(messages=["Profit!", symbolo])
                    break

                if client.futures_position_information(symbol=symbolo)[-1]['positionAmt'] == '0.00000':
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    telegram_send.send(messages=["Sell order closed y ordenes cerradas!", symbolo])
                    break
                if len(strongSell_list2) != 1:
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    client.futures_create_order(symbol=symbolo, side='BUY', type="MARKET", quantity=cantidad,
                                                redueOnly='True')
                    telegram_send.send(messages=["Buy cerrado y ordenes cerradas!", symbolo])
                    break
                # else: print("na de na")

        elif len(strongBuy_list7) != 0:
            f = c.shape
            g = f[0] - 1
            cc = c[g]
            CC = float(round((cc), 4))
            print("el close es:", CC)
            tpl = cc + ((1.5) * datr2h)
            # sls=cc+((1.5)*datr1d)
            # tps=cc-((1.5)*datr1d)
            sll = cc - ((1.5) * datr2h)
            TPL = float(round((tpl), 4))
            SLL = float(round((sll), 4))
            # TPS = float(round((tps) , 4))
            # SLS = float(round((sls) , 4))
            print("TPL:", TPL)
            print("SLL:", SLL)
            # print("TPS:",TPS)
            # print("SLS:",SLS)
            client.futures_change_leverage(symbol=symbolo, leverage=leve)

            order_buy = client.futures_create_order(
                symbol=symbolo,
                side='BUY',
                type='MARKET',
                quantity=cantidad,
                # reduceOnly=True
            )
            order_Buy_ID = order_buy['orderId']
            print(order_buy)
            precio = order_buy['price']
            print("Order ID: " + str(order_Buy_ID))

            order_buysl = client.futures_create_order(
                symbol=symbolo,
                # quantity=cantidad,
                side='SELL',
                type='STOP_MARKET',
                stopPrice=SLL,
                closePosition=True
                # activationPrice=cc*1.01,
                # callbackRate=1,
                ##reduceOnly='True'
            )
            order_Buysl_ID = order_buysl['orderId']
            print(order_buysl)
            print("Or der IDsl: " + str(order_Buysl_ID))
            order_buytp = client.futures_create_order(
                symbol=symbolo,
                side='SELL',
                type='TAKE_PROFIT_MARKET',
                stopPrice=TPL,
                closePosition=True
            )
            order_Buytp_ID = order_buytp['orderId']
            print(order_buytp)
            print("Or der IDtp: " + str(order_Buytp_ID))
            telegram_send.send(messages=["Buy order!", symbolo])
            print("Order ID: " + str(order_Buy_ID))
            print("Order IDtp: " + str(order_Buytp_ID))
            print("Order IDsl: " + str(order_Buysl_ID))
            mylista1[0] = (order_Buy_ID)
            mylista1[1] = (order_Buytp_ID)
            mylista1[2] = (order_Buysl_ID)
            print(str(mylista1))
            print("primeras órdenes")
            print(client.futures_position_information(symbol=symbolo))
            abc = time.time() + (60 * 60 * 2)
            while True:
                print(client.futures_position_information(symbol=symbolo))
                orders = client.futures_get_open_orders(symbol=symbolo)
                now = datetime.now()
                fecha = now.strftime("%d-%m-%y %H:%M:%S")
                lista = [symbolo]
                print("length lista es:")
                print(len(lista))
                print("length simbolo es:")
                print(len(symbolo))
                print(now)
                print(order_Buy_ID)
                # openOrder = client.get_order(symbol=symbolo,orderId=order_Buy_ID)
                ##print(openOrder)
                ##orderId = openOrder[0]['orderId']
                ##print(orderId)
                strongBuy_list7 = []
                strongSell_list7 = []
                entrada[0] = None
                amount[0] = None
                for i in lista:
                    tesla = TA_Handler()
                    tesla.set_symbol_as(i)
                    tesla.set_exchange_as_crypto_or_stock("BINANCE")
                    tesla.set_screener_as_crypto()
                    tesla.set_interval_as(Interval.INTERVAL_2_HOURS)
                    print(i)
                    try:
                        print(tesla.get_analysis().summary)
                    except Exception as e:
                        print("No Data")
                        continue
                    if ((tesla.get_analysis().summary)["RECOMMENDATION"]) == "STRONG_BUY":
                        print(f" Compar más fuerte {i}", fecha)
                        strongBuy_list7.append(i)
                print("*** STRONG buy LIST              2 horas***")

                print(strongBuy_list7)
                entrada[0] = float(client.futures_position_information(symbol=symbolo)[-1]['entryPrice'])
                print("precio entrada es", entrada)
                amount[0] = float(client.futures_position_information(symbol=symbolo)[-1]['positionAmt'])
                print("el lotaje es: ", amount)
                leverage[0] = float(client.futures_position_information(symbol=symbolo)[-1]['leverage'])
                print("el leverage es: ", leverage)
                bingo[-1] = float(entrada[0] + (comision * amount[0] * entrada[0] * 3.0))
                print("bingo es: ", bingo)
                markPrice[0] = float(client.futures_position_information(symbol=symbolo)[-1]['markPrice'])
                print("el precio mark es:", markPrice)
                if bingo[-1] < markPrice[-1]:
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    client.futures_create_order(symbol=symbolo, side='SELL', type="MARKET", quantity=cantidad,
                                                redueOnly='True')
                    telegram_send.send(messages=["Profit!", symbolo])
                    break
                if client.futures_position_information(symbol=symbolo)[-1]['positionAmt'] == '0.00000':
                    time.sleep(2)
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    telegram_send.send(messages=["Buy cerrado y ordenes cerradas!", symbolo])
                    break
                if len(strongBuy_list7) != 1:
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    client.futures_create_order(symbol=symbolo, side='SELL', type="MARKET", quantity=cantidad,
                                                redueOnly='True')
                    telegram_send.send(messages=["Buy cerrado y ordenes cerradas!", symbolo])
                    break
                # else: print("na de na")

        elif len(strongSell_list7) != 0:
            f = c.shape
            g = f[0] - 1
            cc = c[g]
            CC = float(round((cc), 4))
            print("el close es:", CC)
            # tpl=cc+((1.5)*datr1d)
            sls = cc + ((1.5) * datr2h)
            tps = cc - ((1.5) * datr2h)
            # sll=cc-((1.5)*datr1d)
            # TPL = float(round((tpl) , 4))
            # SLL = float(round((sll) , 4))
            TPS = float(round((tps), 4))
            SLS = float(round((sls), 4))
            # print("TPL:",TPL)
            # print("SLL:",SLL)
            print("TPS:", TPS)
            print("SLS:", SLS)
            client.futures_change_leverage(symbol=symbolo, leverage=leve)
            order_sell = client.futures_create_order(
                symbol=symbolo,
                side='SELL',
                type='MARKET',
                quantity=cantidad,
                # reduceOnly=True
            )

            order_Sell_ID = order_sell['orderId']
            print(order_sell)
            precio = order_sell['price']
            print("Order ID: " + str(order_Sell_ID))
            order_sellsl = client.futures_create_order(
                symbol=symbolo,
                # quantity=cantidad,
                side='BUY',
                type='STOP_MARKET',
                stopPrice=SLS,
                closePosition=True
                # activationPrice=cc*1.01,
                # callbackRate=1,
                ##reduceOnly='True'
            )
            order_Sellsl_ID = order_sellsl['orderId']
            print(order_sellsl)
            print("Or der IDsl: " + str(order_Sellsl_ID))
            order_selltp = client.futures_create_order(
                symbol=symbolo,
                side='BUY',
                type='TAKE_PROFIT_MARKET',
                stopPrice=TPS,
                closePosition=True
            )

            order_Selltp_ID = order_selltp['orderId']
            print(order_selltp)
            print("Order ID: " + str(order_Selltp_ID))
            telegram_send.send(messages=["Sell order!", symbolo])
            print("Order ID: " + str(order_Sell_ID))
            print("Order IDtp: " + str(order_Selltp_ID))
            print("Order IDsl: " + str(order_Sellsl_ID))
            mylista1[0] = (order_Sell_ID)
            mylista1[1] = (order_Selltp_ID)
            mylista1[2] = (order_Sellsl_ID)
            print(str(mylista1))
            print("primeras órdenes")
            print(client.futures_position_information(symbol=symbolo))
            abc = time.time() + (60 * 60 * 2)
            while True:
                print(client.futures_position_information(symbol=symbolo))
                orders = client.futures_get_open_orders(symbol=symbolo)
                now = datetime.now()
                fecha = now.strftime("%d-%m-%y %H:%M:%S")
                lista = [symbolo]
                print("length lista es:")
                print(len(lista))
                print("length simbolo es:")
                print(len(symbolo))
                print(now)
                print(order_Sell_ID)
                # openOrder = client.get_order(symbol=symbolo,orderId=order_Sell_ID)
                ##print(openOrder)
                ##orderId = openOrder[0]['orderId']
                ##print(orderId)
                strongBuy_list7 = []
                strongSell_list7 = []
                for i in lista:
                    tesla = TA_Handler()
                    tesla.set_symbol_as(i)
                    tesla.set_exchange_as_crypto_or_stock("BINANCE")
                    tesla.set_screener_as_crypto()
                    tesla.set_interval_as(Interval.INTERVAL_2_HOURS)
                    print(i)
                    try:
                        print(tesla.get_analysis().summary)
                    except Exception as e:
                        print("No Data")
                        continue

                    if ((tesla.get_analysis().summary)["RECOMMENDATION"]) == "STRONG_SELL":
                        print(f" Compar más fuerte {i}", fecha)
                        strongSell_list7.append(i)

                print("*** STRONG SELL LIST              2horas***")

                print(strongSell_list7)

                entrada[0] = float(client.futures_position_information(symbol=symbolo)[-1]['entryPrice'])
                print("precio entrada es", entrada)
                amount[0] = float(client.futures_position_information(symbol=symbolo)[-1]['positionAmt'])
                print("el lotaje es: ", amount)
                leverage[0] = float(client.futures_position_information(symbol=symbolo)[-1]['leverage'])
                print("el leverage es: ", leverage)
                bingo[-1] = float(entrada[0] + (comision * amount[0] * entrada[0] * 3.0))
                print("bingo es: ", bingo)
                markPrice[0] = float(client.futures_position_information(symbol=symbolo)[-1]['markPrice'])
                print("el precio mark es:", markPrice)
                if bingo[-1] > markPrice[-1]:
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    client.futures_create_order(symbol=symbolo, side='BUY', type="MARKET", quantity=cantidad,
                                                redueOnly='True')
                    telegram_send.send(messages=["Profit!", symbolo])
                    break
                if client.futures_position_information(symbol=symbolo)[-1]['positionAmt'] == '0.00000':
                    time.sleep(2)
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    telegram_send.send(messages=["Buy cerrado y ordenes cerradas!", symbolo])
                    break
                if len(strongSell_list7) != 1:
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    client.futures_create_order(symbol=symbolo, side='BUY', type="MARKET", quantity=cantidad,
                                                redueOnly='True')
                    telegram_send.send(messages=["Buy cerrado y ordenes cerradas!", symbolo])
                    break
                # else: print("na de na")

        elif len(strongBuy_list3) != 0:
            f = c.shape
            g = f[0] - 1
            cc = c[g]
            CC = float(round((cc), 4))
            print("el close es:", CC)
            tpl = cc + ((1.5) * datr1h)
            # sls=cc+((1.5)*datr1d)
            # tps=cc-((1.5)*datr1d)
            sll = cc - ((1.5) * datr1h)
            TPL = float(round((tpl), 4))
            SLL = float(round((sll), 4))
            # TPS = float(round((tps) , 4))
            # SLS = float(round((sls) , 4))
            print("TPL:", TPL)
            print("SLL:", SLL)
            # print("TPS:",TPS)
            # print("SLS:",SLS)
            client.futures_change_leverage(symbol=symbolo, leverage=leve)

            order_buy = client.futures_create_order(
                symbol=symbolo,
                side='BUY',
                type='MARKET',
                quantity=cantidad,
                # reduceOnly=True
            )
            order_Buy_ID = order_buy['orderId']
            print(order_buy)
            precio = order_buy['price']
            print("Order ID: " + str(order_Buy_ID))

            order_buysl = client.futures_create_order(
                symbol=symbolo,
                # quantity=cantidad,
                side='SELL',
                type='STOP_MARKET',
                stopPrice=SLL,
                closePosition=True
                # activationPrice=cc*1.01,
                # callbackRate=1,
                ##reduceOnly='True'
            )
            order_Buysl_ID = order_buysl['orderId']
            print(order_buysl)
            print("Or der IDsl: " + str(order_Buysl_ID))
            order_buytp = client.futures_create_order(
                symbol=symbolo,
                side='SELL',
                type='TAKE_PROFIT_MARKET',
                stopPrice=TPL,
                closePosition=True
            )
            order_Buytp_ID = order_buytp['orderId']
            print(order_buytp)
            print("Or der IDtp: " + str(order_Buytp_ID))
            telegram_send.send(messages=["Buy order!", symbolo])
            print("Order ID: " + str(order_Buy_ID))
            print("Order IDtp: " + str(order_Buytp_ID))
            print("Order IDsl: " + str(order_Buysl_ID))
            mylista1[0] = (order_Buy_ID)
            mylista1[1] = (order_Buytp_ID)
            mylista1[2] = (order_Buysl_ID)
            print(str(mylista1))
            print("primeras órdenes")
            print(client.futures_position_information(symbol=symbolo))
            abc = time.time() + (60 * 60)
            while True:
                print(client.futures_position_information(symbol=symbolo))
                orders = client.futures_get_open_orders(symbol=symbolo)
                now = datetime.now()
                fecha = now.strftime("%d-%m-%y %H:%M:%S")
                lista = [symbolo]
                print("length lista es:")
                print(len(lista))
                print("length simbolo es:")
                print(len(symbolo))
                print(now)
                print(order_Buy_ID)
                # openOrder = client.get_order(symbol=symbolo,orderId=order_Buy_ID)
                ##print(openOrder)
                ##orderId = openOrder[0]['orderId']
                ##print(orderId)
                strongBuy_list3 = []
                strongSell_list3 = []
                for i in lista:
                    tesla = TA_Handler()
                    tesla.set_symbol_as(i)
                    tesla.set_exchange_as_crypto_or_stock("BINANCE")
                    tesla.set_screener_as_crypto()
                    tesla.set_interval_as(Interval.INTERVAL_1_HOUR)
                    print(i)
                    try:
                        print(tesla.get_analysis().summary)
                    except Exception as e:
                        print("No Data")
                        continue
                    if ((tesla.get_analysis().summary)["RECOMMENDATION"]) == "STRONG_BUY":
                        print(f" Compar más fuerte {i}", fecha)
                        strongBuy_list3.append(i)
                print("*** STRONG buy LIST              1 hora***")

                print(strongBuy_list3)
                entrada[0] = float(client.futures_position_information(symbol=symbolo)[-1]['entryPrice'])
                print("precio entrada es", entrada)
                amount[0] = float(client.futures_position_information(symbol=symbolo)[-1]['positionAmt'])
                print("el lotaje es: ", amount)
                leverage[0] = float(client.futures_position_information(symbol=symbolo)[-1]['leverage'])
                print("el leverage es: ", leverage)
                bingo[-1] = float(entrada[0] + (comision * amount[0] * entrada[0] * 3.0))
                print("bingo es: ", bingo)
                markPrice[0] = float(client.futures_position_information(symbol=symbolo)[-1]['markPrice'])
                print("el precio mark es:", markPrice)
                if bingo[-1] < markPrice[-1]:
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    client.futures_create_order(symbol=symbolo, side='SELL', type="MARKET", quantity=cantidad,
                                                redueOnly='True')
                    telegram_send.send(messages=["Profit!", symbolo])
                    break
                if client.futures_position_information(symbol=symbolo)[-1]['positionAmt'] == '0.00000':
                    time.sleep(2)
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    telegram_send.send(messages=["Buy cerrado y ordenes cerradas!", symbolo])
                    break
                if len(strongBuy_list3) != 1:
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    client.futures_create_order(symbol=symbolo, side='SELL', type="MARKET", quantity=cantidad,
                                                redueOnly='True')
                    telegram_send.send(messages=["Buy cerrado y ordenes cerradas!", symbolo])
                    break
                # else: print("na de na")

        elif len(strongSell_list3) != 0:
            f = c.shape
            g = f[0] - 1
            cc = c[g]
            CC = float(round((cc), 4))
            print("el close es:", CC)
            # tpl=cc+((1.5)*datr1d)
            sls = cc + ((1.5) * datr1h)
            tps = cc - ((1.5) * datr1h)
            # sll=cc-((1.5)*datr1d)
            # TPL = float(round((tpl) , 4))
            # SLL = float(round((sll) , 4))
            TPS = float(round((tps), 4))
            SLS = float(round((sls), 4))
            # print("TPL:",TPL)
            # print("SLL:",SLL)
            print("TPS:", TPS)
            print("SLS:", SLS)
            client.futures_change_leverage(symbol=symbolo, leverage=leve)
            order_sell = client.futures_create_order(
                symbol=symbolo,
                side='SELL',
                type='MARKET',
                quantity=cantidad,
                # reduceOnly=True
            )

            order_Sell_ID = order_sell['orderId']
            print(order_sell)
            precio = order_sell['price']
            print("Order ID: " + str(order_Sell_ID))
            order_sellsl = client.futures_create_order(
                symbol=symbolo,
                # quantity=cantidad,
                side='BUY',
                type='STOP_MARKET',
                stopPrice=SLS,
                closePosition=True
                # activationPrice=cc*1.01,
                # callbackRate=1,
                ##reduceOnly='True'
            )
            order_Sellsl_ID = order_sellsl['orderId']
            print(order_sellsl)
            print("Or der IDsl: " + str(order_Sellsl_ID))
            order_selltp = client.futures_create_order(
                symbol=symbolo,
                side='BUY',
                type='TAKE_PROFIT_MARKET',
                stopPrice=TPS,
                closePosition=True
            )

            order_Selltp_ID = order_selltp['orderId']
            print(order_selltp)
            print("Order ID: " + str(order_Selltp_ID))
            telegram_send.send(messages=["Sell order!", symbolo])
            print("Order ID: " + str(order_Sell_ID))
            print("Order IDtp: " + str(order_Selltp_ID))
            print("Order IDsl: " + str(order_Sellsl_ID))
            mylista1[0] = (order_Sell_ID)
            mylista1[1] = (order_Selltp_ID)
            mylista1[2] = (order_Sellsl_ID)
            print(str(mylista1))
            print("primeras órdenes")
            print(client.futures_position_information(symbol=symbolo))
            abc = time.time() + (60 * 60)
            while True:
                print(client.futures_position_information(symbol=symbolo))
                orders = client.futures_get_open_orders(symbol=symbolo)
                now = datetime.now()
                fecha = now.strftime("%d-%m-%y %H:%M:%S")
                lista = [symbolo]
                print("length lista es:")
                print(len(lista))
                print("length simbolo es:")
                print(len(symbolo))
                print(now)
                print(order_Sell_ID)
                # openOrder = client.get_order(symbol=symbolo,orderId=order_Sell_ID)
                ##print(openOrder)
                ##orderId = openOrder[0]['orderId']
                ##print(orderId)
                strongBuy_list3 = []
                strongSell_list3 = []
                for i in lista:
                    tesla = TA_Handler()
                    tesla.set_symbol_as(i)
                    tesla.set_exchange_as_crypto_or_stock("BINANCE")
                    tesla.set_screener_as_crypto()
                    tesla.set_interval_as(Interval.INTERVAL_1_HOUR)
                    print(i)
                    try:
                        print(tesla.get_analysis().summary)
                    except Exception as e:
                        print("No Data")
                        continue

                    if ((tesla.get_analysis().summary)["RECOMMENDATION"]) == "STRONG_SELL":
                        print(f" Compar más fuerte {i}", fecha)
                        strongSell_list3.append(i)

                print("*** STRONG SELL LIST              1 hora***")

                print(strongSell_list3)

                entrada[0] = float(client.futures_position_information(symbol=symbolo)[-1]['entryPrice'])
                print("precio entrada es", entrada)
                amount[0] = float(client.futures_position_information(symbol=symbolo)[-1]['positionAmt'])
                print("el lotaje es: ", amount)
                leverage[0] = float(client.futures_position_information(symbol=symbolo)[-1]['leverage'])
                print("el leverage es: ", leverage)
                bingo[-1] = float(entrada[0] + (comision * amount[0] * entrada[0] * 3.0))
                print("bingo es: ", bingo)
                markPrice[0] = float(client.futures_position_information(symbol=symbolo)[-1]['markPrice'])
                print("el precio mark es:", markPrice)
                if bingo[-1] > markPrice[-1]:
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    client.futures_create_order(symbol=symbolo, side='BUY', type="MARKET", quantity=cantidad,
                                                redueOnly='True')
                    telegram_send.send(messages=["Profit!", symbolo])
                    break
                if client.futures_position_information(symbol=symbolo)[-1]['positionAmt'] == '0.00000':
                    time.sleep(2)
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    telegram_send.send(messages=["Buy cerrado y ordenes cerradas!", symbolo])
                    break
                if len(strongSell_list3) != 1:
                    client.futures_cancel_all_open_orders(symbol=symbolo)
                    client.futures_create_order(symbol=symbolo, side='BUY', type="MARKET", quantity=cantidad,
                                                redueOnly='True')
                    telegram_send.send(messages=["Buy cerrado y ordenes cerradas!", symbolo])
                    break
                # else: print("na de na")

        """
        elif len(strongBuy_list4)!=0:
                 f=c.shape
                 g=f[0]-1
                 cc=c[g]
                 CC = float(round((cc) , 4))
                 print("el close es:",CC)
                 tpl=cc+((1.5)*datr30m)
                 #sls=cc+((1.5)*datr1d)
                 #tps=cc-((1.5)*datr1d)
                 sll=cc-((1.5)*datr30m)
                 TPL = float(round((tpl) , 4))
                 SLL = float(round((sll) , 4))    
                 #TPS = float(round((tps) , 4))
                 #SLS = float(round((sls) , 4))
                 print("TPL:",TPL)
                 print("SLL:",SLL)
                 #print("TPS:",TPS)
                 #print("SLS:",SLS)
                 client.futures_change_leverage(symbol=symbolo,leverage=leve)

                 order_buy=client.futures_create_order(
                                                     symbol=symbolo,
                                                     side='BUY',
                                                     type='MARKET',
                                                     quantity=cantidad,
                                                     #reduceOnly=True
                                                     )
                 order_Buy_ID=order_buy['orderId']
                 print(order_buy)
                 precio=order_buy['price']
                 print("Order ID: " + str(order_Buy_ID))


                 order_buysl=client.futures_create_order(
                                                     symbol=symbolo,
                                                     #quantity=cantidad,
                                                     side='SELL',
                                                     type='STOP_MARKET',
                                                     stopPrice=SLL,
                                                     closePosition=True
                                                     #activationPrice=cc*1.01,
                                                     #callbackRate=1,
                                                     ##reduceOnly='True'
                                                     )
                 order_Buysl_ID=order_buysl['orderId']
                 print(order_buysl)
                 print("Or der IDsl: " + str(order_Buysl_ID))    
                 order_buytp=client.futures_create_order(
                                                 symbol=symbolo,
                                                 side='SELL',
                                                 type='TAKE_PROFIT_MARKET',
                                                 stopPrice=TPL,
                                                 closePosition=True
                                                 )
                 order_Buytp_ID=order_buytp['orderId']
                 print(order_buytp)
                 print("Or der IDtp: " + str(order_Buytp_ID))
                 telegram_send.send(messages=["Buy order!",symbolo])
                 print("Order ID: " + str(order_Buy_ID))
                 print("Order IDtp: " + str(order_Buytp_ID))
                 print("Order IDsl: " + str(order_Buysl_ID))
                 mylista1[0]=(order_Buy_ID)
                 mylista1[1]=(order_Buytp_ID)
                 mylista1[2]=(order_Buysl_ID)
                 print(str(mylista1))
                 print("primeras órdenes")
                 print(client.futures_position_information(symbol=symbolo))
                 abc=time.time()+(60*3.00)
                 while True:
                     print(client.futures_position_information(symbol=symbolo))
                     orders = client.futures_get_open_orders(symbol=symbolo)
                     now = datetime.now()
                     fecha = now.strftime("%d-%m-%y %H:%M:%S")
                     lista = [symbolo]
                     print("length lista es:")
                     print(len(lista))
                     print("length simbolo es:")
                     print(len(symbolo))
                     print(now)
                     print(order_Buy_ID)
                     #openOrder = client.get_order(symbol=symbolo,orderId=order_Buy_ID)
                     ##print(openOrder)
                     ##orderId = openOrder[0]['orderId']
                     ##print(orderId)
                     strongBuy_list4 = []
                     strongSell_list4 = []
                     for i in lista:
                         tesla = TA_Handler()
                         tesla.set_symbol_as(i)
                         tesla.set_exchange_as_crypto_or_stock("BINANCE")
                         tesla.set_screener_as_crypto()
                         tesla.set_interval_as(Interval.INTERVAL_30_MINUTES)
                         print(i)
                         try:
                            print(tesla.get_analysis().summary)
                         except Exception as e:
                           print("No Data")
                           continue
                         if((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_BUY":
                             print(f" Compar más fuerte {i}", fecha)
                             strongBuy_list4.append(i)
                     print("*** STRONG buy LIST              30 min***")

                     print(strongBuy_list4)
                     entrada[0]=float(client.futures_position_information(symbol=symbolo)[-1]['entryPrice'])
                     print("precio entrada es",entrada)
                     amount[0]=float(client.futures_position_information(symbol=symbolo)[-1]['positionAmt'])
                     print("el lotaje es: ",amount)
                     leverage[0]=float(client.futures_position_information(symbol=symbolo)[-1]['leverage'])
                     print("el leverage es: ",leverage)
                     bingo[-1]=float(entrada[0]+(comision*amount[0]*entrada[0]*3.0))
                     print("bingo es: ",bingo)
                     markPrice[0]=float(client.futures_position_information(symbol=symbolo)[-1]['markPrice'])
                     print("el precio mark es:", markPrice)
                     if bingo[-1] < markPrice[-1]:
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         client.futures_create_order(symbol=symbolo,side='SELL', type="MARKET", quantity=cantidad, redueOnly='True')
                         telegram_send.send(messages=["Profit!",symbolo])
                         break
                     if client.futures_position_information(symbol=symbolo)[-1]['positionAmt'] == '0.00000':
                         time.sleep(2)
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         telegram_send.send(messages=["Buy cerrado y ordenes cerradas!",symbolo])
                         break
                     if len(strongBuy_list4) !=1:
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         client.futures_create_order(symbol=symbolo,side='SELL', type="MARKET", quantity=cantidad, redueOnly='True')
                         telegram_send.send(messages=["Buy cerrado y ordenes cerradas!",symbolo])
                         break
                     #else: print("na de na")

        elif len(strongSell_list4)!=0:
                 f=c.shape
                 g=f[0]-1
                 cc=c[g]
                 CC = float(round((cc) , 4))
                 print("el close es:",CC)
                 #tpl=cc+((1.5)*datr1d)
                 sls=cc+((1.5)*datr30m)
                 tps=cc-((1.5)*datr30m)
                 #sll=cc-((1.5)*datr1d)
                 #TPL = float(round((tpl) , 4))
                 #SLL = float(round((sll) , 4))    
                 TPS = float(round((tps) , 4))
                 SLS = float(round((sls) , 4))
                 #print("TPL:",TPL)
                 #print("SLL:",SLL)
                 print("TPS:",TPS)
                 print("SLS:",SLS)
                 client.futures_change_leverage(symbol=symbolo,leverage=leve)
                 order_sell=client.futures_create_order(
                                         symbol=symbolo,
                                         side='SELL',
                                         type='MARKET',
                                         quantity=cantidad,
                                         #reduceOnly=True
                                         )

                 order_Sell_ID=order_sell['orderId']
                 print(order_sell)
                 precio=order_sell['price']
                 print("Order ID: " + str(order_Sell_ID))
                 order_sellsl=client.futures_create_order(
                                                     symbol=symbolo,
                                                     #quantity=cantidad,
                                                     side='BUY',
                                                     type='STOP_MARKET',
                                                     stopPrice=SLS,
                                                     closePosition=True
                                                     #activationPrice=cc*1.01,
                                                     #callbackRate=1,
                                                     ##reduceOnly='True'
                                                     )
                 order_Sellsl_ID=order_sellsl['orderId']
                 print(order_sellsl)
                 print("Or der IDsl: " + str(order_Sellsl_ID))    
                 order_selltp=client.futures_create_order(
                                                             symbol=symbolo,
                                                             side='BUY',
                                                             type='TAKE_PROFIT_MARKET',
                                                             stopPrice=TPS,
                                                             closePosition=True
                                                             )

                 order_Selltp_ID=order_selltp['orderId']
                 print(order_selltp)
                 print("Order ID: " + str(order_Selltp_ID))
                 telegram_send.send(messages=["Sell order!",symbolo])
                 print("Order ID: " + str(order_Sell_ID))
                 print("Order IDtp: " + str(order_Selltp_ID))
                 print("Order IDsl: " + str(order_Sellsl_ID))
                 mylista1[0]=(order_Sell_ID)
                 mylista1[1]=(order_Selltp_ID)
                 mylista1[2]=(order_Sellsl_ID)
                 print(str(mylista1))
                 print("primeras órdenes")
                 print(client.futures_position_information(symbol=symbolo))
                 abc=time.time()+(60*3.00)
                 while True:
                     print(client.futures_position_information(symbol=symbolo))
                     orders = client.futures_get_open_orders(symbol=symbolo)
                     now = datetime.now()
                     fecha = now.strftime("%d-%m-%y %H:%M:%S")
                     lista = [symbolo]
                     print("length lista es:")
                     print(len(lista))
                     print("length simbolo es:")
                     print(len(symbolo))
                     print(now)
                     print(order_Sell_ID)
                     #openOrder = client.get_order(symbol=symbolo,orderId=order_Sell_ID)
                     ##print(openOrder)
                     ##orderId = openOrder[0]['orderId']
                     ##print(orderId)
                     strongBuy_list4 = []
                     strongSell_list4 = []
                     for i in lista:
                         tesla = TA_Handler()
                         tesla.set_symbol_as(i)
                         tesla.set_exchange_as_crypto_or_stock("BINANCE")
                         tesla.set_screener_as_crypto()
                         tesla.set_interval_as(Interval.INTERVAL_30_MINUTES)
                         print(i)
                         try:
                            print(tesla.get_analysis().summary)
                         except Exception as e:
                           print("No Data")
                           continue

                         if((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_SELL":
                             print(f" Compar más fuerte {i}", fecha)
                             strongSell_list4.append(i)
                     print("*** STRONG SELL LIST              30 min***")

                     print(strongSell_list4)


                     entrada[0]=float(client.futures_position_information(symbol=symbolo)[-1]['entryPrice'])
                     print("precio entrada es",entrada)
                     amount[0]=float(client.futures_position_information(symbol=symbolo)[-1]['positionAmt'])
                     print("el lotaje es: ",amount)
                     leverage[0]=float(client.futures_position_information(symbol=symbolo)[-1]['leverage'])
                     print("el leverage es: ",leverage)
                     bingo[-1]=float(entrada[0]+(comision*amount[0]*entrada[0]*3.0))
                     print("bingo es: ",bingo)
                     markPrice[0]=float(client.futures_position_information(symbol=symbolo)[-1]['markPrice'])
                     print("el precio mark es:", markPrice)
                     if bingo[-1] > markPrice[-1]:
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         client.futures_create_order(symbol=symbolo,side='BUY', type="MARKET", quantity=cantidad, redueOnly='True')
                         telegram_send.send(messages=["Profit!",symbolo])
                         break
                     if client.futures_position_information(symbol=symbolo)[-1]['positionAmt'] == '0.00000':
                         time.sleep(2)
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         telegram_send.send(messages=["Buy cerrado y ordenes cerradas!",symbolo])
                         break
                     if len(strongSell_list4) !=1:
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         client.futures_create_order(symbol=symbolo,side='BUY', type="MARKET", quantity=cantidad, redueOnly='True')
                         telegram_send.send(messages=["Buy cerrado y ordenes cerradas!",symbolo])
                         break
                     #else: print("na de na")

        elif len(strongBuy_list5)!=0:
                 f=c.shape
                 g=f[0]-1
                 cc=c[g]
                 CC = float(round((cc) , 4))
                 print("el close es:",CC)
                 tpl=cc+((1.5)*datr15m)
                 #sls=cc+((1.5)*datr1d)
                 #tps=cc-((1.5)*datr1d)
                 sll=cc-((1.5)*datr15m)
                 TPL = float(round((tpl) , 4))
                 SLL = float(round((sll) , 4))    
                 #TPS = float(round((tps) , 4))
                 #SLS = float(round((sls) , 4))
                 print("TPL:",TPL)
                 print("SLL:",SLL)
                 #print("TPS:",TPS)
                 #print("SLS:",SLS)
                 client.futures_change_leverage(symbol=symbolo,leverage=leve)

                 order_buy=client.futures_create_order(
                                                     symbol=symbolo,
                                                     side='BUY',
                                                     type='MARKET',
                                                     quantity=cantidad,
                                                     #reduceOnly=True
                                                     )
                 order_Buy_ID=order_buy['orderId']
                 print(order_buy)
                 precio=order_buy['price']
                 print("Order ID: " + str(order_Buy_ID))


                 order_buysl=client.futures_create_order(
                                                     symbol=symbolo,
                                                     #quantity=cantidad,
                                                     side='SELL',
                                                     type='STOP_MARKET',
                                                     stopPrice=SLL,
                                                     closePosition=True
                                                     #activationPrice=cc*1.01,
                                                     #callbackRate=1,
                                                     ##reduceOnly='True'
                                                     )
                 order_Buysl_ID=order_buysl['orderId']
                 print(order_buysl)
                 print("Or der IDsl: " + str(order_Buysl_ID))    
                 order_buytp=client.futures_create_order(
                                                 symbol=symbolo,
                                                 side='SELL',
                                                 type='TAKE_PROFIT_MARKET',
                                                 stopPrice=TPL,
                                                 closePosition=True
                                                 )
                 order_Buytp_ID=order_buytp['orderId']
                 print(order_buytp)
                 print("Or der IDtp: " + str(order_Buytp_ID))
                 telegram_send.send(messages=["Buy order!",symbolo])
                 print("Order ID: " + str(order_Buy_ID))
                 print("Order IDtp: " + str(order_Buytp_ID))
                 print("Order IDsl: " + str(order_Buysl_ID))
                 mylista1[0]=(order_Buy_ID)
                 mylista1[1]=(order_Buytp_ID)
                 mylista1[2]=(order_Buysl_ID)
                 print(str(mylista1))
                 print("primeras órdenes")
                 print(client.futures_position_information(symbol=symbolo))
                 abc=time.time()+(60*15)
                 while True:
                     print(client.futures_position_information(symbol=symbolo))
                     orders = client.futures_get_open_orders(symbol=symbolo)
                     now = datetime.now()
                     fecha = now.strftime("%d-%m-%y %H:%M:%S")
                     lista = [symbolo]
                     print("length lista es:")
                     print(len(lista))
                     print("length simbolo es:")
                     print(len(symbolo))
                     print(now)
                     print(order_Buy_ID)
                     #openOrder = client.get_order(symbol=symbolo,orderId=order_Buy_ID)
                     ##print(openOrder)
                     ##orderId = openOrder[0]['orderId']
                     ##print(orderId)
                     strongBuy_list5 = []
                     strongSell_list5 = []
                     for i in lista:
                         tesla = TA_Handler()
                         tesla.set_symbol_as(i)
                         tesla.set_exchange_as_crypto_or_stock("BINANCE")
                         tesla.set_screener_as_crypto()
                         tesla.set_interval_as(Interval.INTERVAL_15_MINUTES)
                         print(i)
                         try:
                            print(tesla.get_analysis().summary)
                         except Exception as e:
                           print("No Data")
                           continue
                         if((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_BUY":
                             print(f" Compar más fuerte {i}", fecha)
                             strongBuy_list5.append(i)
                     print("*** STRONG buy LIST              15 min***")

                     print(strongBuy_list5)
                     entrada[0]=float(client.futures_position_information(symbol=symbolo)[-1]['entryPrice'])
                     print("precio entrada es",entrada)
                     amount[0]=float(client.futures_position_information(symbol=symbolo)[-1]['positionAmt'])
                     print("el lotaje es: ",amount)
                     leverage[0]=float(client.futures_position_information(symbol=symbolo)[-1]['leverage'])
                     print("el leverage es: ",leverage)
                     bingo[-1]=float(entrada[0]+(comision*amount[0]*entrada[0]*3.0))
                     print("bingo es: ",bingo)
                     markPrice[0]=float(client.futures_position_information(symbol=symbolo)[-1]['markPrice'])
                     print("el precio mark es:", markPrice)
                     if bingo[-1] < markPrice[-1]:
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         client.futures_create_order(symbol=symbolo,side='SELL', type="MARKET", quantity=cantidad, redueOnly='True')
                         telegram_send.send(messages=["Profit!",symbolo])
                         break
                     if client.futures_position_information(symbol=symbolo)[-1]['positionAmt'] == '0.00000':
                         time.sleep(2)
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         telegram_send.send(messages=["Buy cerrado y ordenes cerradas!",symbolo])
                         break
                     if len(strongBuy_list5) !=1:
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         client.futures_create_order(symbol=symbolo,side='SELL', type="MARKET", quantity=cantidad, redueOnly='True')
                         telegram_send.send(messages=["Buy cerrado y ordenes cerradas!",symbolo])
                         break
                     #else: print("na de na")

        elif len(strongSell_list5)!=0:
                 f=c.shape
                 g=f[0]-1
                 cc=c[g]
                 CC = float(round((cc) , 4))
                 print("el close es:",CC)
                 #tpl=cc+((1.5)*datr1d)
                 sls=cc+((1.5)*datr15m)
                 tps=cc-((1.5)*datr15m)
                 #sll=cc-((1.5)*datr1d)
                 #TPL = float(round((tpl) , 4))
                 #SLL = float(round((sll) , 4))    
                 TPS = float(round((tps) , 4))
                 SLS = float(round((sls) , 4))
                 #print("TPL:",TPL)
                 #print("SLL:",SLL)
                 print("TPS:",TPS)
                 print("SLS:",SLS)
                 client.futures_change_leverage(symbol=symbolo,leverage=leve)
                 order_sell=client.futures_create_order(
                                         symbol=symbolo,
                                         side='SELL',
                                         type='MARKET',
                                         quantity=cantidad,
                                         #reduceOnly=True
                                         )

                 order_Sell_ID=order_sell['orderId']
                 print(order_sell)
                 precio=order_sell['price']
                 print("Order ID: " + str(order_Sell_ID))
                 order_sellsl=client.futures_create_order(
                                                     symbol=symbolo,
                                                     #quantity=cantidad,
                                                     side='BUY',
                                                     type='STOP_MARKET',
                                                     stopPrice=SLS,
                                                     closePosition=True
                                                     #activationPrice=cc*1.01,
                                                     #callbackRate=1,
                                                     ##reduceOnly='True'
                                                     )
                 order_Sellsl_ID=order_sellsl['orderId']
                 print(order_sellsl)
                 print("Or der IDsl: " + str(order_Sellsl_ID))    
                 order_selltp=client.futures_create_order(
                                                             symbol=symbolo,
                                                             side='BUY',
                                                             type='TAKE_PROFIT_MARKET',
                                                             stopPrice=TPS,
                                                             closePosition=True
                                                             )

                 order_Selltp_ID=order_selltp['orderId']
                 print(order_selltp)
                 print("Order ID: " + str(order_Selltp_ID))
                 telegram_send.send(messages=["Sell order!",symbolo])
                 print("Order ID: " + str(order_Sell_ID))
                 print("Order IDtp: " + str(order_Selltp_ID))
                 print("Order IDsl: " + str(order_Sellsl_ID))
                 mylista1[0]=(order_Sell_ID)
                 mylista1[1]=(order_Selltp_ID)
                 mylista1[2]=(order_Sellsl_ID)
                 print(str(mylista1))
                 print("primeras órdenes")
                 print(client.futures_position_information(symbol=symbolo))
                 abc=time.time()+(60*15)
                 while True:
                     print(client.futures_position_information(symbol=symbolo))
                     orders = client.futures_get_open_orders(symbol=symbolo)
                     now = datetime.now()
                     fecha = now.strftime("%d-%m-%y %H:%M:%S")
                     lista = [symbolo]
                     print("length lista es:")
                     print(len(lista))
                     print("length simbolo es:")
                     print(len(symbolo))
                     print(now)
                     print(order_Sell_ID)
                     #openOrder = client.get_order(symbol=symbolo,orderId=order_Sell_ID)
                     ##print(openOrder)
                     ##orderId = openOrder[0]['orderId']
                     ##print(orderId)
                     strongBuy_list5 = []
                     strongSell_list5 = []
                     for i in lista:
                         tesla = TA_Handler()
                         tesla.set_symbol_as(i)
                         tesla.set_exchange_as_crypto_or_stock("BINANCE")
                         tesla.set_screener_as_crypto()
                         tesla.set_interval_as(Interval.INTERVAL_15_MINUTES)
                         print(i)
                         try:
                            print(tesla.get_analysis().summary)
                         except Exception as e:
                           print("No Data")
                           continue

                         if((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_SELL":
                             print(f" Compar más fuerte {i}", fecha)
                             strongSell_list5.append(i)
                     print("*** STRONG SELL LIST              15 min***")

                     print(strongSell_list5)


                     entrada[0]=float(client.futures_position_information(symbol=symbolo)[-1]['entryPrice'])
                     print("precio entrada es",entrada)
                     amount[0]=float(client.futures_position_information(symbol=symbolo)[-1]['positionAmt'])
                     print("el lotaje es: ",amount)
                     leverage[0]=float(client.futures_position_information(symbol=symbolo)[-1]['leverage'])
                     print("el leverage es: ",leverage)
                     bingo[-1]=float(entrada[0]+(comision*amount[0]*entrada[0]*3.0))
                     print("bingo es: ",bingo)
                     markPrice[0]=float(client.futures_position_information(symbol=symbolo)[-1]['markPrice'])
                     print("el precio mark es:", markPrice)
                     if bingo[-1] > markPrice[-1]:
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         client.futures_create_order(symbol=symbolo,side='BUY', type="MARKET", quantity=cantidad, redueOnly='True')
                         telegram_send.send(messages=["Profit!",symbolo])
                         break
                     if client.futures_position_information(symbol=symbolo)[-1]['positionAmt'] == '0.00000':
                         time.sleep(2)
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         telegram_send.send(messages=["Buy cerrado y ordenes cerradas!",symbolo])
                         break
                     if len(strongSell_list5) !=1:
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         client.futures_create_order(symbol=symbolo,side='BUY', type="MARKET", quantity=cantidad, redueOnly='True')
                         telegram_send.send(messages=["Buy cerrado y ordenes cerradas!",symbolo])
                         break
                     #else: print("na de na")

        elif len(strongBuy_list6)!=0:
                 f=c.shape
                 g=f[0]-1
                 cc=c[g]
                 CC = float(round((cc) , 4))
                 print("el close es:",CC)
                 tpl=cc+((1.5)*datr5m)
                 #sls=cc+((1.5)*datr1d)
                 #tps=cc-((1.5)*datr1d)
                 sll=cc-((1.5)*datr5m)
                 TPL = float(round((tpl) , 4))
                 SLL = float(round((sll) , 4))    
                 #TPS = float(round((tps) , 4))
                 #SLS = float(round((sls) , 4))
                 print("TPL:",TPL)
                 print("SLL:",SLL)
                 #print("TPS:",TPS)
                 #print("SLS:",SLS)
                 client.futures_change_leverage(symbol=symbolo,leverage=leve)

                 order_buy=client.futures_create_order(
                                                     symbol=symbolo,
                                                     side='BUY',
                                                     type='MARKET',
                                                     quantity=cantidad,
                                                     #reduceOnly=True
                                                     )
                 order_Buy_ID=order_buy['orderId']
                 print(order_buy)
                 precio=order_buy['price']
                 print("Order ID: " + str(order_Buy_ID))


                 order_buysl=client.futures_create_order(
                                                     symbol=symbolo,
                                                     #quantity=cantidad,
                                                     side='SELL',
                                                     type='STOP_MARKET',
                                                     stopPrice=SLL,
                                                     closePosition=True
                                                     #activationPrice=cc*1.01,
                                                     #callbackRate=1,
                                                     ##reduceOnly='True'
                                                     )
                 order_Buysl_ID=order_buysl['orderId']
                 print(order_buysl)
                 print("Or der IDsl: " + str(order_Buysl_ID))    
                 order_buytp=client.futures_create_order(
                                                 symbol=symbolo,
                                                 side='SELL',
                                                 type='TAKE_PROFIT_MARKET',
                                                 stopPrice=TPL,
                                                 closePosition=True
                                                 )
                 order_Buytp_ID=order_buytp['orderId']
                 print(order_buytp)
                 print("Or der IDtp: " + str(order_Buytp_ID))
                 telegram_send.send(messages=["Buy order!",symbolo])
                 print("Order ID: " + str(order_Buy_ID))
                 print("Order IDtp: " + str(order_Buytp_ID))
                 print("Order IDsl: " + str(order_Buysl_ID))
                 mylista1[0]=(order_Buy_ID)
                 mylista1[1]=(order_Buytp_ID)
                 mylista1[2]=(order_Buysl_ID)
                 print(str(mylista1))
                 print("primeras órdenes")
                 print(client.futures_position_information(symbol=symbolo))
                 abc=time.time()+(60*5)
                 while True:
                     print(client.futures_position_information(symbol=symbolo))
                     orders = client.futures_get_open_orders(symbol=symbolo)
                     now = datetime.now()
                     fecha = now.strftime("%d-%m-%y %H:%M:%S")
                     lista = [symbolo]
                     print("length lista es:")
                     print(len(lista))
                     print("length simbolo es:")
                     print(len(symbolo))
                     print(now)
                     print(order_Buy_ID)
                     #openOrder = client.get_order(symbol=symbolo,orderId=order_Buy_ID)
                     ##print(openOrder)
                     ##orderId = openOrder[0]['orderId']
                     ##print(orderId)
                     strongBuy_list6 = []
                     strongSell_list6 = []
                     for i in lista:
                         tesla = TA_Handler()
                         tesla.set_symbol_as(i)
                         tesla.set_exchange_as_crypto_or_stock("BINANCE")
                         tesla.set_screener_as_crypto()
                         tesla.set_interval_as(Interval.INTERVAL_5_MINUTES)
                         print(i)
                         try:
                            print(tesla.get_analysis().summary)
                         except Exception as e:
                           print("No Data")
                           continue
                         if((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_BUY":
                             print(f" Compar más fuerte {i}", fecha)
                             strongBuy_list6.append(i)
                     print("*** STRONG buy LIST              5 min***")

                     print(strongBuy_list6)
                     entrada[0]=float(client.futures_position_information(symbol=symbolo)[-1]['entryPrice'])
                     print("precio entrada es",entrada)
                     amount[0]=float(client.futures_position_information(symbol=symbolo)[-1]['positionAmt'])
                     print("el lotaje es: ",amount)
                     leverage[0]=float(client.futures_position_information(symbol=symbolo)[-1]['leverage'])
                     print("el leverage es: ",leverage)
                     bingo[-1]=float(entrada[0]+(comision*amount[0]*entrada[0]*3.0))
                     print("bingo es: ",bingo)
                     markPrice[0]=float(client.futures_position_information(symbol=symbolo)[-1]['markPrice'])
                     print("el precio mark es:", markPrice)
                     if bingo[-1] < markPrice[-1]:
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         client.futures_create_order(symbol=symbolo,side='SELL', type="MARKET", quantity=cantidad, redueOnly='True')
                         telegram_send.send(messages=["Profit!",symbolo])
                         break
                     if client.futures_position_information(symbol=symbolo)[-1]['positionAmt'] == '0.00000':
                         time.sleep(2)
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         telegram_send.send(messages=["Buy cerrado y ordenes cerradas!",symbolo])
                         break
                     if len(strongBuy_list6) !=1:
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         client.futures_create_order(symbol=symbolo,side='SELL', type="MARKET", quantity=cantidad, redueOnly='True')
                         telegram_send.send(messages=["Buy cerrado y ordenes cerradas!",symbolo])
                         break
                     #else: print("na de na")

        elif len(strongSell_list6)!=0:
                 f=c.shape
                 g=f[0]-1
                 cc=c[g]
                 CC = float(round((cc) , 4))
                 print("el close es:",CC)
                 #tpl=cc+((1.5)*datr1d)
                 sls=cc+((1.5)*datr5m)
                 tps=cc-((1.5)*datr5m)
                 #sll=cc-((1.5)*datr1d)
                 #TPL = float(round((tpl) , 4))
                 #SLL = float(round((sll) , 4))    
                 TPS = float(round((tps) , 4))
                 SLS = float(round((sls) , 4))
                 #print("TPL:",TPL)
                 #print("SLL:",SLL)
                 print("TPS:",TPS)
                 print("SLS:",SLS)
                 client.futures_change_leverage(symbol=symbolo,leverage=leve)
                 order_sell=client.futures_create_order(
                                         symbol=symbolo,
                                         side='SELL',
                                         type='MARKET',
                                         quantity=cantidad,
                                         #reduceOnly=True
                                         )

                 order_Sell_ID=order_sell['orderId']
                 print(order_sell)
                 precio=order_sell['price']
                 print("Order ID: " + str(order_Sell_ID))
                 order_sellsl=client.futures_create_order(
                                                     symbol=symbolo,
                                                     #quantity=cantidad,
                                                     side='BUY',
                                                     type='STOP_MARKET',
                                                     stopPrice=SLS,
                                                     closePosition=True
                                                     #activationPrice=cc*1.01,
                                                     #callbackRate=1,
                                                     ##reduceOnly='True'
                                                     )
                 order_Sellsl_ID=order_sellsl['orderId']
                 print(order_sellsl)
                 print("Or der IDsl: " + str(order_Sellsl_ID))    
                 order_selltp=client.futures_create_order(
                                                             symbol=symbolo,
                                                             side='BUY',
                                                             type='TAKE_PROFIT_MARKET',
                                                             stopPrice=TPS,
                                                             closePosition=True
                                                             )

                 order_Selltp_ID=order_selltp['orderId']
                 print(order_selltp)
                 print("Order ID: " + str(order_Selltp_ID))
                 telegram_send.send(messages=["Sell order!",symbolo])
                 print("Order ID: " + str(order_Sell_ID))
                 print("Order IDtp: " + str(order_Selltp_ID))
                 print("Order IDsl: " + str(order_Sellsl_ID))
                 mylista1[0]=(order_Sell_ID)
                 mylista1[1]=(order_Selltp_ID)
                 mylista1[2]=(order_Sellsl_ID)
                 print(str(mylista1))
                 print("primeras órdenes")
                 print(client.futures_position_information(symbol=symbolo))
                 abc=time.time()+(60*5)
                 while True:
                     print(client.futures_position_information(symbol=symbolo))
                     orders = client.futures_get_open_orders(symbol=symbolo)
                     now = datetime.now()
                     fecha = now.strftime("%d-%m-%y %H:%M:%S")
                     lista = [symbolo]
                     print("length lista es:")
                     print(len(lista))
                     print("length simbolo es:")
                     print(len(symbolo))
                     print(now)
                     print(order_Sell_ID)
                     #openOrder = client.get_order(symbol=symbolo,orderId=order_Sell_ID)
                     ##print(openOrder)
                     ##orderId = openOrder[0]['orderId']
                     ##print(orderId)
                     strongBuy_list6 = []
                     strongSell_list6 = []
                     for i in lista:
                         tesla = TA_Handler()
                         tesla.set_symbol_as(i)
                         tesla.set_exchange_as_crypto_or_stock("BINANCE")
                         tesla.set_screener_as_crypto()
                         tesla.set_interval_as(Interval.INTERVAL_5_MINUTES)
                         print(i)
                         try:
                            print(tesla.get_analysis().summary)
                         except Exception as e:
                           print("No Data")
                           continue

                         if((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_SELL":
                             print(f" Compar más fuerte {i}", fecha)
                             strongSell_list6.append(i)
                     print("*** STRONG SELL LIST              5 min***")

                     print(strongSell_list6)


                     entrada[0]=float(client.futures_position_information(symbol=symbolo)[-1]['entryPrice'])
                     print("precio entrada es",entrada)
                     amount[0]=float(client.futures_position_information(symbol=symbolo)[-1]['positionAmt'])
                     print("el lotaje es: ",amount)
                     leverage[0]=float(client.futures_position_information(symbol=symbolo)[-1]['leverage'])
                     print("el leverage es: ",leverage)
                     bingo[-1]=float(entrada[0]+(comision*amount[0]*entrada[0]*3.0))
                     print("bingo es: ",bingo)
                     markPrice[0]=float(client.futures_position_information(symbol=symbolo)[-1]['markPrice'])
                     print("el precio mark es:", markPrice)
                     if bingo[-1] > markPrice[-1]:
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         client.futures_create_order(symbol=symbolo,side='BUY', type="MARKET", quantity=cantidad, redueOnly='True')
                         telegram_send.send(messages=["Profit!",symbolo])
                         break
                     if client.futures_position_information(symbol=symbolo)[-1]['positionAmt'] == '0.00000':
                         time.sleep(2)
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         telegram_send.send(messages=["Buy cerrado y ordenes cerradas!",symbolo])
                         break
                     if len(strongSell_list6) !=1:
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         client.futures_create_order(symbol=symbolo,side='BUY', type="MARKET", quantity=cantidad, redueOnly='True')
                         telegram_send.send(messages=["Buy cerrado y ordenes cerradas!",symbolo])
                         break
                     #else: print("na de na")



        elif len(strongBuy_list8)!=0:
                 f=c.shape
                 g=f[0]-1
                 cc=c[g]
                 CC = float(round((cc) , 4))
                 print("el close es:",CC)
                 tpl=cc+((1.5)*datr1m)
                 #sls=cc+((1.5)*datr1d)
                 #tps=cc-((1.5)*datr1d)
                 sll=cc-((1.5)*datr1m)
                 TPL = float(round((tpl) , 4))
                 SLL = float(round((sll) , 4))    
                 #TPS = float(round((tps) , 4))
                 #SLS = float(round((sls) , 4))
                 print("TPL:",TPL)
                 print("SLL:",SLL)
                 #print("TPS:",TPS)
                 #print("SLS:",SLS)
                 client.futures_change_leverage(symbol=symbolo,leverage=leve)

                 order_buy=client.futures_create_order(
                                                     symbol=symbolo,
                                                     side='BUY',
                                                     type='MARKET',
                                                     quantity=cantidad,
                                                     #reduceOnly=True
                                                     )
                 order_Buy_ID=order_buy['orderId']
                 print(order_buy)
                 precio=order_buy['price']
                 print("Order ID: " + str(order_Buy_ID))


                 order_buysl=client.futures_create_order(
                                                     symbol=symbolo,
                                                     #quantity=cantidad,
                                                     side='SELL',
                                                     type='STOP_MARKET',
                                                     stopPrice=SLL,
                                                     closePosition=True
                                                     #activationPrice=cc*1.01,
                                                     #callbackRate=1,
                                                     ##reduceOnly='True'
                                                     )
                 order_Buysl_ID=order_buysl['orderId']
                 print(order_buysl)
                 print("Or der IDsl: " + str(order_Buysl_ID))    
                 order_buytp=client.futures_create_order(
                                                 symbol=symbolo,
                                                 side='SELL',
                                                 type='TAKE_PROFIT_MARKET',
                                                 stopPrice=TPL,
                                                 closePosition=True
                                                 )
                 order_Buytp_ID=order_buytp['orderId']
                 print(order_buytp)
                 print("Or der IDtp: " + str(order_Buytp_ID))
                 telegram_send.send(messages=["Buy order!",symbolo])
                 print("Order ID: " + str(order_Buy_ID))
                 print("Order IDtp: " + str(order_Buytp_ID))
                 print("Order IDsl: " + str(order_Buysl_ID))
                 mylista1[0]=(order_Buy_ID)
                 mylista1[1]=(order_Buytp_ID)
                 mylista1[2]=(order_Buysl_ID)
                 print(str(mylista1))
                 print("primeras órdenes")
                 print(client.futures_position_information(symbol=symbolo))
                 abc=time.time() +(60)
                 while True:
                     print(client.futures_position_information(symbol=symbolo))
                     orders = client.futures_get_open_orders(symbol=symbolo)
                     now = datetime.now()
                     fecha = now.strftime("%d-%m-%y %H:%M:%S")
                     lista = [symbolo]
                     print("length lista es:")
                     print(len(lista))
                     print("length simbolo es:")
                     print(len(symbolo))
                     print(now)
                     print(order_Buy_ID)
                     #openOrder = client.get_order(symbol=symbolo,orderId=order_Buy_ID)
                     ##print(openOrder)
                     ##orderId = openOrder[0]['orderId']
                     ##print(orderId)
                     strongBuy_list8 = []
                     strongSell_list8 = []
                     for i in lista:
                         tesla = TA_Handler()
                         tesla.set_symbol_as(i)
                         tesla.set_exchange_as_crypto_or_stock("BINANCE")
                         tesla.set_screener_as_crypto()
                         tesla.set_interval_as(Interval.INTERVAL_1_MINUTE)
                         print(i)
                         try:
                            print(tesla.get_analysis().summary)
                         except Exception as e:
                           print("No Data")
                           continue
                         if((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_BUY":
                             print(f" Compar más fuerte {i}", fecha)
                             strongBuy_list8.append(i)
                     print("*** STRONG BUY LIST              1 min***")

                     print(strongBuy_list8)
                     entrada[0]=float(client.futures_position_information(symbol=symbolo)[-1]['entryPrice'])
                     print("precio entrada es",entrada)
                     amount[0]=float(client.futures_position_information(symbol=symbolo)[-1]['positionAmt'])
                     print("el lotaje es: ",amount)
                     leverage[0]=float(client.futures_position_information(symbol=symbolo)[-1]['leverage'])
                     print("el leverage es: ",leverage)
                     bingo[-1]=float(entrada[0]+(comision*amount[0]*entrada[0]*3.0))
                     print("bingo es: ",bingo)
                     markPrice[0]=float(client.futures_position_information(symbol=symbolo)[-1]['markPrice'])
                     print("el precio mark es:", markPrice)
                     if bingo[-1] < markPrice[-1]:
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         client.futures_create_order(symbol=symbolo,side='SELL', type="MARKET", quantity=cantidad, redueOnly='True')
                         telegram_send.send(messages=["Profit!",symbolo])
                         break
                     if client.futures_position_information(symbol=symbolo)[-1]['positionAmt'] == '0.00000':
                         time.sleep(2)
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         telegram_send.send(messages=["Buy cerrado y ordenes cerradas!",symbolo])
                         break
                     if len(strongBuy_list8) !=1:
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         client.futures_create_order(symbol=symbolo,side='SELL', type="MARKET", quantity=cantidad, redueOnly='True')

                         telegram_send.send(messages=["Buy cerrado y ordenes cerradas!",symbolo])
                         break
                     #else: print("na de na")

        elif len(strongSell_list8)!=0:
                 f=c.shape
                 g=f[0]-1
                 cc=c[g]
                 CC = float(round((cc) , 4))
                 print("el close es:",CC)
                 #tpl=cc+((1.5)*datr1d)
                 sls=cc+((1.5)*datr1m)
                 tps=cc-((1.5)*datr1m)
                 #sll=cc-((1.5)*datr1d)
                 #TPL = float(round((tpl) , 4))
                 #SLL = float(round((sll) , 4))    
                 TPS = float(round((tps) , 4))
                 SLS = float(round((sls) , 4))
                 #print("TPL:",TPL)
                 #print("SLL:",SLL)
                 print("TPS:",TPS)
                 print("SLS:",SLS)
                 client.futures_change_leverage(symbol=symbolo,leverage=leve)
                 order_sell=client.futures_create_order(
                                         symbol=symbolo,
                                         side='SELL',
                                         type='MARKET',
                                         quantity=cantidad,
                                         #reduceOnly=True
                                         )

                 order_Sell_ID=order_sell['orderId']
                 print(order_sell)
                 precio=order_sell['price']

                 print("Order ID: " + str(order_Sell_ID))
                 order_sellsl=client.futures_create_order(
                                                     symbol=symbolo,
                                                     #quantity=cantidad,
                                                     side='BUY',
                                                     type='STOP_MARKET',
                                                     stopPrice=SLS,
                                                     closePosition=True
                                                     #activationPrice=cc*1.01,
                                                     #callbackRate=1,
                                                     ##reduceOnly='True'
                                                     )
                 order_Sellsl_ID=order_sellsl['orderId']
                 print(order_sellsl)
                 print("Or der IDsl: " + str(order_Sellsl_ID))    
                 order_selltp=client.futures_create_order(
                                                             symbol=symbolo,
                                                             side='BUY',
                                                             type='TAKE_PROFIT_MARKET',
                                                             stopPrice=TPS,
                                                             closePosition=True
                                                             )

                 order_Selltp_ID=order_selltp['orderId']
                 print(order_selltp)
                 print("Order ID: " + str(order_Selltp_ID))
                 telegram_send.send(messages=["Sell order!",symbolo])
                 print("Order ID: " + str(order_Sell_ID))
                 print("Order IDtp: " + str(order_Selltp_ID))
                 print("Order IDsl: " + str(order_Sellsl_ID))
                 mylista1[0]=(order_Sell_ID)
                 mylista1[1]=(order_Selltp_ID)
                 mylista1[2]=(order_Sellsl_ID)
                 print(str(mylista1))
                 print("primeras órdenes")
                 print(client.futures_position_information(symbol=symbolo))
                 abc=time.time() + 60
                 while True:
                     print(client.futures_position_information(symbol=symbolo))
                     orders = client.futures_get_open_orders(symbol=symbolo)
                     #order_ids = [order['clientOrderId'] for order in orders]
                     now = datetime.now()
                     fecha = now.strftime("%d-%m-%y %H:%M:%S")
                     lista = [symbolo]
                     print("length lista es:")
                     print(len(lista))
                     print("length simbolo es:")
                     print(len(symbolo))
                     print(now)
                     print(order_Sell_ID)
                     #openOrder = client.get_order(symbol=symbolo,orderId=order_Sell_ID)
                     ##print(openOrder)
                     ##orderId = openOrder[0]['orderId']
                     ##print(orderId)
                     strongBuy_list8 = []
                     strongSell_list8 = []
                     for i in lista:
                         tesla = TA_Handler()
                         tesla.set_symbol_as(i)
                         tesla.set_exchange_as_crypto_or_stock("BINANCE")
                         tesla.set_screener_as_crypto()
                         tesla.set_interval_as(Interval.INTERVAL_1_MINUTE)
                         print(i)
                         try:
                            print(tesla.get_analysis().summary)
                         except Exception as e:
                            print("No Data")
                            continue

                         if((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_SELL":
                             print(f" Compar más fuerte {i}", fecha)
                             strongSell_list8.append(i)
                     print("*** STRONG SELL LIST              1 min***")

                     print(strongSell_list8)


                     entrada[0]=float(client.futures_position_information(symbol=symbolo)[-1]['entryPrice'])
                     print("precio entrada es",entrada)
                     amount[0]=float(client.futures_position_information(symbol=symbolo)[-1]['positionAmt'])
                     print("el lotaje es: ",amount)
                     leverage[0]=float(client.futures_position_information(symbol=symbolo)[-1]['leverage'])
                     print("el leverage es: ",leverage)
                     bingo[-1]=float(entrada[0]+(comision*amount[0]*entrada[0]*3.0))
                     print("bingo es: ",bingo)
                     markPrice[0]=float(client.futures_position_information(symbol=symbolo)[-1]['markPrice'])
                     print("el precio mark es:", markPrice)
                     if bingo[-1] > markPrice[-1]:
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         client.futures_create_order(symbol=symbolo,side='BUY', type="MARKET", quantity=cantidad, redueOnly='True')
                         telegram_send.send(messages=["Profit!",symbolo])
                         break
                     if client.futures_position_information(symbol=symbolo)[-1]['positionAmt'] == '0.00000':
                         time.sleep(2)
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         telegram_send.send(messages=["Buy cerrado y ordenes cerradas!",symbolo])
                         break
                     if len(strongSell_list8) !=1:
                         client.futures_cancel_all_open_orders(symbol=symbolo)
                         client.futures_create_order(symbol=symbolo,side='BUY', type="MARKET", quantity=cantidad, redueOnly='True')
                         telegram_send.send(messages=["Buy cerrado y ordenes cerradas!",symbolo])
                         break
                     #else: print("na de na")           
        else:
            print("no hay strong buy o sell")
            #time.sleep()
        """
ii = +1
