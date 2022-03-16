import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.float_format = '{:.6f}'.format
import ccxt
import matplotlib.pyplot as plt
import ta
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import coint

from tradingview_ta import TA_Handler, Interval, Exchange

def get_ohlcv(symbol, tf):
    df = pd.DataFrame(exchange.fetch_ohlcv(symbol, tf, limit=5000))
    df = df.rename(columns={0: 'timestamp', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'})
    df = df.set_index(df['timestamp'])
    df.index = pd.to_datetime(df.index, unit='ms')
    del df['timestamp']
    return df

def custom_filter(symbol):
    if(
        symbol[-4:] == "/USD"
        and "BULL" not in symbol
        and "HALF" not in symbol
        and "EDGE" not in symbol
        and "BEAR" not in symbol
    ):
        return True

def set_tradingview_data(df, symbol, data_handler, summary):
    #df['TV_exchange'] = np.where(df['TV_symbol'] == symbol, exchange, df['TV_exchange'])

    str_interval = data_handler.interval
    recommendation = "RECOMMENDATION_" + str_interval
    df.loc[symbol, 'exchange'] = data_handler.exchange
    df.loc[symbol, 'screener'] = data_handler.screener
    df.loc[symbol, recommendation] = summary['RECOMMENDATION']
    sum = summary['BUY'] + summary['SELL'] + summary['NEUTRAL']
    df.loc[symbol, 'buy_' + str_interval] = int(summary['BUY'] / sum * 100)
    df.loc[symbol, 'sell_' + str_interval] = int(summary['SELL'] / sum * 100)
    df.loc[symbol, 'neutral_' + str_interval] = int(summary['NEUTRAL'] / sum * 100)

    return df

def get_tradingview_recommendation(df, interval):
    list_symbol = df['symbolTV'].tolist()
    df = df.set_index('symbolTV', drop=False)

    for symbol in list_symbol:

        screener = df.loc[symbol, 'screener']
        exchange = df.loc[symbol, 'exchange']

        data_handler = TA_Handler(
            symbol=symbol,
            screener=screener,
            exchange=exchange,
            interval=interval,
        )
        try:
            tradingview_summary = data_handler.get_analysis().summary
            df = set_tradingview_data(df, symbol, data_handler, tradingview_summary)
            #print(tradingview_summary)
        except:
            print('no data')

    df.reset_index(inplace=True, drop=True)

    return df

"""
    CSL module: Compute Symbol List
"""
if __name__ == '__main__':
    exchange = ccxt.ftx()
    # exchange = ccxt.binance()

    markets = exchange.load_markets()
    symbols = exchange.symbols
    df_list = {}

    symbols = list(filter(custom_filter, symbols))



    list_crypto_symbols = []
    for symbol in symbols:
        ohlcv = get_ohlcv(symbol, "1h")
        #ohlcv = get_ohlcv(symbol, "1d")
        if ohlcv["volume"].mean() > 10000:
            list_crypto_symbols.append(symbol)
            #df_list[symbol] = ohlcv


    df_symbol = pd.DataFrame(list_crypto_symbols, columns =['symbol'])
    df_symbol['symbolTV'] = df_symbol['symbol'].str.replace("/", "")

    df_symbol['exchange'] = "ftx"
    df_symbol['screener'] = "crypto"

    df_symbol = get_tradingview_recommendation(df_symbol, Interval.INTERVAL_2_HOURS)
    df_symbol = get_tradingview_recommendation(df_symbol, Interval.INTERVAL_4_HOURS)
    df_symbol = get_tradingview_recommendation(df_symbol, Interval.INTERVAL_1_DAY)

    df_symbol.to_csv('screener.csv')

    print("done!")