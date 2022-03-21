import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.float_format = '{:.6f}'.format
import ccxt
import config
from datetime import datetime
import matplotlib.pyplot as plt
import ta
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import coint
import datetime  # For datetime objects
import os
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

from tradingview_ta import TA_Handler, Interval, Exchange

def get_ohlcv(exchange, symbol, tf):
    df = pd.DataFrame(exchange.fetch_ohlcv(symbol, tf, limit=5000))
    df = df.rename(columns={0: 'timestamp', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'})
    df = df.set_index(df['timestamp'])
    df.index = pd.to_datetime(df.index, unit='ms')
    del df['timestamp']
    return df

def custom_filter(symbol):
    if(
        symbol[-4:] == config.PAIR_USD
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

def set_tradingview_no_data(df, symbol, interval):
    recommendation = "RECOMMENDATION_" + interval
    df.loc[symbol, recommendation] = ""

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
        except:
            df = set_tradingview_no_data(df, symbol, interval)

    df.reset_index(inplace=True, drop=True)

    return df

def get_exchange():
    if config.EXCHANGE == config.EXCHANGE_FTX:
        exchange = ccxt.ftx()
    else:
        exchange = ccxt.binance()

    return exchange

def filter_df_level(df, lst_filter):
    lst_to_clear = config.RECOMMENDATION_ALL
    for item in lst_filter:
        lst_to_clear.remove(item)
    lst_to_clear.append("")

    lst_columns = df.columns.tolist()
    for columns_name in df.columns.tolist():
        if columns_name.startswith("RECOMMENDATION_") == False:
            lst_columns.remove(columns_name)

    for value in lst_to_clear:
        for column in lst_columns:
            df.drop(df.index[df[column] == value], inplace=True)

    df.reset_index(inplace=True, drop=True)
    return df

def get_tradingview_recommendation_list(list_crypto_symbols):
    df_symbol = pd.DataFrame(list_crypto_symbols, columns =['symbol'])
    df_symbol['symbolTV'] = df_symbol['symbol'].str.replace("/", "")
    df_symbol['exchange'] = config.EXCHANGE
    df_symbol['screener'] = config.SCREENER_TYPE

    for interval in config.INTERVAL:
        df_symbol = get_tradingview_recommendation(df_symbol, interval)

    df_symbol.to_csv('screener_all.csv')

    df_symbol = filter_df_level(df_symbol, config.FILTER)

    df_symbol.to_csv('screener_filtered.csv')

    list_crypto_symbols = df_symbol['symbol'].to_list()

    print("tradingview recommendation: ", config.FILTER)
    print("nb symbols:", len(list_crypto_symbols))
    print(list_crypto_symbols)

    return list_crypto_symbols

def get_top_gainer(df):
    df_top_gainer_24h = df.copy()
    df_top_gainer_24h.sort_values(by=['change24h'], ascending=False, inplace=True)
    df_top_gainer_24h = df_top_gainer_24h.set_index('ranking24h', drop=False)
    df_top_gainer_24h = df_top_gainer_24h[:config.PRICE_TOP_GAINER]

    df_top_gainer_1h = df.copy()
    df_top_gainer_1h.sort_values(by=['change1h'], ascending=False, inplace=True)
    df_top_gainer_1h = df_top_gainer_1h.set_index('ranking1h', drop=False)
    df_top_gainer_1h = df_top_gainer_1h[:config.PRICE_TOP_GAINER]

    frame = [df_top_gainer_24h, df_top_gainer_1h]
    df = pd.concat(frame)
    df.sort_values(by=['symbol'], ascending=False, inplace=True)
    df = df[df['symbol'].duplicated() == True]
    df.sort_values(by=['change1h'], ascending=False, inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df_top_gainer_24h, df_top_gainer_1h, df


def get_price_chage(df, markets):
    df = df.set_index('symbol', drop=False)
    lst_symbol = df['symbol'].to_list()

    for symbol in lst_symbol:
        df.loc[symbol, 'change24h'] = float(markets[symbol]['info']['change24h']) * 100
        df.loc[symbol, 'change1h'] = float(markets[symbol]['info']['change1h']) * 100

    df.reset_index(inplace=True, drop=True)

    df.sort_values(by=['change24h'], ascending=False, inplace=True)
    df.reset_index(inplace=True, drop=True)
    df['ranking24h'] = df.index

    df.sort_values(by=['change1h'], ascending=False, inplace=True)
    df.reset_index(inplace=True, drop=True)
    df['ranking1h'] = df.index

    df.reset_index(inplace=True, drop=True)

    df.to_csv('screener_all_price.csv')

    df_top_gainer24h, df_top_gainer1h, df_top_gainer_crossover = get_top_gainer(df)

    df_top_gainer24h.to_csv('screener_top_gainer24h.csv')
    df_top_gainer1h.to_csv('screener_top_gainer1h.csv')
    df_top_gainer_crossover.to_csv('screener_top_gainer_crossover.csv')

    return df_top_gainer_crossover


def get_market_price_changes(list_crypto_symbols, markets):
    df_symbol = pd.DataFrame(list_crypto_symbols, columns=['symbol'])
    df_symbol = get_price_chage(df_symbol, markets)

    list_top_gainer = df_symbol['symbol'].to_list()

    print("price top gainers: ")
    print("nb symbols:", len(list_top_gainer))
    print(list_top_gainer)

    return list_top_gainer

def filter_symbol_by_volume(symbols):
    list_crypto_symbols = []
    if config.VOLUME_FILTERED_BY_INFO:
        for symbol in symbols:
            info = markets[symbol]['info']
            volume_info = float(info['quoteVolume24h'])
            if volume_info > config.VOLUME_THRESHOLD:
                list_crypto_symbols.append(symbol)
    else:
        for symbol in symbols:
            ohlcv = get_ohlcv(exchange, symbol, config.TV_INTERVAL_1_DAY)
            volume_mean = float(ohlcv["volume"].mean())
            if ohlcv["volume"].mean() > config.VOLUME_THRESHOLD:
                list_crypto_symbols.append(symbol)

    return list_crypto_symbols

def get_price_and_tradingview_common(lst1, lst2):
    lst = []
    for item in lst1:
        if item in lst2:
            lst.append(item)
    return lst


def collect_ohlcv_crypto(exchange, symbols, interval):
    for symbol in symbols:
        ohlcv = get_ohlcv(exchange, symbol, interval)
        fileneame = './CRYPTO_DATAS/' + symbol.replace("/", "-") +'.csv'
        ohlcv.to_csv(fileneame)
        print(symbol, ' -> ', fileneame)



def set_tradingview_recommendation(list_crypto_symbols, interval):
    return

"""
    CSL module: Compute Symbol List
"""
if __name__ == '__main__':
    from datetime import datetime

    start_time = datetime.now()

    exchange = get_exchange()
    markets = exchange.load_markets()
    symbols = exchange.symbols

    print("list symbols available: ", len(symbols))
    symbols_total_size = len(symbols)

    symbols = list(filter(custom_filter, symbols))
    print("symbol filtered: ", symbols_total_size - len(symbols))

    list_crypto_symbols = filter_symbol_by_volume(symbols)

    print("low volume symbol dropped: ", len(symbols) - len(list_crypto_symbols))
    print("symbol remaining: ", len(list_crypto_symbols))


    collect_ohlcv_crypto(exchange, list_crypto_symbols, config.TV_INTERVAL_1_DAY)
    set_tradingview_recommendation(list_crypto_symbols, config.TV_INTERVAL_1_DAY)

    list_price = get_market_price_changes(list_crypto_symbols, markets)

    list_tradingview = get_tradingview_recommendation_list(list_crypto_symbols)

    list_reinforced = get_price_and_tradingview_common(list_price, list_tradingview)
    print("common symbol: ", len(list_reinforced))
    print(list_reinforced)

    end_time = datetime.now()
    duration_time = end_time - start_time
    print('duration: ', duration_time)





