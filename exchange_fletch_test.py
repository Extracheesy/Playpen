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



def get_ohlcv(symbol, tf):
    df = pd.DataFrame(exchange.fetch_ohlcv(symbol, tf, limit=5000))
    df = df.rename(columns={0: 'timestamp', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'})
    df = df.set_index(df['timestamp'])
    df.index = pd.to_datetime(df.index, unit='ms')
    del df['timestamp']
    return df


# exchange = ccxt.ftx()
exchange = ccxt.binance()
# exchange = ccxt.hitbtc()
markets = exchange.load_markets()
symbols = exchange.symbols
df_list = {}

def custom_filter(symbol):
    if(
        #symbol[-4:] == "/USD"
        symbol[-4:] == "/EURS"
        and "BULL" not in symbol
        and "HALF" not in symbol
        and "EDGE" not in symbol
        and "BEAR" not in symbol
    ):
	    return True


symbols = list(filter(custom_filter, symbols))

symbols = ['BTC/EUR']

df_symbol = pd.DataFrame(symbols, columns =['Symbols'])


for symbol in symbols:
    #ohlcv = get_ohlcv(symbol, "1h")
    ohlcv = get_ohlcv(symbol, "1d")
    if ohlcv["volume"].mean() > 10000:
        df_list[symbol] = ohlcv

ohlcv.to_csv('ohlcc.csv')

ohlcv.sort_index(ascending=False, inplace=True)

ohlcv.to_csv('ohlcc_sorted.csv')

print("done!")