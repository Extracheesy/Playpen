#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/CryptoRobotFr/pair_trading/blob/main/pair_selection.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


#get_ipython().system('pip install ccxt statsmodels ta')


# In[ ]:


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


# In[ ]:


def get_ohlcv(symbol, tf):
    df = pd.DataFrame(exchange.fetch_ohlcv(symbol, tf, limit=5000))
    df = df.rename(columns={0: 'timestamp', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'})
    df = df.set_index(df['timestamp'])
    df.index = pd.to_datetime(df.index, unit='ms')
    del df['timestamp']
    return df

exchange = ccxt.ftx()
markets = exchange.load_markets()
symbols = exchange.symbols
df_list = {}

def custom_filter(symbol):
    if(
        symbol[-4:] == "/USD"
        and "BULL" not in symbol
        and "HALF" not in symbol
        and "EDGE" not in symbol
        and "BEAR" not in symbol
    ):
	    return True

symbols = list(filter(custom_filter, symbols))

df_symbol = pd.DataFrame(symbols, columns =['Symbols'])

for symbol in symbols:
    ohlcv = get_ohlcv(symbol, "1h")
    if ohlcv["volume"].mean() > 10000:
        df_list[symbol] = ohlcv


# In[ ]:


df_list_origin = df_list.copy()
for symbol in df_list_origin:
    if len(df_list[symbol]) < 5000:
        del df_list[symbol]


# In[ ]:


full_df = pd.DataFrame()
for symbol in df_list:
    full_df[symbol] = df_list[symbol]['close']

df_symbol = pd.DataFrame(full_df.columns, columns=['Symbols'])

cumret = np.log(full_df).diff().cumsum()+1 # calculate cumulative returns
cumret.dropna(inplace=True) 
cumret


# In[ ]:


pre_select_obj = {}
for col in list(cumret.columns.values):
    pre_select_obj[col] = {
        "return": cumret[col].iloc[-1] - cumret[col].iloc[0],
        # "return": cumret[col].diff().mean(),
        "std": cumret[col].std()
    }


# In[ ]:


df_pre_select = pd.DataFrame.from_dict(pre_select_obj, orient='index')
df_pre_select.plot.scatter(x='return', y='std')


# In[ ]:


df_pre_select = pd.DataFrame.from_dict(pre_select_obj, orient='index')
df_pre_select.plot.scatter(x='return', y='std')
  
kmeans = KMeans(n_clusters=5).fit(df_pre_select)
centroids = kmeans.cluster_centers_
# print(centroids)

cluster_map = pd.DataFrame()
cluster_map['data_index'] = df_pre_select.index.values
cluster_map['cluster'] = kmeans.labels_

plt.scatter(df_pre_select['return'], df_pre_select['std'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()


# In[ ]:


train_cumret = cumret.copy().iloc[:3500] # formation period
test_cumret = cumret.copy().iloc[-1500:] # trading period

get_sub_cluster = 1

df_train = train_cumret.copy()[list(cluster_map[cluster_map.cluster == get_sub_cluster]["data_index"])]
df_test = test_cumret.copy()[list(cluster_map[cluster_map.cluster == get_sub_cluster]["data_index"])]
df_train.columns.values


# In[ ]:


train = df_train.copy()
tested = []
metrics_obj = {}

for s1 in train.columns:
    for s2 in train.columns:
        if s1!=s2 and (f'{s1}-{s2}' not in tested) and (f'{s2}-{s1}' not in tested):
                tested.append(f'{s1}-{s2}')
                cadf_p = coint(train[s1], train[s2])[1]
                spread_window = 25
                spread = np.log10(train[s1] / train[s1].shift(periods=spread_window)) - np.log10(train[s2] / train[s2].shift(periods=spread_window))
                spread = spread.fillna(0)
                std_spread = spread.std()
                correlation = train[s1].corr(train[s2])
                metrics_obj[f'{s1}-{s2}'] = {
                    "co-int": cadf_p,
                    "std_spread": std_spread,
                    "correlation": correlation
                }


# In[ ]:


pd.options.display.float_format = '{:.6f}'.format
df_metrics = pd.DataFrame.from_dict(metrics_obj, orient='index')
df_metrics["rating"] = (1 - df_metrics["co-int"]) + df_metrics["correlation"] + 10*df_metrics["std_spread"]
df_metrics.sort_values(by="rating", ascending=False).iloc[:30]


# In[ ]:


def merge_into_pair_df(df1, df2):
    _df1 = df1.copy()
    _df2 = df2.copy()
    _df1 = _df1.rename(columns={"open": "open_1", "high": "high_1", "low": "low_1", "close": "close_1", "volume": "volume_1"})
    _df2 = _df2.rename(columns={"open": "open_2", "high": "high_2", "low": "low_2", "close": "close_2", "volume": "volume_2"})
    df = pd.concat([_df1, _df2], axis=1)
    return df

def get_pair_informations(s1, s2, tf="1h", spread_window=25):
    df1 = get_ohlcv(s1, tf)
    df2 = get_ohlcv(s2, tf)
    df = merge_into_pair_df(df1, df2)
    
    df['rol_1'] = df['close_1'].shift(periods=spread_window)
    df['rol_2'] = df['close_2'].shift(periods=spread_window)

    df['spread'] = np.log10(df['close_1'] / df['rol_1']) - np.log10(df['close_2'] / df['rol_2'])

    fig, ax_left = plt.subplots(figsize=(20, 15), nrows=2, ncols=1)
    ax_right = ax_left[0].twinx()

    c1 = df['close_1'].copy().loc['2021':]
    c2 = df['close_2'].copy().loc['2021':]
    spread = df['spread'].copy().loc['2021':]

    ax_left[0].plot(c1, color='blue', label=s1)
    ax_right.plot(c2, color='orange', label=s2)
    ax_right.legend()
    ax_left[0].legend(loc=2)

    spread_std = df['spread'].std()
    ax_left[1].plot(spread, color='black')
    ax_left[1].axhline(2*spread_std, color='green')
    ax_left[1].axhline(-2*spread_std, color='red')
    ax_left[1].axhline(0, color='orange')
    
    print("Correlation =",round(c1.corr(c2)*100,2),"%")
    print("Co-integration =", round(coint(c1, c2)[1],5))
    


# In[ ]:


get_pair_informations("CRO/USD", "SAND/USD")


# In[ ]:


get_pair_informations("AAVE/USD", "UNI/USD")


# In[ ]:


get_pair_informations("TSLA/USD", "CRV/USD")


# In[ ]:


get_pair_informations("FIDA/USD", "SECO/USD")


# In[ ]:


get_pair_informations("DAI/USD", "ETH/USD")


# In[ ]:


get_pair_informations("BTC/USD", "WBTC/USD")

