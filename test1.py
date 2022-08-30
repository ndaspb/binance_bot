#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import copy


# To find a slope of price line
def indSlope(series, n):
    array_sl = [j * 0 for j in range(n - 1)]

    for j in range(n, len(series) + 1):
        y = series[j - n:j]
        x = np.array(range(n))
        x_sc = (x - x.min()) / (x.max() - x.min())
        y_sc = (y - y.min()) / (y.max() - y.min())
        x_sc = sm.add_constant(x_sc)
        model = sm.OLS(y_sc, x_sc)
        results = model.fit()
        array_sl.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(array_sl))))
    return np.array(slope_angle)


# True Range and Average True Range indicator
def indATR(source_DF, n):
    df = source_DF.copy()
    df['H-L'] = abs(df['high'] - df['low'])
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    df_temp = df.drop(['H-L', 'H-PC', 'L-PC'], axis=1)
    return df_temp


# generate data frame with all needed data
def PrepareDF(DF):
    ohlc = DF.iloc[:, [0, 1, 2, 3, 4, 5]]
    ohlc.columns = ["date", "open", "high", "low", "close", "volume"]
    ohlc = ohlc.set_index('date')
    df = indATR(ohlc, 14).reset_index()
    df['slope'] = indSlope(df['close'], 5)
    df['slope2'] = indSlope(df['close'], 100)
    df['channel_max'] = df['high'].rolling(14).max()
    df['channel_min'] = df['low'].rolling(14).min()
    df['position_in_channel'] = (df['close'] - df['channel_min']) / (df['channel_max'] - df['channel_min'])
    df = df.set_index('date')
    df = df.reset_index()
    return (df)


# find local mimimum / local maximum
def isLCC(DF, i):
    df = DF.copy()
    LCC = 0

    if df['close'][i] <= df['close'][i + 1] and df['close'][i] <= df['close'][i - 1] and df['close'][i + 1] > \
            df['close'][i - 1]:
        # найдено Дно
        LCC = i - 1
    return LCC


def isHCC(DF, i):
    df = DF.copy()
    HCC = 0
    if df['close'][i] >= df['close'][i + 1] and df['close'][i] >= df['close'][i - 1] and df['close'][i + 1] < \
            df['close'][i - 1]:
        # найдена вершина
        HCC = i
    return HCC


def getMaxMinChannel(DF, n):
    maxx = 0
    minn = DF['low'].max()
    for i in range(1, n):
        if maxx < DF['high'][len(DF) - i]:
            maxx = DF['high'][len(DF) - i]
        if minn > DF['low'][len(DF) - i]:
            minn = DF['low'][len(DF) - i]
    return (maxx, minn)


apiKey = 'ZM8PKDZDCX817GT6'  # добавьте ключ API

interval_var = '1min'
symbol = 'BNB'

path = 'https://www.alphavantage.co/query?function=CRYPTO_INTRADAY&symbol=' + symbol + '&market=USD&interval=' + interval_var + '&apikey=' + apiKey + '&datatype=csv&outputsize=full'
df = pd.read_csv(path)


# convert time order
df = df[::-1]

prepared_df = PrepareDF(df)

lend = len(prepared_df)

prepared_df['hcc'] = [None] * lend
prepared_df['lcc'] = [None] * lend

for i in range(4, lend - 1):
    if isHCC(prepared_df, i) > 0:
        prepared_df.at[i, 'hcc'] = prepared_df['close'][i]
    if isLCC(prepared_df, i) > 0:
        prepared_df.at[i, 'lcc'] = prepared_df['close'][i]


position = 0
eth_proffit_array = [[0.5, 2], [1, 2], [1.5,2], [2, 2], [2.5, 2]]#, [200, 1], [200, 0]]
# eth_proffit_array = [[20, 2], [40, 2], [60, 2], [80, 2], [100, 2], [100, 0]]#, [200, 1], [200, 0]]
deal = 0
prepared_df['deal_o'] = [None] * lend
prepared_df['deal_c'] = [None] * lend
prepared_df['earn'] = [None] * lend
count = 0
all_sum = 0
for i in range(9, lend - 1):
    prepared_df.at[i, 'earn'] = deal

    if position > 0:
        # add profit/loss for long
        if (prepared_df['close'][i] < stop_price) or ((prepared_df['close'][i] < open_price) and (prepared_df['hcc'][i - 1] != None)):
            # stop loss
            deal = deal + (prepared_df['close'][i] - open_price) * position
            prepared_df.at[i, 'deal_c'] = prepared_df['close'][i]
            all_sum += prepared_df['close'][i]*0.04/100
            position = 0
        else:
            temp_arr = copy.copy(proffit_array)
            for j in range(0, len(temp_arr) - 1):
                delta = temp_arr[j][0]
                contracts = temp_arr[j][1]
                if (prepared_df['close'][i] > (open_price + delta)):
                    # take profit
                    prepared_df.at[i, 'deal_c'] = prepared_df['close'][i]
                    all_sum += prepared_df['close'][i]*0.04/100
                    deal = deal + (prepared_df['close'][i] - open_price) * contracts
                    position = position - contracts
                    del proffit_array[0]

    elif position < 0:
        # add profit/loss for short
        if (prepared_df['close'][i] > stop_price) or ((prepared_df['close'][i] > open_price) and (prepared_df['lcc'][i - 1] != None)):
            # stop loss
            deal = deal + (open_price - prepared_df['close'][i]) * position
            prepared_df.at[i, 'deal_c'] = prepared_df['close'][i]
            all_sum += prepared_df['close'][i]*0.04/100
            position = 0
        else:
            temp_arr = copy.copy(proffit_array)
            for j in range(0, len(temp_arr) - 1):
                delta = temp_arr[j][0]
                contracts = temp_arr[j][1]
                if ((open_price - prepared_df['close'][i]) > delta):
                    # take profit
                    prepared_df.at[i, 'deal_c'] = prepared_df['close'][i]
                    all_sum += prepared_df['close'][i]*0.04/100
                    deal = deal + (open_price - prepared_df['close'][i]) * contracts
                    position = position + contracts
                    del proffit_array[0]

    else:
        # try to find enter point
        if prepared_df['lcc'][i - 1] != None:
            # found bottom - OPEN LONG
            if prepared_df['position_in_channel'][i - 1] < 0.5:
                # close to top of channel
                if prepared_df['slope'][i - 1] < -20:# and (prepared_df['slope2'][i] > 0):
                    # found a good enter point
                    if position == 0:
                        proffit_array = copy.copy(eth_proffit_array)
                        position = 10
                        open_price = prepared_df['close'][i]
                        all_sum += prepared_df['close'][i]*0.02/100
                        count += 1
                        stop_price = prepared_df['close'][i] * 0.99
                        prepared_df.at[i, 'deal_o'] = prepared_df['close'][i]
        if prepared_df['hcc'][i - 1] != None:
            # found top - OPEN SHORT
            if prepared_df['position_in_channel'][i - 1] > 0.5:
                # close to top of channel
                if prepared_df['slope'][i - 1] > 20:# and (prepared_df['slope2'][i] < 0):
                    # found a good enter point
                    if position == 0:
                        proffit_array = copy.copy(eth_proffit_array)
                        position = -10
                        open_price = prepared_df['close'][i]
                        all_sum += prepared_df['close'][i]*0.02/100
                        count += 1
                        stop_price = prepared_df['close'][i] * 1.01
                        prepared_df.at[i, 'deal_o'] = prepared_df['close'][i]

hours = round(lend * 1 / 60, 0)
print(count, all_sum)
print('Total erned in ', hours, ' hours =', int(deal), '$')


# рисовалка

aa = prepared_df[0:999]
aa = aa.reset_index()

# labels = ['close',"deal_o","deal_c"]
labels = ['close', "deal_o", "deal_c", "channel_max", "channel_min"]

labels_line = ['--', "*-", "*-", "g-", "r-"]

j = 0
x = pd.DataFrame()
y = pd.DataFrame()
for i in labels:
    x[j] = aa['index']
    y[j] = aa[i]
    j = j + 1

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

fig.suptitle('Deals')
fig.set_size_inches(15, 7)

for j in range(0, len(labels)):
    ax1.plot(x[j], y[j], labels_line[j])

ax1.set_ylabel('Price')
ax1.grid(True)

ax2.plot(x[0], aa['earn'], 'g-')  # EMA
ax3.plot(x[0], aa['position_in_channel'], '.-')  # EMA

ax2.grid(True)
ax3.grid(True)

plt.show()

# In[ ]:



