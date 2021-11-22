import numpy as np
import pandas as pd
# Used to grab the stock prices, with yahoo
import pandas_datareader as web
from datetime import datetime
# To visualize the results
import matplotlib.pyplot as plt
import seaborn
import streamlit as st
import yfinance as yf


def get_price(ticker, start, end=datetime.now()):
    print(ticker)
    df = yf.download(ticker, start=start, end=end).reset_index()
    #df = stock.history(start=start, end=end)
    #Calculate ADJ Close
    #df = calculate_adjusted_close_prices_iterative(df, 'Close').reset_index()
    print(ticker)
    df["ticker"] = ticker
    return df

def create_correlation_table(dataframe):
    df = dataframe.reset_index()
    df = df[['Date', 'Close', 'ticker']]
    df.head()
    df_pivot = df.pivot('Date', 'Symbol', 'Close').reset_index()
    corr_df = df_pivot.corr(method='pearson')
    return corr_df


def account_plot(modelname):
    df = get_model_trade_history(modelname)
    model_one = get_account_value('snp_40_stock_model', rebalance_window, validation_window, unique_trade_date,
                                  df_trade_date)


def display_buy_sell(ticker, trade_mem):
    trade_mem.Date = pd.to_datetime(trade_mem.Date).dt.date

    start = trade_mem.Date.values[0]
    end = trade_mem.Date.values[-1]

    start = datetime.strptime(str(start), '%Y-%m-%d')
    end = datetime.strptime(str(end), '%Y-%m-%d')

    stock = get_price(ticker, start, end)

    stock = stock.set_index('Date')
    stock_trades = trade_mem[trade_mem.ticker == ticker]
    stock_trades['action'] = stock_trades.amount.apply(lambda x: -1 if x < 0 else 0 if x == 0 else 1)
    sells = stock_trades[stock_trades.action <= 0]

    buys = stock_trades[stock_trades.action > 0]

    # ax = fig.add_subplot(111)
    # plotting
    # plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    ax1 = sells.plot(kind='scatter', x='Date', y='price', alpha=0.6
                     , marker='o', color='r')
    ax2 = buys.plot(kind='scatter', x='Date', y='price', alpha=0.6,
                    marker='o', color='b', ax=ax1)
    fig = stock['Adj Close'].plot(rot=70, alpha=0.7, color='grey', figsize=(12, 12))
    plt.xticks(rotation='vertical')
    ax2.set_xticks(ax1.get_xticks()[::12])
    # plt.savefig('pos_neg.png', dpi=200)
    # plt.title(ticker)
    # plt.show()
    print('plotting')
    st.pyplot()
    return True