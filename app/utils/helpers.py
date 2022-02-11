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

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go


def build_history_df(trade_df):
    start = trade_df.index.values[0]
    end = trade_df.index.values[-1]

    history_dfs_by_asset = {}

    for asset in np.append(trade_df.name.unique(), ['SPY']):
        asset_values = get_price(asset, start, end)
        asset_values['name'] = asset
        history_dfs_by_asset[asset] = asset_values

    return history_dfs_by_asset

def get_price_from_df(ticker,date,history_dfs_by_asset):
    i = history_dfs_by_asset[ticker].index.get_loc(date)
    price_as_at_date = history_dfs_by_asset[ticker]['Close'][i-1]
    return price_as_at_date

#Split trade memories by asset.
def split_trade_df(trade_df):
    '''Splits a dataframe of trades into a dictionary of dfs, by asset.'''
    trade_df_grouped = trade_df.groupby('name')
    temp = [trade_df_grouped.get_group(x).drop_duplicates().sort_index() for x in trade_df_grouped.groups]
    #Consolidate trades made in same day. Sum of amounts, divided by value to calculate avg price.
    trade_dfs_by_asset = {}
    for df1 in temp:
        df = df1.groupby("Date").sum()
        df["price"] = df["value"] / df["amount"]
        df.insert(loc = 0, column = 'name', value = df1["name"][0])
        trade_dfs_by_asset[df1['name'][0]] = df
    return trade_dfs_by_asset


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


def display_holdings(trade_df, history_dfs_by_asset, start, end):
    unique_days = trade_df[start:end].index.unique()
    trade_dfs_by_asset = split_trade_df(trade_df)

    daily_breakdown = pd.DataFrame()
    daily_value = []

    for day in unique_days:
        value, breakdown = get_portfolio_value_breakdown(trade_dfs_by_asset, day, history_dfs_by_asset)
        day_df = pd.DataFrame.from_dict(breakdown)
        day_df.index = [day]
        daily_breakdown = daily_breakdown.append(day_df)
        daily_value.append(value)

    data = []
    for ticker in daily_breakdown.columns:
        data.append(go.Bar(name=ticker, x=daily_breakdown.index, y=daily_breakdown[ticker]))

    fig = go.Figure(data)

    fig.add_trace(go.Scatter(
        x=daily_breakdown.index,
        y=[1.05] * len(daily_breakdown.index),
        mode="markers+text",
        name="Portfolio Value (M)",
        text=[round(i / 1000000, 1) for i in daily_value],
        textposition="top center"
    ))
    # Change the bar mode
    fig.update_layout(barmode='stack')
    fig.show()

    return daily_breakdown


def build_portfolio_value_df(trade_df, history_dfs_by_asset):
    start = trade_df.index.values[0]
    end = trade_df.index.values[-1]

    unique_days = trade_df.index.unique()
    unique_assets = trade_df.name.unique()

    df = pd.DataFrame()

    # starting conditions
    portfolio = {}
    portfolio['cash'] = 1000000
    for asset in unique_assets:
        portfolio[asset] = 0

    daily_portfolio_values = []
    daily_bh_values = []
    daily_sp_values = []

    spy_first_day = get_price_from_df('SPY', start, history_dfs_by_asset)

    for day in unique_days:
        daily_trade_df = trade_df[day:day]
        daily_cash_change = 0
        daily_portfolio_value = 0
        daily_bh_value = 0

        for index, row in daily_trade_df.iterrows():
            portfolio[row['name']] += row['amount']
            daily_cash_change += row['value']

        portfolio['cash'] -= daily_cash_change

        for asset in portfolio:
            if asset != 'cash':
                if asset in daily_trade_df['name']:
                    daily_asset_value = daily_trade_df[daily_trade_df['name'] == asset]['value']
                else:
                    try:
                        daily_asset_value = get_price_from_df(asset, day, history_dfs_by_asset)
                    except KeyError:
                        daily_asset_value = get_price(asset, day)['Close'][0]

                daily_portfolio_value += daily_asset_value * portfolio[asset]
                # bh assumes 462 shares per asset. (starting value = 1M)
                daily_bh_value += daily_asset_value * 462

        daily_portfolio_value += portfolio['cash']

        daily_portfolio_values.append(daily_portfolio_value)
        daily_bh_values.append(daily_bh_value)

        try:
            spy_value = get_price_from_df('SPY', day, history_dfs_by_asset)
        except KeyError:
            spy_value = get_price('SPY', day)['Close'][0]

        daily_sp_values.append(spy_value * 1000000 / spy_first_day)

        print(str(day) + 'completed', end='\r')

    df['portfolio'] = daily_portfolio_values
    df['bh'] = daily_bh_values
    df['sp'] = daily_sp_values
    df.index = unique_days

    return df