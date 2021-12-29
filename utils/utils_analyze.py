import pandas as pd
import numpy as np
from datetime import datetime
import glob
import yfinance as yf
import re

def get_account_value(model_name, rebalance_window, validation_window, unique_trade_date, df_trade_date):
    df_account_value=pd.DataFrame()
    for i in range(rebalance_window+validation_window, len(unique_trade_date)+1,rebalance_window):
        try:
            temp = pd.read_csv('results/account_value_trade_{}_{}.csv'.format(model_name,i))
            df_account_value = df_account_value.append(temp,ignore_index=True)
        except: break
    df_account_value = pd.DataFrame({'account_value':df_account_value['account_value']})
    sharpe=(252**0.5)*df_account_value.account_value.pct_change(1).mean()/df_account_value.account_value.pct_change(1).std()
    print(sharpe)
    df_account_value=df_account_value.join(df_trade_date[63:].reset_index(drop=True))
    return df_account_value


def get_model_trade_history(model_name):
    listdir = glob.glob('results/trade_history_{}_*.csv'.format(model_name))
    list_of_nums = [int(re.findall(r'\d+', s)[1]) for s in listdir]
    list_of_nums.sort()
    print(list_of_nums)
    trade_history=pd.DataFrame()
    for i in list_of_nums:
        temp = pd.read_csv('results/trade_history_{}_{}_backtest.csv'.format(model_name,i),index_col=0)
        trade_history = trade_history.append(temp,ignore_index=True)

    return trade_history


def calculate_holdings(df):
    """
    Checks how much holding you have on particualr stock on each given day
    :param df:
    :return:
    """
    df = df.reset_index().drop('index', axis=1)
    df['holdings'] = 0
    for i in range(len(df)):
        df.iloc[i, 4] = df.iloc[:i, 2].sum()
    return df

def get_price(ticker, start, end=datetime.now()):
    df = yf.download(ticker, start=start, end=end).reset_index()
    #df = stock.history(start=start, end=end)
    #Calculate ADJ Close
    #df = calculate_adjusted_close_prices_iterative(df, 'Close').reset_index()

    df["ticker"] = ticker
    return df

def check_index_dim(df_non_zero,index_dim, indexes):
    prev_date = ''
    df_non_zero.index = df_non_zero.Date.factorize()[0]
    for i in df_non_zero.index.unique():
        if len(df_non_zero.loc[i, ['index_close', 'index_macd', 'index_rsi']].dropna()) != index_dim:
            temp1 = list(df_non_zero.loc[i, ['index_close', 'index_macd', 'index_rsi','index_ticker']].dropna().index_ticker.unique())
            missing_list = list(set(indexes) - set(temp1))
            date = df_non_zero.loc[i,'Date'].iloc[0]

            print(date)
            for ind in missing_list:
                to_append = {'index_close':0, 'index_macd':0 ,'index_rsi':0, 'index_ticker':ind,'Date':date}
                df_non_zero = df_non_zero.append(to_append, ignore_index=True).sort_values(['Date','ticker'])
                df_non_zero.index = df_non_zero.Date.factorize()[0]



            """ 
               index_close = df_non_zero[df_non_zero.Date == prev_date].index_close.iloc[-1]
            index_macd = df_non_zero[df_non_zero.Date == prev_date].index_macd.iloc[-1]
            index_rsi = df_non_zero[df_non_zero.Date == prev_date].index_rsi.iloc[-1]

               df_non_zero = df_non_zero.append({'Date': date, 'index_close': index_close, 'index_macd':index_macd, 'index_rsi':index_rsi}, ignore_index=True)
            """


    return df_non_zero

def backtest_strat(df):
    strategy_ret= df.copy()
    strategy_ret['Date'] = pd.to_datetime(strategy_ret['Date'])
    strategy_ret.set_index('Date', drop = False, inplace = True)
    strategy_ret.index = strategy_ret.index.tz_localize('UTC')
    del strategy_ret['Date']
    ts = pd.Series(strategy_ret['daily_return'].values, index=strategy_ret.index)
    return ts

def get_buy_and_hold(path, start_test, end_test):
    """

    :param path:
    :return:
    """
    snp_t = pd.read_csv(path)
    sn = pd.DataFrame(columns=['Date', 'daily_return'])
    for t in snp_t.ticker.unique():
        df = snp_t[snp_t.ticker == t]

        df['daily_return'] = df['adjcp'].pct_change(1)

        sn = sn.append(df)
    snpthirty = pd.DataFrame(columns=['Date', 'daily_return'])
    for d in sn.Date.unique():
        mean_of_adj = sum(sn[sn.Date == d]['daily_return']) / len(sn[sn.Date == d]['daily_return'])
        df2 = pd.DataFrame([[d, mean_of_adj]], columns=['Date', 'daily_return'])

        snpthirty = snpthirty.append(df2, ignore_index=True)

    snpthirty['Date'] = pd.to_datetime(snpthirty['Date'])
    snpthirty = snpthirty[(snpthirty['Date'] > start_test) & (snpthirty['Date'] < end_test)]
    return snpthirty