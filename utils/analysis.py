import numpy as np
import pandas as pd


def buy_and_hold(df, initial_holding=100000):
    start_date = df["Date"][0]
    end_date = df["Date"][0]
    return True

def plot_portfolio_allocation(df):
    """
    return pie chart of portfolio allocation
    :param df:
    :return:
    """
    pass


def calculate_sharpe(df):
    df_total_value = df.copy()
    df_total_value.columns = ["account_value_train"]

    df_total_value["daily_return"] = df_total_value.pct_change(1)
    sharpe = (
        (4 ** 0.5)
        * df_total_value["daily_return"].mean()
        / (df_total_value["daily_return"].std() + 0.00001)
    )
    return sharpe


def calculate_var(df):
    raise NotImplementedError
