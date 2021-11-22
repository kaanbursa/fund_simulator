from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from data.preprocessing import preprocess_for_train
from data.Clustering import UnsupervisedLearning


def add_indicators_for_stocks( name, ticker_list, index_list, indicators_stock_stats, start_date, end_date) -> pd.DataFrame:
    """
    Preprocessing needed for the stock before modeling the trade
    returns dataset for stocks and index
    """
    load_from_date = start_date - timedelta(days=60)
    sn, index_df = preprocess_for_train(name, ticker_list=ticker_list,
                                        indicator_list=indicators_stock_stats,
                                        start_date=load_from_date,
                                        turbulance=True,
                                        indexes=index_list,
                                        end_date=end_date)
    sn['Date'] = pd.to_datetime(sn['Date'])
    df_non_zero = sn.sort_values(['Date', 'ticker'])

    dataset = df_non_zero[df_non_zero.Date >= start_date]

    return dataset, index_df


def get_clustered_stocks(df, from_date, to_date, number_of_clusters = 5):

    # Checkout if the stock is listed on stock ex at that date range

    lst1 = df[(df.Date >= pd.to_datetime(from_date)) & (
            df.Date < pd.to_datetime(from_date + timedelta(days=4)))].ticker.unique()
    lst2 = df[(df.Date >= pd.to_datetime(to_date)) & (
            df.Date < pd.to_datetime(to_date + timedelta(days=4)))].ticker.unique()
    existing_stocks = [value for value in lst1 if value in lst2]
    training_df = df[df['ticker'].isin(existing_stocks)]
    # filter nan and inf values replace with 0

    training_df = training_df.replace([np.inf, -np.inf], 0).fillna(0)


    training_df = training_df[(training_df.Date >= pd.to_datetime(from_date)) & (training_df.Date < pd.to_datetime(to_date))]
    # For clustering
    training_df.Date = training_df.Date.apply(lambda x: x.strftime('%Y-%m-%d'))
    UL = UnsupervisedLearning(training_df)

    cluster = UL.kmeans_cluster_stocks(number_of_clusters)

    return cluster, training_df

def pick_from_kmeans_cluster(df_clst, full_stock_df, qtile = 0.75, n_from_each=5):
    """
    Grabs stock volumes from fullstock, applies a 0.9 quantile mean dail vol ceiling on stock we clıstered, picks most decorrelated stock from each cluster
    :param df_clst:
    :param full_stock_df:
    :param qtile:
    :param n_from_each:
    :return:
    """
    volumes = pd.DataFrame()
    for stock in full_stock_df.ticker.unique():

        volumes[stock] = [full_stock_df[full_stock_df.ticker == stock]['Volume'].mean()]
    volumes = volumes.T

    volumes.rename(columns={0: 'mean_vol'}, inplace=True)
    clst_stocks = df_clst.merge(volumes, left_index=True, right_index=True)
    above_vol_ceiling = clst_stocks[clst_stocks['mean_vol'] > clst_stocks['mean_vol'].quantile(qtile)]

    each_from_cluster = []
    for i in range(df_clst['label'].max() + 1):
        each_from_cluster.append(list(
            above_vol_ceiling[above_vol_ceiling['label'] == i].sort_values(by='distance', ascending=False).head(n_from_each).index
        ))

    flat_list = [item for sublist in each_from_cluster for item in sublist]

    return flat_list
def order_stocks_by_volume(clusters, dataframe, stocks_per_cluster):
    stocks  = []
    number_of_cluster = len(clusters)
    stock_needed = number_of_cluster * stocks_per_cluster

    for cluster in clusters:
        volumes = {}
        for ticker in cluster:

            stock = dataframe[dataframe.ticker == ticker]
            if len(stock[stock.Close == 0]) > 0:
                continue
            volumes[ticker] = stock['Volume'].mean()
        cluster_stocks = sorted(volumes.items(), key=lambda x:x[1], reverse=True)

        stocks.extend([k[0] for k in cluster_stocks][:stocks_per_cluster])
    clusters.sort(key=len, )
    if len(stocks) != stock_needed:
        print('Stocks does not match adding new stocks')
        for cluster in clusters:
            for stock in cluster: # take the longest clustter
                    if len(stocks) == stock_needed:
                        break
                    if stock not in stocks:
                        stocks.append(stock)
            if len(stocks) == stock_needed:
                break


    return stocks