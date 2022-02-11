import numpy as np
import pandas as pd
import yfinance as yf
import os
from finta import TA
from stockstats import StockDataFrame as Sdf
from config.config import ALL_STOCKS_DATA_FILE
from datetime import date, timedelta, datetime
import ccxt
import csv
from pathlib import Path
import enum

class DataProcessor():
    def __init__(self, type:str, tickers:list, start:str, end:datetime):
        self.tickers = tickers
        self.type = type

    def data_split(self, df, start, end):
        """
        split the dataset into training or testing using date
        :param data: (df) pandas dataframe, start, end
        :return: (df) pandas dataframe
        """

        data = df[(df.Date >= start) & (df.Date < end)]

        data = data.sort_values(["Date", "ticker"], ignore_index=True)
        # data  = data[final_columns]
        data.index = data.Date.factorize()[0]
        return data

    def get_price(self, ticker, start, end=datetime.now()):
        if self.type == 'Stocks':

            df = yf.download(ticker, start=start, end=end).reset_index()
            # df = stock.history(start=start, end=end)
            # Calculate ADJ Close
            # df = calculate_adjusted_close_prices_iterative(df, 'Close').reset_index()

            df["ticker"] = ticker
            return df
        else:
            # TODO: Import crypto
            raise NotImplementedError

    def add_turbulance(self, df):
        """
        add turbulence index from a precalcualted dataframe based on mahanolobis distance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = df.sort_values(["Date", "ticker"])
        turbulence_index = self.calculate_turbulance(df)
        df = df.merge(turbulence_index, on="Date")
        df = df.sort_values(["Date", "ticker"]).reset_index(drop=True)
        return df

    def add_categories(self, df):

        if os.path.exists('./' + ALL_STOCKS_DATA_FILE):

            sto = pd.read_csv(ALL_STOCKS_DATA_FILE)
        else:
            sto = pd.read_csv(f'../{ALL_STOCKS_DATA_FILE}')
        dictionary = {k: v for k, v in zip(sto['Symbol'], sto['Sector'])}

        df["category"] = df["ticker"].apply(lambda x: dictionary[x.upper()])
        df["category"] = df["category"].astype("category").cat.codes
        return df

    def calculate_turbulance(self, df):
        """calculate turbulence index based on dow 30"""
        if 'Adj Close' in df.columns:
            close_name = 'Adj Close'
        else:
            close_name = 'adjcp'

        df_price_pivot = df.pivot(index="Date", columns="ticker", values=close_name)

        unique_date = df.Date.unique()
        # start after a year
        start = 252
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0

        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]

            hist_price = df_price_pivot[
                [n in pd.to_datetime(unique_date[0:i]) for n in df_price_pivot.index]
            ]
            # Clear stocks from interviening calculation
            current_price = current_price.loc[:, (current_price != 0.0).any(axis=0)]
            hist_price = hist_price.loc[:, (hist_price != 0.0).any(axis=0)]

            cov_temp = hist_price.cov()

            current_temp = current_price - np.mean(hist_price, axis=0)

            temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"Date": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index

    def _add_stock_info_for_missing_days(self, df):
        sn_n = df.copy()
        for date in df.Date.unique():
            for tick in df.ticker.unique():
                if tick not in df[df.Date == date].ticker.unique():
                    d = {"ticker": tick, "Date": date}
                    sn_n = sn_n.append(d, ignore_index=True)
        return sn_n

    def add_technical_indicator(self, data, indicator_list):
        """
        calcualte technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["ticker", "Date"])
        stock = Sdf.retype(df.copy())
        stock["close"] = stock["adj close"]
        unique_ticker = stock.ticker.unique()

        for indicator in indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):

                try:
                    temp_indicator = stock[stock.ticker == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["ticker"] = unique_ticker[i]
                    temp_indicator["Date"] = df[df.ticker == unique_ticker[i]][
                        "Date"
                    ].to_list()
                    indicator_df = indicator_df.append(
                        temp_indicator, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(
                indicator_df[["ticker", "Date", indicator]], on=["ticker", "Date"], how="left"
            )

        df = df.sort_values(by=["Date", "ticker"])
        return df

    def preprocess_for_train(self,
            name, turbulance, indicator_list, start_date, end_date=datetime.now(), save=True
    ):
        """
        The order is important for replicability
        :param name: (str) name of the dataset for saving
        :return: df(pd.Dataframe) of stock prices with indicators and index_df (pd.Dataframe) on sepearte dataframe
        """

        print("1. Getting prices of stocks and adding technical indicators")

        path = "./datasets/" + name + ".csv"
        if os.path.exists(path):
            print('loading', path)
            sn_n = pd.read_csv(path)
            if turbulance and 'turbulence' not in sn_n.columns:
                print('Calculating turbulance')

                sn_n = sn_n.reset_index().sort_values(by=["ticker", "Date"])
                sn_n = add_turbulance(sn_n)

        else:
            df = self.get_price(self.tickers[0], start_date, end_date)
            # df = add_technical_indicator(df.rename(columns={"adjcp": "Adj Close"}), indicator_list)
            for tick in self.tickers[1:]:
                stock = self.get_price(tick, start_date, end_date)
                # stock = add_technical_indicator(
                #    stock.sort_values(by=["ticker", "Date"])
                #    .reset_index(drop=True)
                #    .rename(columns={"adjcp": "Adj Close"}),
                #    indicator_list,
                # )

                df = pd.concat([df, stock])
            df = df[(df.Date >= pd.to_datetime(start_date)) & (df.Date <= pd.to_datetime(end_date))]
            df = self.add_technical_indicator(df.rename(columns={"adjcp": "Adj Close"}).sort_values(by=["ticker", "Date"])
                                         , indicator_list)
            print("2. Saving processed csv as stocks " + name + ".csv")
            print("Total tickers: ", len(df.ticker.unique()))
            print("3. Adding company categories")
            try:
                df = self.add_categories(df)
            except:
                print('Category problem')
            df = df.fillna(0)
            print("4. Adding stock information for empty stock days")

            sn_n = add_stock_info_for_missing_days(df)
            sn_n.to_csv("./datasets/" + name + ".csv", index=False)

            if turbulance:
                print("5. Calculating turbulance")
                # Ticker first for turbulance calculation
                sn_n = sn_n.reset_index().sort_values(by=["ticker", "Date"])

                sn_n = self.add_turbulance(sn_n)
                # print("Total tickers: ", len(sn_n.ticker.unique()))
            else:
                sn_n['turbulence'] = 0
        # else:
        #    sn_n['turbulence'] = 0
        #    sn_n = sn_n.reset_index(drop=True)
        sn_n = sn_n.fillna(0)

        print("6. Adding indexes")

        # index_df = add_indices_and_ema(sn_n.sort_values("Date"), indexes)
        index_df = pd.DataFrame()
        print("Total tickers: ", len(sn_n.ticker.unique()))
        """
        print("7. Check if the dates are the same in two dataframes")
        for date in sn_n.Date.unique():
            if date not in index_df.Date.unique():
                sn_n = sn_n[sn_n.Date != date]
        for date in index_df.Date.unique():
            if date not in sn_n.Date.unique():
                index_df = index_df[index_df.Date != date]
        """
        sn_n.rename(columns={"Volume": "volume", "Adj Close": "adjcp"}, inplace=True)
        sn_n.Date = pd.to_datetime(sn_n.Date)
        sn_n['month'] = sn_n.Date.apply(lambda x: x.month)
        sn_n['day'] = sn_n.Date.apply(lambda x: x.day)
        print("8. Saving last csv as stocks " + name + ".csv")
        sn_n = sn_n.replace([np.inf, -np.inf], 0).sort_values(['Date', 'ticker'])

        if save:
            sn_n.to_csv("./datasets/" + name + ".csv", index=False)

        return sn_n, index_df

    def _retry_fetch_ohlcv(self, exchange, max_retries, symbol, timeframe, since, limit):
        num_retries = 0
        try:
            num_retries += 1
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            # print('Fetched', len(ohlcv), symbol, 'candles from', exchange.iso8601 (ohlcv[0][0]), 'to', exchange.iso8601 (ohlcv[-1][0]))
            return ohlcv
        except Exception:
            if num_retries > max_retries:
                raise  # Exception('Failed to fetch', timeframe, symbol, 'OHLCV in', max_retries, 'attempts')

    def _scrape_ohlcv(self, exchange, max_retries, symbol, timeframe, since, limit):
        earliest_timestamp = exchange.milliseconds()
        timeframe_duration_in_seconds = exchange.parse_timeframe(timeframe)
        timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
        timedelta = limit * timeframe_duration_in_ms
        all_ohlcv = []
        while True:
            fetch_since = earliest_timestamp - timedelta
            ohlcv = self._retry_fetch_ohlcv(
                exchange, max_retries, symbol, timeframe, fetch_since, limit
            )
            # if we have reached the beginning of history
            if ohlcv[0][0] >= earliest_timestamp:
                break
            earliest_timestamp = ohlcv[0][0]
            all_ohlcv = ohlcv + all_ohlcv
            print(
                len(all_ohlcv),
                symbol,
                "candles in total from",
                exchange.iso8601(all_ohlcv[0][0]),
                "to",
                exchange.iso8601(all_ohlcv[-1][0]),
            )
            # if we have reached the checkpoint
            if fetch_since < since:
                break
        return all_ohlcv

    def _write_to_csv(self, filename, exchange, data):
        p = Path("./data/raw/", str(exchange))
        p.mkdir(parents=True, exist_ok=True)
        full_path = p / str(filename)
        with Path(full_path).open("w+", newline="") as output_file:
            csv_writer = csv.writer(
                output_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerows(data)
        df = pd.read_csv(full_path, names=['Date','open','high','low','close','volume'])
        df.Date = df.Date.apply(lambda x: datetime.fromtimestamp(x / 1000))
        df.to_csv(full_path)
        return df

    def get_crypto_price(self, filename, exchange_id, max_retries, symbol, timeframe, since, limit):
        # instantiate the exchange by id
        exchange = getattr(ccxt, exchange_id)(
            {
                "enableRateLimit": True,  # required by the Manual
            }
        )
        # convert since from string to milliseconds integer if needed
        if isinstance(since, str):
            since = exchange.parse8601(since)
        # preload all markets from the exchange
        exchange.load_markets()
        # fetch all candles
        ohlcv = self._scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit)
        # save them to csv file
        df = self._write_to_csv(filename, exchange, ohlcv)
        print(
            "Saved",
            len(ohlcv),
            "candles from",
            exchange.iso8601(ohlcv[0][0]),
            "to",
            exchange.iso8601(ohlcv[-1][0]),
            "to",
            filename,
        )
        return df



def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """

    data = df[(df.Date >= start) & (df.Date < end)]

    data = data.sort_values(["Date", "ticker"], ignore_index=True)
    # data  = data[final_columns]
    data.index = data.Date.factorize()[0]
    return data

def daterange(start_date, end_date,date_jump=365):
    for n in range(int((end_date - start_date).days),date_jump):
        yield start_date + timedelta(n)

def calculate_adjusted_close_prices_iterative(df, column):
    """ Iteratively calculates the adjusted prices for the specified column in
    the provided DataFrame. This creates a new column called 'adj_<column name>'
    with the adjusted prices. This function requires that the DataFrame have
    columns with dividend and split_ratio values.

    :param df: DataFrame with raw prices along with dividend and split_ratio
        values
    :param column: String of which price column should have adjusted prices
        created for it
    :return: DataFrame with the addition of the adjusted price column
    """
    adj_column = 'adjcp'

    # Set default values for adjusted price column to 0
    df[adj_column] = 0
    df['Stock Splits'] = df['Stock Splits'].map({0:1})

    # Reverse the DataFrame order, sorting by date in descending order
    df.sort_index(ascending=False, inplace=True)

    price_col = df[column].values
    split_col = df['Stock Splits'].values
    dividend_col = df['Dividends'].values
    adj_price_col = np.zeros(len(df.index))
    adj_price_col[0] = price_col[0]

    for i in range(1, len(price_col)):
        adj_price_col[i] = round((adj_price_col[i - 1] + adj_price_col[i - 1] *
               (((price_col[i] * split_col[i - 1]) -
                 price_col[i - 1] -
                 dividend_col[i - 1]) / price_col[i - 1])), 4)


    df[adj_column] = adj_price_col

    # Change the DataFrame order back to dates ascending
    df.sort_index(ascending=True, inplace=True)
    print(df.head())
    return df

def get_price(ticker, start, end=datetime.now()):
    df = yf.download(ticker, start=start, end=end).reset_index()
    #df = stock.history(start=start, end=end)
    #Calculate ADJ Close
    #df = calculate_adjusted_close_prices_iterative(df, 'Close').reset_index()

    df["ticker"] = ticker
    return df


def add_indicators_to_indice(df):
    stock = Sdf.retype(df)
    stock["macd"]
    stock["rsi_30"]
    return pd.DataFrame(stock)


def add_indices_and_ema(df, indexes):
    start = df.Date.iloc[0]
    end = df.Date.iloc[-1]

    sn = pd.DataFrame()
    for index in indexes:
        snp = get_price(index, start)
        snp = add_indicators_to_indice(snp)
        snp = snp.reset_index().rename(columns={"date": "Date"})
        snp = snp[(snp.Date >= start) & (snp.Date <= end)]
        snp["index_close"] = snp["adj close"]
        snp["index_macd"] = snp["macd"]
        snp["index_rsi"] = snp["rsi_30"]
        snp = snp[["Date", "index_close", "index_macd", "index_rsi", "ticker"]]
        sn = sn.append(snp, ignore_index=True)

    # Some indeces are closed on some days return 0 for those days
    sn.Date = pd.to_datetime(sn.Date)
    df.Date = pd.to_datetime(df.Date)
    sn = sn.reset_index(drop=True).sort_values(by=["Date", "ticker"])
    for date in df.Date.unique():
        if date not in sn.Date.unique():
            d = {"Date": date}

            sn = sn.append(d, ignore_index=True)

        for tick in sn.ticker.unique():
            if tick not in sn[sn.Date == date].ticker.unique():
                d = {"ticker": tick, "Date": date}
                sn = sn.append(d, ignore_index=True)
    sn = sn.sort_values(by=["Date", "ticker"])[
        ["Date", "ticker", "index_close", "index_macd", "index_rsi"]
    ].rename(columns={"ticker": "index_ticker"})
    # merged = merged.set_index('Date').merge(sn.set_index('Date'), left_index=True, right_index=True).reset_index()

    return sn.reset_index(drop=True).fillna(0)


def create_and_preprocess_dataset(company_dictionary, start, end, indices):
    df = get_price(company_dictionary.keys()[0], start, end)
    df["category"] = company_dictionary.values()[0]
    for ticker in company_dictionary.keys()[1:]:
        print("Downloading : " + ticker)
        new_df = get_price(ticker, start, end)
        new_df["category"] = company_dictionary[ticker]
        df = pd.concat([df, new_df])
    df.Date = pd.to_datetime(df.Date)
    print("Start pre-processing")
    sn = preprocess_for_train(df, index_name=indices)
    return sn


def add_stock_info_for_missing_days(df):
    sn_n = df.copy()
    for date in df.Date.unique():
        for tick in df.ticker.unique():
            if tick not in df[df.Date == date].ticker.unique():
                d = {"ticker": tick, "Date": date}
                sn_n = sn_n.append(d, ignore_index=True)
    return sn_n

def add_correlation_matrix(df):
    # TODO: add correlation of stocks to dataframe
    cor_matrix = df
    return df


def preprocess_for_train(
    name, ticker_list, indexes, turbulance, indicator_list, start_date, end_date=datetime.now(), save=True
):
    """
    The order is important for replicability
    :param name: (str) name of the dataset for saving
    :param ticker_list: (List) list of string and dictionaries for stock tickers
    :param indexes: (List) list of tickers
    :return: df(pd.Dataframe) of stock prices with indicators and index_df (pd.Dataframe) on sepearte dataframe
    """

    print("1. Getting prices of stocks and adding technical indicators")

    path = "./datasets/" + name + ".csv"
    if os.path.exists(path):
        print('loading', path)
        sn_n = pd.read_csv(path)
        if turbulance and 'turbulence' not in sn_n.columns:
            print('Calculating turbulance')


            sn_n = sn_n.reset_index().sort_values(by=["ticker", "Date"])
            sn_n = add_turbulance(sn_n)

    else:
        df = get_price(ticker_list[0], start_date, end_date)
        #df = add_technical_indicator(df.rename(columns={"adjcp": "Adj Close"}), indicator_list)
        for tick in ticker_list[1:]:
            stock = get_price(tick, start_date, end_date)
            #stock = add_technical_indicator(
            #    stock.sort_values(by=["ticker", "Date"])
            #    .reset_index(drop=True)
            #    .rename(columns={"adjcp": "Adj Close"}),
            #    indicator_list,
            #)


            df = pd.concat([df, stock])
        df = df[(df.Date >= pd.to_datetime(start_date)) & (df.Date <= pd.to_datetime(end_date))]
        df = add_technical_indicator(df.rename(columns={"adjcp": "Adj Close"}).sort_values(by=["ticker", "Date"])
                                     , indicator_list)
        print("2. Saving processed csv as stocks " + name + ".csv")
        print("Total tickers: ",len(df.ticker.unique()))
        print("3. Adding company categories")
        try:
            df = add_categories(df)
        except:
            print('Category problem')
        df = df.fillna(0)
        print("4. Adding stock information for empty stock days")

        sn_n = add_stock_info_for_missing_days(df)
        sn_n.to_csv("./datasets/" + name + ".csv", index=False)

        if turbulance:
            print("5. Calculating turbulance")
            # Ticker first for turbulance calculation
            sn_n = sn_n.reset_index().sort_values(by=["ticker", "Date"])

            sn_n = add_turbulance(sn_n)
            #print("Total tickers: ", len(sn_n.ticker.unique()))
        else:
            sn_n['turbulence'] = 0
    #else:
    #    sn_n['turbulence'] = 0
    #    sn_n = sn_n.reset_index(drop=True)
    sn_n = sn_n.fillna(0)

    print("6. Adding indexes")

    #index_df = add_indices_and_ema(sn_n.sort_values("Date"), indexes)
    index_df = pd.DataFrame()
    print("Total tickers: ", len(sn_n.ticker.unique()))
    """
    print("7. Check if the dates are the same in two dataframes")
    for date in sn_n.Date.unique():
        if date not in index_df.Date.unique():
            sn_n = sn_n[sn_n.Date != date]
    for date in index_df.Date.unique():
        if date not in sn_n.Date.unique():
            index_df = index_df[index_df.Date != date]
    """
    sn_n.rename(columns={"Volume": "volume", "Adj Close": "adjcp"}, inplace=True)
    sn_n.Date = pd.to_datetime(sn_n.Date)
    sn_n['month'] = sn_n.Date.apply(lambda x: x.month)
    sn_n['day'] = sn_n.Date.apply(lambda x: x.day)
    print("8. Saving last csv as stocks " + name + ".csv")
    sn_n = sn_n.replace([np.inf, -np.inf], 0).sort_values(['Date','ticker'])

    if save:
        sn_n.to_csv("./datasets/" + name + ".csv", index=False)

    return sn_n, index_df


def add_technical_indicator(data, indicator_list):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    df = data.copy()
    df = df.sort_values(by=["ticker", "Date"])
    stock = Sdf.retype(df.copy())
    stock["close"] = stock["adj close"]
    unique_ticker = stock.ticker.unique()

    for indicator in indicator_list:
        indicator_df = pd.DataFrame()
        for i in range(len(unique_ticker)):

            try:
                temp_indicator = stock[stock.ticker == unique_ticker[i]][indicator]
                temp_indicator = pd.DataFrame(temp_indicator)
                temp_indicator["ticker"] = unique_ticker[i]
                temp_indicator["Date"] = df[df.ticker == unique_ticker[i]][
                    "Date"
                ].to_list()
                indicator_df = indicator_df.append(
                    temp_indicator, ignore_index=True
                )
            except Exception as e:
                print(e)
        df = df.merge(
            indicator_df[["ticker", "Date", indicator]], on=["ticker", "Date"], how="left"
        )

    df = df.sort_values(by=["Date", "ticker"])
    return df


def add_turbulance(df):
    """
    add turbulence index from a precalcualted dataframe based on mahanolobis distance
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    df = df.sort_values(["Date", "ticker"])
    turbulence_index = calculate_turbulance(df)
    df = df.merge(turbulence_index, on="Date")
    df = df.sort_values(["Date", "ticker"]).reset_index(drop=True)
    return df


def add_categories(df):

    if os.path.exists('./'+ALL_STOCKS_DATA_FILE):

        sto = pd.read_csv(ALL_STOCKS_DATA_FILE)
    else:
        sto = pd.read_csv(f'../{ALL_STOCKS_DATA_FILE}')
    dictionary = {k:v for k,v in zip(sto['Symbol'],sto['Sector'])}

    df["category"] = df["ticker"].apply(lambda x: dictionary[x.upper()])
    df["category"] = df["category"].astype("category").cat.codes
    return df


def calculate_turbulance(df):
    """calculate turbulence index based on dow 30"""
    if 'Adj Close' in df.columns:
        close_name = 'Adj Close'
    else:
        close_name = 'adjcp'

    df_price_pivot = df.pivot(index="Date", columns="ticker", values=close_name)

    unique_date = df.Date.unique()
    # start after a year
    start = 252
    turbulence_index = [0] * start
    # turbulence_index = [0]
    count = 0

    for i in range(start, len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]

        hist_price = df_price_pivot[
            [n in pd.to_datetime(unique_date[0:i]) for n in df_price_pivot.index]
        ]
        # Clear stocks from interviening calculation
        current_price = current_price.loc[:, (current_price != 0.0).any(axis=0)]
        hist_price = hist_price.loc[:, (hist_price != 0.0).any(axis=0)]

        cov_temp = hist_price.cov()

        current_temp = current_price - np.mean(hist_price, axis=0)

        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(
            current_temp.values.T
        )
        if temp > 0:
            count += 1
            if count > 2:
                turbulence_temp = temp[0][0]
            else:
                # avoid large outlier because of the calculation just begins
                turbulence_temp = 0
        else:
            turbulence_temp = 0
        turbulence_index.append(turbulence_temp)

    turbulence_index = pd.DataFrame(
        {"Date": df_price_pivot.index, "turbulence": turbulence_index}
    )
    return turbulence_index
