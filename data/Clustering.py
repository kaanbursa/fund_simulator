import sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MiniBatchKMeans, KMeans
from scipy.cluster.hierarchy import fcluster

from data.preprocessing import add_stock_info_for_missing_days


class UnsupervisedLearning:

    def __init__(self, dataframe):
        """
        Dataframe is from yfinance a concatinated dataframe of stocks
        preprocessing will be done here
        :param dataframe:
        """
        self.dataframe = self.preprocess_dataframe(dataframe)

        self._normalize()

    def preprocess_dataframe(self, df):
        df = add_stock_info_for_missing_days(df)
        data = []
        headers = []
        full_stock_df = []
        index = df.Date.unique()
        for tick in df.ticker.unique():
            stock = df[df.ticker == tick]
            data.append(stock['Close'].values)
            headers.append(tick)
            full_stock_df.append(stock.set_index('Date'))


        stock_prices = pd.DataFrame()

        for i in range(len(data)):


            stock_prices[headers[i]] = data[i]

        stock_prices = stock_prices.set_index(index)

        return stock_prices

    def _set_date_index(self):
        pass

    def _normalize(self):
        scaler = MinMaxScaler()
        normalized_stocks = pd.DataFrame(scaler.fit_transform(self.dataframe), columns=self.dataframe.columns)
        # To prevent value error
        # TODO: check for better implementation than filling with 0
        normalized_stocks = normalized_stocks.replace([np.inf, -np.inf], 0).fillna(0)
        self.dataframe = normalized_stocks.T

    def principle(self, n_components=5):
        """
        We have to make sure that stocks are rows days are columns
        n_componenets: how many components do you want to seperate the dataset into
        :return:
        """
        data = self.dataframe.T
        pcamodel = PCA(n_components=n_components)
        return pcamodel.fit_transform(data)

    def pick_stocks_by_volatility(self):
        raise NotImplementedError

    def kmeans_cluster_stocks(self, number_of_clusters = 5, minibatch=False):
        """
        Applies kmeans clustering to data
        :param data:
        :param number_of_clusters:
        :param minibatch:
        :return: N stocks which are most representative of their data. List of Lists
        """
        dataframe = self.dataframe
        if minibatch:
            kmeans = MiniBatchKMeans(n_clusters=number_of_clusters, random_state=0)
        else:
            kmeans = KMeans(n_clusters=number_of_clusters, random_state=0)

        ds = kmeans.fit_transform(dataframe)
        cluster_labels = kmeans.labels_

        df_clst = pd.DataFrame()
        df_clst["index"] = dataframe.index
        df_clst.set_index("index", inplace=True)

        labels = []
        d = []

        for i in range(len(dataframe)):
            labels.append(cluster_labels[i])

            #pick by maximizing distance to other clusters
            d.append(sum([a for a in ds[i] if a != min(ds[i])]))
        df_clst["label"] = labels
        df_clst["distance"] = d

        return df_clst


    def _create_linkage_matrix_for_cluster(self, model):
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
        return linkage_matrix

    def agglomerative_cluster(self, distance_threshold, number_of_clusters=10):
        """
        Return flattened clusteres
        :param distance_threshold:
        :return:
        """
        dataframe = self.dataframe
        model = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None)

        model = model.fit(dataframe)
        linkage_matrix = self._create_linkage_matrix_for_cluster(model)
        label = fcluster(linkage_matrix, number_of_clusters, criterion='maxclust')
        df_clst = pd.DataFrame()
        df_clst['index'] = dataframe.index
        df_clst['label'] = label
        clusters = [x[1]['index'] for x in df_clst.groupby('label')]
        return clusters

    def k_means_cluster(self, number_of_clusters=5):
        """
        Return flattened clusters using k-means clustering algorithm
        Will always return n clusters as long as >n assets are passed.
        """
        dataframe = self.dataframe

        df_clst = pd.DataFrame()
        df_clst["index"] = dataframe.index

        kmeans = MiniBatchKMeans(n_clusters=number_of_clusters)

        df_clst["label"] = kmeans.fit_predict(dataframe)
        df_clst["label"] = df_clst["label"].astype("int")

        clusters = [x[1]["index"] for x in df_clst.groupby('label')]

        return clusters

    def pick_stocks_by_volatility(self):
        raise NotImplementedError







