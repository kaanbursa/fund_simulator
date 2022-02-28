import numpy as np
import pandas as pd
from data.preprocessing import DataProcessor
from newsapi import NewsApiClient
import yfinance as yf


class OrganizationBase:
    def __init__(self, name, org_type):
        self.name = name
        self.type = org_type
        self.data_processor = DataProcessor(org_type, tickers=[])
        news_api = ''
        self.newsapi = NewsApiClient(api_key=news_api)
        self.sources = self.newsapi.get_sources()

    def get_price(self, start, end):
        """
        Get price for given date
        :return:  return price for the day
        """

        if self.type.lower() == 'stocks':

            df = yf.download(self.name, start=start, end=end).reset_index()
            # df = stock.history(start=start, end=end)
            # Calculate ADJ Close
            # df = calculate_adjusted_close_prices_iterative(df, 'Close').reset_index()

            df["ticker"] = self.name
            self.data = df
        else:
            # TODO: Import crypto
            raise NotImplementedError

    def get_key_people(self):
        """
        Returns key people of the company
        :return:
        """
    def create_metadata(self):
        """
        Creates metadata of the organization
        :return:
        """

    def get_balance_sheet(self, period):
        """
        Returns the balance sheet of the company
        :param period:
        :return:
        """

    def get_metadata(self) -> dict:
        """
        Returns fundamentals of the company as dict
        :return:
        """

    def get_news(self, start, end, sources= '', domain=''):
        """
        Finds news of the company
        :param start: Starting period
        :param end: Ending period for the new
        :return:
        """

        assert sources in self.sources, 'source not found'
        all_articles = self.newsapi.get_everything(q=self.name,
                                              sources=sources,
                                              domains=domain,
                                              from_param=start,
                                              to=end,
                                              language='en',
                                              sort_by='relevancy',
                                              page=2)
        return all_articles

    def create_technical_indicators(self, indicators_list : list):
        """
        Create Technical indicators for the company using stockstats
        :param indicators_list:
        :return:
        """
        self.indicators_data = self.data_processor.add_technical_indicator(self.data, indicators_list)

    def add_news(self, news):
        """
        Add news to the company profile
        :param news:
        :return:
        """

    def search_news(self, start, end):
        """
        Searches new given period
        :param start: Starting date
        :param end: Ending period for the news
        :return:
        """

    def add_social(self, social_profile: dict):
        """
        Add link to its social profile
        :param social_profile: social profile of the company
        :return:
        """



class Company(OrganizationBase):
    def __init__(self, name, org_type):
        super(Company, self).__init__(name, org_type)





