


class Stock():
    def __init__(self, ticker, indicators):

        self.ticker = ticker
        self.indicators = indicators
        self.holdings = 0
        self.holding_history = {}

    def get_indicators(self, i):
        return self.indicators[i]

    def transact_holdings(self, amount, date):
        self.holdings += amount
        self.holding_history[date] = self.holdings

    def get_total_holdings(self):
        return self.holdings

    def get_ticker(self):
        return self.ticker