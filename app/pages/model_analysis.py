import json

import streamlit as st
import math
import pandas as pd
from loguru import logger
from utils.helpers import display_buy_sell
import os
import sys

@logger.catch
def app():
    root_path = '../results/'
    trade_memory = os.listdir(root_path)
    trade_memory = filter( lambda x: 'trade_memory' in x, trade_memory)
    user_input = st.selectbox(label='Select a trade history to analyze', options=trade_memory)
    if 'csv' in user_input:
        df= pd.read_csv(root_path + user_input)
        df.Date = pd.to_datetime(df.Date)
        tickers = df.ticker.unique()
        st.write(df)
        value = st.selectbox("Stocks", tickers)

        st.markdown(value)
        print(value)
        with st.spinner('Wait for it...'):
            display_buy_sell(value, df)

    else:
        st.write('Waiting for csv')
