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
    user_input = st.text_input("label goes here", '../results/trade_memory_wb-v8.csv')
    if 'csv' in user_input:
        df= pd.read_csv(user_input)
        df.Date = pd.to_datetime(df.Date)
        tickers = df.ticker.unique()
        st.write(df)
        value = st.selectbox("Stocks", tickers)

        st.write(value)
        print(value)
        fig = display_buy_sell(value, df)
        st.pyplot(fig=fig)
    else:
        st.write('Waiting for csv')
