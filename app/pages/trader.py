import streamlit as st
import sys
import pandas as pd
from loguru import logger
from matplotlib.backends.backend_agg import RendererAgg
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px

sys.path.append('./../')
#from model.models import Trainer, TrainerConfig
# Configs
st.set_page_config(layout="wide")
matplotlib.use("agg")
i = 0
def start_trading():
    global i
    i += 1
    return i

@logger.catch
def app():
    first_container_html = "<h1 style='text-align: center; color: black;'>Simulator for Fund Manager Agent</h1>" \
                           "<br>" \
                           "<h2 style='text-align: center; color: black;'>Analyzing Trading Strategy and Agent Behaviour</h2>"
    st.markdown(first_container_html, unsafe_allow_html=True)
    st.button('Start Trading', on_click=start_trading)

    st.text(start_trading())

    _lock = RendererAgg.lock
    all_stocks = pd.read_csv('../data/all_stocks.csv')
    all_stocks = [symbol + ' - ' + name for symbol, name in zip(all_stocks.Symbol, all_stocks.Name)]
    all_indicators = ['CM','SMA']
    models = ['PPO','SAC','DDPG']

    st.text("")
    st.text("")
    st.text("")
    col1, col2 = st.columns(2)

    trade_submited=False

    with st.form("my_form"):

        st.write("Inside the form")
        tickers = st.multiselect(label='Select tickers', options=all_stocks)
        indicators = st.multiselect(label='Select indicators', options=all_indicators)
        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = st.date_input('Start Training Date')
        with col2:
            start_trade_date = st.date_input('Start Trading Date')
        with col3:
            end_date = st.date_input('End Date')
        col1, col2 = st.columns(2)
        with col1:
            model = st.selectbox('Select model', models)
        with col2:
            if model is not None:
                policy_kwargs = st.selectbox('Select policy', models)

        checkbox_val = st.checkbox("Form checkbox")

        submitted = st.form_submit_button("Submit")

        if submitted:
            st.write( "checkbox", checkbox_val)
            trade_submited = True
    st.text("")
    st.text("")
    if trade_submited:
        st.markdown(f"**Model Training** is {model} <br /> stocks selected for trading is {', '.join(tickers)}"
                    f"")

        with col1:
            st.header('Start Trading')

        with col2:
            st.header('Current Stocks Trading')

        st.text("")
        st.text("")
        st.text("")
        st.markdown('Trading from period to : TSLA AAPL')
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Balance", "1000000$", "0 %")
        col2.metric("Profit", "0 $", "0%")
        col3.metric("Stocks Traded", "10", "+10")

        df = px.data.gapminder()

        fig = px.scatter(df.query("year==2007"), x="gdpPercap", y="lifeExp",
                         size="pop", color="continent",
                         hover_name="country", log_x=True, size_max=60)
        st.plotly_chart(fig, use_container_width=True)
        row1_spacer1, row1_1, row1_spacer2 = st.columns((.1, 3.2, .1))

        row1_1.header('Current Period Portfolio Allocation')
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = 'TSLA', 'AAPL', 'GOOGL', 'DOLAR'
        sizes = [15, 30, 45, 10]
        explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.patch.set_alpha(0.0)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        row1_1.pyplot(fig1, )