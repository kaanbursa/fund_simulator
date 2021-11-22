import streamlit as st
import sys

from matplotlib.backends.backend_agg import RendererAgg
import matplotlib
import matplotlib.pyplot as plt
from pages import model_analysis, trader
sys.path.append('./../')
#from model.models import Trainer, TrainerConfig

PAGES = {
    'Trading Dashboard': trader,
    "Model Analysis Dashboard": model_analysis,

}
st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()