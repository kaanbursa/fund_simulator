import streamlit as st
import sys

from matplotlib.backends.backend_agg import RendererAgg
import matplotlib
import matplotlib.pyplot as plt
from pages import model_analysis
sys.path.append('./../')
#from model.models import Trainer, TrainerConfig

PAGES = {
    "model_analysis": model_analysis,
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()