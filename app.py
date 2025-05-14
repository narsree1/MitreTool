# app.py - Beginning part with correct imports
import pandas as pd
import streamlit as st
import datetime
import time
import json
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

# Import modules - NO TORCH DEPENDENCIES
from modules.data_loader import load_mitre_data, load_library_data_with_embeddings, create_local_mitre_data_cache
from modules.embedding import load_sentence_transformer_model, load_claude_config, CLAUDE_MODELS
from modules.mapper import process_mappings, get_suggested_use_cases
from modules.utils import load_lottie_url
from modules.visualizations import create_navigator_layer

# App configuration
st.set_page_config(
    page_title="MITRE ATT&CK Mapping Tool",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
