from image_processing import *
import streamlit as st
from streamlit import caching

st.set_page_config(
    page_title="4M",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# > Creator: Gordon D. Pisciotta  ·  4M  ·  [modern.millennial.market.mapping]",
    }
)


st.title('MRI tumor detection')

st.subheader('Upload the MRI scan of the brain')
uploaded_file = st.file_uploader(' ',accept_multiple_files = False)

if uploaded_file is not None:
        st.text(type(img))
        img = preprocessing(uploaded_file)
        img = predict_tumur(img)

        st.image(img)
        
else:
        st.write("Image not found")
        
