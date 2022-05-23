from io import BufferedReader, BytesIO
import streamlit as st
from PIL import Image

import json
import cv2
import numpy as np

from face_dectec import crop_object, faceDetection
from srcnn import predictCNN
#from srgan import predictSrgan
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from helper_functions import *

# Page config
#st.set_page_config(page_title="SuperResolution",layout="wide")


# app setup 
try:

    # create ss object
    if 'data' not in st.session_state:
        st.session_state.data = None

    # app design
    app_meta('🖼️')
    set_bg_hack('bg.png')

    # set logo in sidebar using PIL
    logo = Image.open('logo.png')
    st.sidebar.image(logo, 
                        use_column_width=True)
    
    # hide warning for st.pyplot() deprecation
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Main panel setup
    display_app_header(main_txt='Data Quality Wrapper',
                       sub_txt='Clean, describe, visualise and select data for AI models')

    st.markdown("""---""")
    # provide options to user to navigate to other dqw apps
    app_section_button('Image Data Section 🖼️',
    '[Tabular Data Section 🏗️](https://share.streamlit.io/soft-nougat/dqw-ivves_structured/main/app.py)',
    '[Audio Data Section 🎶](https://share.streamlit.io/soft-nougat/dqw-ivves_audio/main/app.py)',
    '[Text Data Section 📚](https://share.streamlit.io/soft-nougat/dqw-ivves_text/main/app.py)')
    st.markdown("""---""")
    
    #image_data_app()

except KeyError:
    st.error("Please select a key value from the dropdown to continue.")
    
except ValueError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
    
except TypeError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
