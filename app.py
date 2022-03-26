#!pip install git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git
import streamlit as st

from face_dectec import faceDetection
from save_img import save_image

import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas


image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
if image_file is not None:
  save_image(image_file, image_file.name)

  img_file = "uploaded_image/" + image_file.name
  st.image(img_file)
  [img_faces, num, names] = faceDetection(img_file)
  st.write(num)
  st.write(names)
  st.image(img_faces)
  st.image(names[0])

# Specify canvas parameters in application
drawing_mode = "rect"
stroke_width = 3
realtime_update = True
stroke_color = "#000000"
bg_color = "#eee"

# Create a canvas component
if img_file is not None:
  bg_image = Image.open(img_file)
  label_color = (
      st.sidebar.color_picker("Annotation color: ", "#EA1010") + "77"
  )  # for alpha from 00 to FF
  label = st.sidebar.text_input("Label", "Default")
  mode = "transform" if st.sidebar.checkbox("Move ROIs", False) else "rect"

  canvas_result = st_canvas(
      fill_color=label_color,
      stroke_width=3,
      background_image=bg_image,
      height=320,
      width=512,
      drawing_mode=mode,
      key="color_annotation_app",
  )


