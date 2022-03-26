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
if image_file is not None:
  canvas_result = st_canvas(
      fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
      stroke_width=stroke_width,
      stroke_color=stroke_color,
      background_color=bg_color,
      background_image=Image.open(image_file) if image_file else None,
      update_streamlit=realtime_update,
      height=150,
      drawing_mode=drawing_mode,
      #point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
      key="canvas",
  )

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"])
    for col in objects.select_dtypes(include=["object"]).columns:
        objects[col] = objects[col].astype("str")
    st.dataframe(objects)


