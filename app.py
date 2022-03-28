#!pip install git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git
import json
import streamlit as st

from face_dectec import faceDetection
from save_img import save_image

import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from write_json import write_json

im = Image.open("icon.ico")
st.set_page_config(
    page_title="SuperResolution",
    page_icon=im,
    layout="wide",
)

image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
if image_file is not None:
  save_image(image_file, image_file.name)

  img_file = "uploaded_image/" + image_file.name
  #st.image(img_file)
  [img_faces, num, names] = faceDetection(img_file)
  #st.write(num)
  #st.write(names)
  #st.image(img_faces)
  #st.image(names[0])
  #save_image(img_faces, "img_faces.png")
  if len(names) > 0:
    x = 0
    st.write(x)
    x += 1

    a_file = open("saved_state.json", "r")
    json_object = json.load(a_file)
    a_file.close()
    print(json_object)

    new_obj = {
      "type": "rect",
        "left": 10,
        "top": 91,
        "width": 10,
        "height": 91,
        "fill": "#00ff00",
        "stroke": "#00ff00",
        "strokeWidth": 3
    }

    a_file = open("saved_state.json", "r+")
    json_object = json.load(a_file)
    json_object["objects"][0] = new_obj
    #a_file.seek(0)
        # convert back to json.
    json.dump(json_object, a_file, indent = 4)
    a_file.close()
   

    with open("saved_state.json", "r") as f:   saved_state = json.load(f)
    st.write(saved_state)
    
    bg_image = Image.open(img_file)
    label_color = (
        st.sidebar.color_picker("Annotation color: ", "#00ff00") + "50"
    )  # for alpha from 00 to FF
    tool_mode = st.sidebar.selectbox(
      "Select tool:", ("draw", "move")
  )
    mode = "transform" if tool_mode=="move" else "rect"

    st.write(label_color)

    canvas_result = st_canvas(
        fill_color=label_color,
        stroke_width=3,
        stroke_color="#00ff00",
        background_image=bg_image,
        height=bg_image.height,
        width=bg_image.width,
        initial_drawing=saved_state,
        drawing_mode=mode,
        key="color_annotation_app",
    )

    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")
        st.dataframe(objects)
  else:
    st.write("NO PERSON DETECTED")





