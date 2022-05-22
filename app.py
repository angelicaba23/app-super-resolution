import streamlit as st
from PIL import Image

import json
import cv2
import numpy as np

from face_dectec import crop_object, faceDetection
from srcnn import predict
import pandas as pd
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="SuperResolution",
    page_icon=Image.open("icon.ico"),
    layout="wide"
)

image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
if image_file is not None:
  #save_image(image_file, image_file.name)

  #img_file = "uploaded_image/" + image_file.name

  file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes, 1)
  print("faceDetection")
  [img_faces, num, boxes] = faceDetection(opencv_image)
  print("numero de rostros = "+ str(num))
  
  #st.write(boxes)
  #st.image(img_faces)
  if len(boxes) > 0:
    list = []
    filename = 'saved_state.json'

    for boxes in boxes:
      list.append({
        "type": "rect",
          "left": boxes[0],
          "top": boxes[1],
          "width": boxes[2]-boxes[0],
          "height": boxes[3]-boxes[1],
          "fill": "#00ff0050",
          "stroke": "#00ff00",
          "strokeWidth": 3
      })

    # Verify updated list
    #st.write(list)

    listObj = {
        "version": "4.4.0",
        "objects": list  
    }

    # Verify updated listObj
    #st.write(listObj)

    with open(filename, 'w') as json_file:
      json.dump(listObj, json_file, 
                          indent=4,  
                          separators=(',',': '))

    with open(filename, "r") as f:   saved_state = json.load(f)
    #st.write(saved_state)
    
    bg_image = Image.open(image_file)
    label_color = (
        st.sidebar.color_picker("Annotation color: ", "#00ff00") + "50"
    )  # for alpha from 00 to FF
    tool_mode = st.sidebar.selectbox(
      "Select tool:", ("draw", "move")
    )
    mode = "transform" if tool_mode=="move" else "rect"

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
        print("Canvas creado")
        rst_objects = canvas_result.json_data["objects"]
        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
        n = int(len(rst_objects))
        cols = st.columns(n)
        cols_srcnn = st.columns(n)
        #cols_srgan = st.columns(n)
        i = 0

        for rst_objects in rst_objects:
          rts_boxes = [rst_objects['left'],rst_objects['top'],rst_objects['width']+rst_objects['left'],rst_objects['height']+rst_objects['top']]
          #st.write(rts_boxes)
          crop_image = crop_object(bg_image, rts_boxes)
          cols[i].image(crop_image)
          #cols_srcnn[i].image(predict(crop_image))
          #cols_srgan[i].image(predictgan(crop_image))
          print("img" + str(i))
          i += 1

        if st.button("PROCESAR"):
          st.warning("SRCNN")
          i = 0
          for rst_objects in rst_objects:
            rts_boxes2 = [rst_objects['left'],rst_objects['top'],rst_objects['width']+rst_objects['left'],rst_objects['height']+rst_objects['top']]
            #st.write(rts_boxes)
            crop_image = crop_object(bg_image, rts_boxes2)
            #cols[i].image(crop_image)
            cols_srcnn[i].image(predict(crop_image))
            #cols_srgan[i].image(predictgan(crop_image))
            print("img" + str(i))
            i += 1

        #for col in objects.select_dtypes(include=['object']).columns:
        #    objects[col] = objects[col].astype("str")
        #st.dataframe(objects)

        #if st.button("Procesar"):
        #  st.write("cargando")
    
  else:
    st.write("NO PERSON DETECTED")