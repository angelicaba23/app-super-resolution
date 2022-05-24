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


# create ss object
if 'data' not in st.session_state:
    st.session_state.data = None

# app design
app_meta('🖼️')
set_bg_hack('extra/bq.png')


#style
styl = f"""
<style>
	.css-14ngwub .st-ct{{
    background-color: #97fdf5;
	}}
	}}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)


# set logo in sidebar using PIL
logo = Image.open('extra/name.png')
st.sidebar.image(logo, 
                    use_column_width=True)


# Main panel setup
display_app_header(main_txt='Super Resolution',
                    sub_txt='Upload, procces, download to get a new resolution')
# Info
with st.expander("What is this app?", expanded=False):
    st.write("hola")
    st.write("")
st.write("")

st.markdown("""---""")
    
#image_data_app()



#sidebar
st.sidebar.image('extra/upload.png', use_column_width=True)
#st.sidebar.app_section_button("[GitHub](https://github.com/angelicaba23/app-super-resolution)")

display_app_header(main_txt = "Step 1",
                  #sub_txt= "Upload data",
                  is_sidebar=True)

image_file = st.sidebar.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if image_file is not None:

  
  #save_image(image_file, image_file.name)

  #img_file = "uploaded_image/" + image_file.name

  file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes, 1)
  
  [img_faces, num, boxes] = faceDetection(opencv_image)
  print("numero de rostros = "+ str(num))
  #st.write(boxes)
  #st.image(img_faces)
  if len(boxes) > 0:
    display_app_header(main_txt = "Step 2",
                  sub_txt= "Edit image",
                  is_sidebar=True)
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
        st.sidebar.color_picker("Annotation color: ", "#00FFB3") 
    )  # for alpha from 00 to FF
    tool_mode = st.sidebar.selectbox(
      "Select tool:", ("Add", "Move & edit")
    )
    mode = "transform" if tool_mode=="move" else "rect"

    canvas_result = st_canvas(
        fill_color=label_color+ "50",
        stroke_width=3,
        stroke_color=label_color,
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
        st.info("SuperResolution (CNN)")
        cols_srcnn = st.columns(n)
        #cols_srgan = st.columns(n)
        i = 0
        for rst_objects in rst_objects:
          rts_boxes = [rst_objects['left'],rst_objects['top'],rst_objects['width']+rst_objects['left'],rst_objects['height']+rst_objects['top']]
          #st.write(rts_boxes)
          crop_image = crop_object(bg_image, rts_boxes)
          cols[i].image(crop_image)
          im_bgr = predictCNN(crop_image)
          cols_srcnn[i].image(im_bgr)

          im_rgb = im_bgr[:, :, [2, 1, 0]] #numpy.ndarray
          ret, img_enco = cv2.imencode(".png", im_rgb)  #numpy.ndarray
          srt_enco = img_enco.tostring()  #bytes
          img_BytesIO = BytesIO(srt_enco) #_io.BytesIO
          img_BufferedReader = BufferedReader(img_BytesIO) #_io.BufferedReader

          cols_srcnn[i].download_button(
            label="Download 📥",
            data=img_BufferedReader,
            file_name="srcnn_img_"+str(i)+".png",
            mime="image/png"
          )

          #cols_srgan[i].image(predictgan(crop_image))
          print("img" + str(i))
          i += 1
        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")
        #st.dataframe(objects)

        #if st.button("Procesar"):
        #  st.write("cargando")
    
  else:
    st.write("NO PERSON DETECTED")