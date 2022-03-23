#!pip install git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git
from io import StringIO
import cv2
from face_dectec import crop_object, faceDetection
from streamlit_cropper import st_cropper
import streamlit as st
import numpy as np

from PIL import Image

from save_img import save_image

image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
if image_file is not None:
  save_image(image_file, image_file.name)

  img_file = "uploaded_image/" + image_file.name
  
  [img_faces, num, names] = faceDetection(img_file)
  st.write(num)
  st.write(names)
  st.image(img_faces)
  
  aspect_dict = {
      "1:1": (1, 1),
      "16:9": (16, 9),
      "4:3": (4, 3),
      "2:3": (2, 3),
      "Free": None
  }
  aspect_ratio = aspect_dict["Free"]
  if img_file:
    img = Image.open(img_file)
    #if not realtime_update:
    #    st.write("Double click to save crop")
    # Get a cropped image from the frontend
    
    cropped_img = st_cropper(img, realtime_update=True, box_color='#0000FF')
    if st.button('CROP'):
      st.image(crop_object(img, cropped_img, num))
    # Manipulate cropped image at will
    st.write("Preview")
    
    _ = cropped_img.thumbnail((150,150))
    st.image(cropped_img)
