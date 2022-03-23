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

