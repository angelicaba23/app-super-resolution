#!pip install git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git
from io import StringIO
import cv2
from face_dectec import faceDetection
import streamlit as st
import numpy as np
import os
from PIL import Image

image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
if image_file is not None:
  file_details = {"FileName": image_file.name,
                  "FileType": image_file.type}
  st.write(file_details)

  # @ save image
  with open(os.path.join("uploaded_image", image_file.name), "wb") as f:
    f.write(image_file.getbuffer())

  img_file = "uploaded_image/" + file_details['FileName']

  img_faces = faceDetection(img_file)
  st.image(img_faces)

