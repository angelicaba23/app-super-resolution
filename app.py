#!pip install git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git
import streamlit as st

import cv2

from face_dectec import faceDetection
from save_img import save_image

from skimage.metrics import structural_similarity as ssim

from srcnn import predict


image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
if image_file is not None:
  save_image(image_file, image_file.name)

  img_file = "uploaded_image/" + image_file.name
  
  [img_faces, num, names] = faceDetection(img_file)
  st.write(num)
  st.write(names)
  st.image(img_faces)
  st.image(names[0])



