#!pip install git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git
from io import StringIO
import cv2
import face_detection
import streamlit as st
import numpy as np
import os
from PIL import Image

print(face_detection.available_detectors)
detector = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)


def faceDetection(input_image_path):
  im = cv2.imread(input_image_path)[:, :, ::-1]
  detections = detector.detect(im)
  print(len(detections))
  num=0

  image = cv2.imread(input_image_path)
  for detections in detections:
    x = int(detections[0])
    y = int(detections[1])
    w = int(detections[2])
    h = int(detections[3])
    cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
    num+=1
  image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  st.image(image)

image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
if image_file is not None:
  file_details = {"FileName": image_file.name,
                  "FileType": image_file.type}
  st.write(file_details)

  # @ save image
  with open(os.path.join("uploaded_image", image_file.name), "wb") as f:
    f.write(image_file.getbuffer())

  img_file = "uploaded_image/" + file_details['FileName']

  faceDetection(img_file)

