#!pip install git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git
import cv2
import face_detection
import streamlit as st
from PIL import Image
import numpy as np

st.write(face_detection.available_detectors)
detector = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

def landmarks(image, box, num):
  x_top_left = box[0]
  y_top_left = box[1]
  x_bottom_right = box[2]
  y_bottom_right = box[3]
  x_center = (x_top_left + x_bottom_right) / 2
  y_center = (y_top_left + y_bottom_right) / 2
  end_point = (int(x_bottom_right),int(y_bottom_right))
  start_point = (int(x_top_left), int(y_top_left))
  color = (0,0,255)
  thickness = 5
  image = cv2.rectangle(image, start_point, end_point, color, thickness)
  name = "crop_img_"+str(num)+".png"
  image.save(name)
  return image

input_image_path = '/content/drive/MyDrive/ObjectDetection/Deep-Learning-Face-Detection-main/imgs/'
input_image_path = input_image_path +'selfie40.jpeg'
input_image_path = 'selfie40.jpeg'
st.image(input_image_path)
st.write(type(input_image_path))

im = cv2.imread(input_image_path)[:, :, ::-1]
detections = detector.detect(im)
print(len(detections))
num=0

image = cv2.imread(input_image_path)
for detections in detections:
  #image = Image.open(input_image_path)
  x = int(detections[0])
  y = int(detections[1])
  w = int(detections[2])
  h = int(detections[3])
  cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
  num+=1
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
st.write(type(image))
st.image(image)

