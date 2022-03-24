import cv2
import face_detection
import streamlit as st
from PIL import Image

print(face_detection.available_detectors)
detector = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

def crop_object(image, box, num, names):
  x_top_left = box[0]
  y_top_left = box[1]
  x_bottom_right = box[2]
  y_bottom_right = box[3]

  crop_img = image.crop((int(x_top_left), int(y_top_left), int(x_bottom_right), int(y_bottom_right)))
  name = "crop_img_"+str(num)+".png"
  names.append(name)
  crop_img.save(name)
  return crop_img

@st.cache(suppress_st_warning=True)
def faceDetection(input_image_path):
  im = cv2.imread(input_image_path)[:, :, ::-1]
  detections = detector.detect(im)
  print(len(detections))
  num=0

  image_landmarks = cv2.imread(input_image_path)
  names = [] 
  for detections in detections:
    x = int(detections[0])
    y = int(detections[1])
    w = int(detections[2])
    h = int(detections[3])
    cv2.rectangle(image_landmarks, (x, y), (w, h), (0, 255, 0), 2)
    print(x, y, w, h)
    cv2.putText(image_landmarks, 'X', (w-10, y+10),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0,255), 3,cv2.LINE_AA )
    image = Image.open(input_image_path)
    st.image(crop_object(image, detections, num, names))

    num+=1
  image_landmarks = cv2.cvtColor(image_landmarks, cv2.COLOR_BGR2RGB)
  return image_landmarks, num, names
