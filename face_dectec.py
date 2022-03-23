import cv2
import face_detection
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
  return image
