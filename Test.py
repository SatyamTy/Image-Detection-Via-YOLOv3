import cv2
import matplotlib.pyplot as plt
import numpy as np

yolo = cv2.dnn.readNet(".//yolov3-tiny.weights", ".//yolov3-tiny.cfg")

classes = []

with open(".//coco.names", 'r') as f:
    classes = f.read().splitlines()

len(classes)

img = cv2.imread("./Target.jpeg")
blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0,0,0), swapRB=True, crop=False)

# i = blob[0].reshape(320, 320, 3)
# plt.imshow(i)

yolo.setInput(blob)
output_layer_name = yolo.getUnconnectedOutLayersNames()
layeroutput = yolo.forward(output_layer_name)

boxes = []
confidences = []
class_ids = []
height = blob.shape[0]
width = blob.shape[1]

for output in layeroutput : 
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > 0.7:
            center_x = int(detection[0]*width)
            center_y = int(detection[0]*height)
            w = int(detection[0]*width)
            h = int(detection[0]*height)

            x = int(center_x- w/2)
            y = int(center_y- h/2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

len(boxes)            