import time

import cv2
import math
from pyzbar.pyzbar import decode

start_time_1 = time.time()
frame = cv2.imread(r"C:\Users\Admin\Desktop\455.jpg")
threshold = 0.6
maxWidth = 1280; maxHeight = 720
imgHeight, imgWidth = frame.shape[:2]
hScale = 1; wScale = 1
thickness = 1
if imgHeight > maxHeight:
    hScale = imgHeight / maxHeight
    thickness = 6
if imgWidth > maxWidth:
    wScale = imgWidth / maxWidth
    thickness = 6

classes = open(r'C:\Users\Admin\Desktop\yolov3_tiny\qrcode.names').read().strip().split('\n')
net = cv2.dnn.readNetFromDarknet(r'C:\Users\Admin\Desktop\yolov4-tiny-custom-416.cfg', r'C:\Users\Admin\Desktop\yolov4-tiny-custom-416_last.weights')

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)


def postprocess(frame, outs):
    frameHeight, frameWidth = frame.shape[:2]

    classIds = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold:
                x, y, width, height = detection[:4] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
                left = int(x - width / 2)
                top = int(y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, int(width), int(height)])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold, threshold - 0.1)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        # Draw bounding box for objects
        cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 0, 255), thickness)

        imgCrop = frame[top:top+height, left:left+width]
        cv2.imshow('IMG', imgCrop)
        cv2.waitKey(0)
        print(decode(imgCrop))
        # Draw class name and confidence
        label = '%s:%.2f' % (classes[classIds[i]], confidences[i])
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

net.setInput(blob)
start_time = time.monotonic()
# Compute
outs = net.forward(ln)
elapsed_ms = (time.monotonic() - start_time) * 1000
print('forward in %.1fms' % (elapsed_ms))

start_time = time.monotonic()
postprocess(frame, outs)
elapsed_ms = (time.monotonic() - start_time) * 1000
print('postprocess in %.1fms' % (elapsed_ms))

print(time.time() - start_time_1)

cv2.imshow('IMG', frame)
cv2.waitKey(0)