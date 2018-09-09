import argparse
import mahotas
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (3,3), 0)
T = mahotas.thresholding.otsu(img)
img[img > T] = 255
edged = cv2.Canny(img, 30, 150)
cv2.imshow("Edges",edged)
cv2.waitKey(0)

(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key = lambda x: x[1])
for (c, _) in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    if w>=100 and h>=50:
        (x,y,w,h) = cv2.boundingRect(c)
        roi = image[y:y+h, x:x+w]
        cv2.imwrite("plate9.jpg", roi)