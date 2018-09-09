from __future__ import print_function
from sklearn.externals import joblib
from hog import HOG
import dataset
import argparse
import mahotas
import cv2
import imutils
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to where the model will be stored")
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

model = joblib.load(args["model"])
hog = HOG(orientations=18, pixelsPerCell=(10,10), cellsPerBlock=(1,1), transform=True)

image = cv2.imread(args["image"])
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (3,3), 0)
cv2.imshow("blur",img)
cv2.waitKey(0)
T = mahotas.thresholding.otsu(img)
img[img > T] = 255
cv2.imshow("otsu",img)
cv2.waitKey(0)
edged = cv2.Canny(img, 30, 150)
cv2.imshow("Edges",edged)
cv2.waitKey(0)
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key = lambda x: x[1])

for (c, _) in cnts:
  (x,y,w,h) = cv2.boundingRect(c)
  if w>=7 and h>=20:
    roi = image[y:y+h, x:x+w]
    thresh = roi.copy()
    T = mahotas.thresholding.otsu(roi)
    thresh[thresh > T] = 255
    thresh = dataset.deskew(thresh, 20)
    thresh = dataset.center_extent(thresh, (20,20))
    hist = hog.describe(thresh)
    digit = model.predict([hist])[0]
    print("I think that number is : {}".format(digit))
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 1)
    cv2.putText(image, str(digit), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)
            