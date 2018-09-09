from sklearn.cluster import MiniBatchKMeans
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-c", "--clusters", required = True, type = int, help = "# of clusters")
args = vars(ap.parse_args())
# Orange Screen 
oLower = np.array([0, 51, 102], dtype = "uint8")
oUpper = np.array([105, 180, 255], dtype = "uint8")
# Green Screen
gLower = np.array([0, 160, 0], dtype = "uint8")
gUpper = np.array([153, 255, 153], dtype = "uint8")
# Blue Screen 
bLower = np.array([204, 0, 0], dtype = "uint8")
bUpper = np.array([255, 255, 204], dtype = "uint8")

meter = cv2.imread(args["image"])
m1 = meter.copy()
m2 = meter.copy()
m3 = meter.copy()
image = meter.copy()
(h, w) = image.shape[:2]

image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
image = image.reshape((image.shape[0] * image.shape[1], 3))
clt = MiniBatchKMeans(n_clusters = args["clusters"])
labels = clt.fit_predict(image)
quant = clt.cluster_centers_.astype("uint8")[labels]
quant = quant.reshape((h, w, 3))
image = image.reshape((h, w, 3))
quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)

img1 = quant.copy()
Display1 = cv2.inRange(img1, oLower, oUpper)
Display1 = cv2.GaussianBlur(Display1, (3, 3), 0)
(cnts, _) = cv2.findContours(Display1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)	
if len(cnts) > 0:
	cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
	rect = np.int32(cv2.cv.BoxPoints(cv2.minAreaRect(cnt)))
	cv2.drawContours(img1, [rect], -1, (0, 255, 0), 2)
	mask1 = np.zeros_like(img1)
	cv2.drawContours(mask1, [rect], -1, (255,255,255), -1) 
	out1 = np.zeros_like(img1) 
	out1[mask1 == 255] = m1[mask1 == 255]
	cv2.imshow('Output1', out1)
	cv2.waitKey(0)

img2 = quant.copy()
Display2 = cv2.inRange(img2, gLower, gUpper)
Display2 = cv2.GaussianBlur(Display2, (3, 3), 0)
(cnts, _) = cv2.findContours(Display2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)	
if len(cnts) > 0:
	cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
	rect = np.int32(cv2.cv.BoxPoints(cv2.minAreaRect(cnt)))
	cv2.drawContours(img2, [rect], -1, (0, 255, 0), 2)
	mask2 = np.zeros_like(img2) 
	cv2.drawContours(mask2, [rect], -1, (255,255,255), -1) 
	out2 = np.zeros_like(img2) 
	out2[mask2 == 255] = m2[mask2 == 255]
	cv2.imshow('Output2', out2)
	cv2.waitKey(0)

img3 = quant.copy()
Display3 = cv2.inRange(img3, bLower, bUpper)
Display3 = cv2.GaussianBlur(Display3, (3, 3), 0)
(cnts, _) = cv2.findContours(Display3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)	
if len(cnts) > 0:
	cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
	rect = np.int32(cv2.cv.BoxPoints(cv2.minAreaRect(cnt)))
	cv2.drawContours(img3, [rect], -1, (0, 255, 0), 2)
	mask3 = np.zeros_like(img3) 
	cv2.drawContours(mask3, [rect], -1, (255,255,255), -1) 
	out3 = np.zeros_like(img3) 
	out3[mask3 == 255] = m3[mask3 == 255]
	cv2.imshow('Output3', out3)
	cv2.waitKey(0)

"""
save = input("Which one do you want to save ? ")
if save == 1:
	cv2.imwrite("PRO/Phase/plateO.jpg",out1)
elif save == 2:
	cv2.imwrite("PRO/Phase/plateG.jpg",out2)
elif save == 3:
	cv2.imwrite("PRO/Phase/plateB.jpg",out3)
"""
