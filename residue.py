from sklearn.cluster import MiniBatchKMeans
import mahotas
import argparse
import imutils
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--clusters", required = True, type = int, help = "# of clusters")
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

image = cv2.imread(args["image"])
output = image.copy()

(h, w) = image.shape[:2]
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
image = image.reshape((image.shape[0] * image.shape[1], 3))
clt = MiniBatchKMeans(n_clusters = args["clusters"])
labels = clt.fit_predict(image)
quant = clt.cluster_centers_.astype("uint8")[labels]
quant = quant.reshape((h, w, 3))
image = image.reshape((h, w, 3))
quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
image1 = quant.copy()

""" 
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255) 

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] #if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None

for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 	if len(approx) == 4:
		displayCnt = approx
		break

warped = imutils.four_point_transform(gray, displayCnt.reshape(4, 2))
output = imutils.four_point_transform(image, displayCnt.reshape(4, 2))	
"""
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
T = mahotas.thresholding.otsu(blurred)
blurred[blurred > T] = 255

edged = cv2.Canny(blurred, 50, 200, 255) 

thresh = cv2.threshold(edged, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cv2.imshow("Output", thresh)
cv2.waitKey(0)

#"""
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] 
digitCnts = []
 
for c in cnts:
	(x, y, w, h) = cv2.boundingRect(c)
	#if w >= 15 and h >= 20 and h <= 130:
	digitCnts.append(c)

digitCnts = imutils.sort_contours(digitCnts, method="left-to-right")[0]
digits = []			
for c in digitCnts:
	(x, y, w, h) = cv2.boundingRect(c)
	roi = thresh[y:y + h, x:x + w]
	(roiH, roiW) = roi.shape
	(dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
	dHC = int(roiH * 0.05)
	segments = [
		((0, 0), (w, dH)),	# top
		((0, 0), (dW, h // 2)),	# top-left
		((w - dW, 0), (w, h // 2)),	# top-right
		((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
		((0, h // 2), (dW, h)),	# bottom-left
		((w - dW, h // 2), (w, h)),	# bottom-right
		((0, h - dH), (w, h))	# bottom
	]
	on = [0] * len(segments)
	for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
		segROI = roi[yA:yB, xA:xB]
		total = cv2.countNonZero(segROI)
		area = (xB - xA) * (yB - yA)
 		if total / float(area) > 0.5:
			on[i]= 1
	digit = DIGITS_LOOKUP[tuple(on)]
	digits.append(digit)
	cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
	cv2.putText(output, str(digit), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

#print(u"{}{}.{} \u00b0C".format(*digits))
#cv2.imshow("Input", image)
#cv2.imshow("Output", output)
#cv2.waitKey(0)
#"""
