from scipy.spatial import distance as dist
import numpy as np
import cv2


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def translate(image, x, y):
  M = np.float32([[1,0,x],[0,1,y]])
  shifted = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
  return shifted 

def rotate(image, angle, center=None, scale=1.0):
  (h,w) = image.shape[:2]
  if center is None:
    center = (w/2, h/2)
  M = cv2.getRotationMatrix2D(center, angle, scale)
  rotated = cv2.warpAffine(image, M, (w,h))
  return rotated	  	

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
  dim = None
  (h,w) = image.shape[:2]
  if width is None and height is None:
    return image
  if width is None:
   	r = height/float(h)
   	dim = (int(w*r),height)
  else:
   	r = width/float(w)
   	dim = (width, int(h*r))
  resized = cv2.resize(image, dim, interpolation=inter)
  return resized			

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes

def label_contour(image, c, i, color=(0, 255, 0), thickness=2):
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.drawContours(image, [c], -1, color, thickness)
    cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return image  