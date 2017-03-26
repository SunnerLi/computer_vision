import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('simple.png',0)
img2 = img

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, img2, color=(255,0,0))
cv2.imwrite('fast_true.png',img2)