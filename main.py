from template import *
import numpy as np
import cv2

def FAST(img):
    # Initiate FAST object with default values
    img2 = img
    fast = cv2.FastFeatureDetector_create()

    # find and draw the keypoints
    kp = fast.detect(img,None)
    img2 = cv2.drawKeypoints(img, kp, img2, color=(255,0,0))
    return img2

if __name__ == "__main__":
    img = cv2.imread('simple.png', 0)
    #cv2.imwrite('result.png', templateMatching(img))
    cv2.imshow('simple template matching', templateMatching(img))
    cv2.waitKey(0)
    cv2.imshow('FAST algorithm', FAST(img))
    cv2.waitKey(0)