from template import *
import numpy as np
import cv2
import nn

def FAST(img):
    # Initiate FAST object with default values
    img2 = img
    fast = cv2.FastFeatureDetector_create()

    # find and draw the keypoints
    kp = fast.detect(img,None)
    img2 = cv2.drawKeypoints(img, kp, img2, color=(255,0,0))
    cv2.imwrite('fast_true.png',img2)
    return cv2.imread('fast_true.png', 1)

def enlarge(img, times=4):
    print img.shape
    height, width, channel = img.shape

    # Copy the image first
    img_copy = np.ones([height, width, channel])
    for i in range(height):
        for j in range(width):
            img_copy[i][j] = img[i][j]
    
    result = np.ones([height*times, width*times, channel])
    for i in range(height*times):
        for j in range(width*times):
            result[i][j] = img_copy[i/times][j/times]
    return result
    

if __name__ == "__main__":
    img = cv2.imread('window.jpg', 0)
    #cv2.imwrite('result.png', templateMatching(img))

    # template matching
    img_show = templateMatching(img)
    cv2.imshow('simple template matching', img_show)  
    cv2.waitKey(0)
    cv2.imwrite('template_result.jpg', img_show)

    # FAST algorithm
    cv2.imshow('FAST algorithm', FAST(img))
    cv2.waitKey(0)

    # CNN recognition
    tensor = nn.formTensor(img)
    result_list = nn.work(tensor)
    nn.draw(result_list, img)
    cv2.imshow('CNN recognition', cv2.imread('draw_result.png'))
    cv2.waitKey(0)