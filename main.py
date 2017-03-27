from template import *
import numpy as np
import argparse
import time
import cv2
import nn

# Deal with argument
IMG_NAME = 'simple.png'
parser = argparse.ArgumentParser()
parser.add_argument('-n', action='store', dest='img_name_input',
                    help='Key in the image name')
result = parser.parse_args()
if not result.img_name_input == None:
    IMG_NAME = result.img_name_input

def FAST(img):
    # Initiate FAST object with default values
    img2 = img
    fast = cv2.FastFeatureDetector_create()

    # find and draw the keypoints
    kp = fast.detect(img,None)
    img2 = cv2.drawKeypoints(img, kp, img2, color=(0,0,255))
    cv2.imwrite('fast_true.png',img2)
    return cv2.imread('fast_true.png', 1)

def enlarge(img, times=4):
    print img.shape
    if len(img.shape) == 2:
        height, width = img.shape
        channel = 3
    else:
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

def show(title, img):
    if img.shape[0] <= 100:
        cv2.imshow(title, enlarge(img))
    else:
        cv2.imshow(title, img)
    cv2.waitKey(0)
    

if __name__ == "__main__":
    img = cv2.imread(IMG_NAME, cv2.IMREAD_GRAYSCALE)
    show('initial', img)

    # template matching
    start_time = time.time()
    img_show = templateMatching(img)
    template_matching_time = time.time() - start_time
    show('simple template matching', img_show)  
    cv2.imwrite('template_result.jpg', img_show)

    # FAST algorithm
    start_time = time.time()
    show('FAST algorithm', FAST(img))
    FAST_time = time.time() - start_time

    # CNN recognition
    start_time = time.time()
    tensor = nn.formTensor(img)
    result_list = nn.work(tensor)
    nn.draw(result_list, img)
    CNN_time = time.time() - start_time

    print "\n\n-----------------------"
    print " Time (sec)"
    print "-----------------------"
    print "template matching: ", template_matching_time
    print "FAST             : ", FAST_time
    print "CNN              : ", CNN_time   

    show('CNN recognition', cv2.imread('draw_result.png'))