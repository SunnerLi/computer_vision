import numpy as np
import random
import cv2
import os

def imgRotate(img, degree=random.random()*360):
    height, width = img.shape

    # Change to three channel and render as red
    img_blue = np.zeros([height, width, 3])
    for i in range(height):
        for j in range(width):
            if img[i][j] == 0:
                img_blue[i][j][0] = 100
                img_blue[i][j][1] = 0
                img_blue[i][j][2] = 0
            else:
                img_blue[i][j][0] = 0
                img_blue[i][j][1] = 0
                img_blue[i][j][2] = 0
    
    # Rotate and render the black area as write
    M = cv2.getRotationMatrix2D((width/2, height/2), degree, 1)
    img_blue = cv2.warpAffine(img_blue, M, (width, height))
    for i in range(height):
        for j in range(width):
            if img_blue[i][j][0] == 0:
                img_blue[i][j][0] = 255
                img_blue[i][j][1] = 255
                img_blue[i][j][2] = 255  

    # Change as gray scale
    img = np.zeros([height, width])
    for i in range(height):
        for j in range(width):
            img[i][j] = img_blue[i][j][0]/255
            if not img[i][j] == 1.0:
                img[i][j] = 0
    return img

def reRender(img, threshold=128, margin=30):
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            if img[i][j] >= threshold:
                img[i][j] -= random.random()*margin
            else:
                img[i][j] += random.random()*margin
    return img

def preProcess():
    # Create working 3 folder
    if not os.path.exists('./work/'):
        os.mkdir('./work/')
    if not os.path.exists('./work/pos'):
        os.mkdir('./work/pos')
    if not os.path.exists('./work/neg'):
        os.mkdir('./work/neg')

    # Copy pos images
    POS_IMG_SITE = './img/pos/'
    pos_list = os.listdir(POS_IMG_SITE)
    for name in pos_list:
        img = cv2.imread(POS_IMG_SITE + name, 1)
        cv2.imwrite('./work/pos/' + name, img)
    print "( 1 / 6 ) Finish copy the positive image to work folder"

    # Get other rotate 9 images (*10)
    POS_IMG_SITE = './work/pos/'
    pos_list = os.listdir(POS_IMG_SITE)
    for name in pos_list:
        for i in range(2):
            img = cv2.imread(POS_IMG_SITE + name, 0)
            cv2.imwrite(POS_IMG_SITE + name[:-4] + str(i) + '.png', imgRotate(img))
    print "( 2 / 6 ) Finish rotate images"

    # Get rerender 4 images (*5)
    POS_IMG_SITE = './work/pos/'
    pos_list = os.listdir(POS_IMG_SITE)
    for name in pos_list:
        img = cv2.imread(POS_IMG_SITE + name, 0)
        for i in range(4):
            cv2.imwrite(POS_IMG_SITE + name[:-4] + str(i) + '.png', reRender(img))
    print "( 3 / 6 ) Finish re-render images"
    
    # Copy neg images
    NEG_IMG_SITE = './img/neg/'
    neg_list = os.listdir(NEG_IMG_SITE)
    for name in neg_list:
        img = cv2.imread(NEG_IMG_SITE + name, 1)
        cv2.imwrite('./work/neg/' + name, img)
    print "( 4 / 6 ) Finish copy the negative image to work folder"

    # Get other rotate 9 images (*10)
    NEG_IMG_SITE = './work/neg/'
    neg_list = os.listdir(NEG_IMG_SITE)
    for name in neg_list:
        for i in range(2):
            img = cv2.imread(NEG_IMG_SITE + name, 0)
            cv2.imwrite(NEG_IMG_SITE + name[:-4] + str(i) + '.png', imgRotate(img))
    print "( 5 / 6 ) Finish rotate images"

    # Get rerender 4 images (*5)
    NEG_IMG_SITE = './work/neg/'
    neg_list = os.listdir(NEG_IMG_SITE)
    for name in neg_list:
        img = cv2.imread(NEG_IMG_SITE + name, 0)
        for i in range(4):
            cv2.imwrite(NEG_IMG_SITE + name[:-4] + str(i) + '.png', reRender(img))
    print "( 6 / 6 ) Finish re-render images"

preProcess()
"""
img = cv2.imread('0.png', 0)
cv2.imshow('result', cv2.resize(imgRotate(img), (np.shape(img)[0] * 5, np.shape(img)[1] * 5)))
cv2.waitKey(0)
"""