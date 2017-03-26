import numpy as np
import random
import cv2

def imgRotate(img, degree=random.random()*360):
    height, width = img.shape

    # Change to three channel and render as red
    img_blue = np.zeros([height, width, 3])
    for i in range(height):
        for j in range(width):
            img_blue[i][j][0] = float(img[i][j])/255
            img_blue[i][j][1] = 0
            img_blue[i][j][2] = 0

    # Rotate and render the black area as write
    M = cv2.getRotationMatrix2D((width/2, height/2), degree, 1)
    img_blue = cv2.warpAffine(img_blue, M, (width, height))
    for i in range(height):
        for j in range(width):
            if img_blue[i][j][0] == 0:
                img_blue[i][j][0] = 1
                img_blue[i][j][1] = 1
                img_blue[i][j][2] = 1

    # Change as gray scale
    img = np.zeros([height, width])
    for i in range(height):
        for j in range(width):
            img[i][j] = img_blue[i][j][0]
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

img = cv2.imread('lana.jpg', 0)
cv2.imshow('result', reRender(img))
cv2.waitKey(0)