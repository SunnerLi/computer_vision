import cv2

img = cv2.imread('fast_true.png', 1)
cv2.imshow('result', cv2.resize(img, (img.shape[0]*2, img.shape[1]*2)))
cv2.waitKey(0)