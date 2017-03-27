import numpy as np
import cv2

template = [
    [[-4, 5, 5], [-4, 5, 5], [-4, -4, -4]],
    [[5, 5, -4], [5, 5, -4], [-4, -4, -4]],
    [[-4, -4, -4], [5, 5, -4], [5, 5, -4]],
    [[-4, -4, -4], [-4, 5, 5], [-4, 5, 5]],
    [[5, 5, 5], [-4, 5, -4], [-4, -4, -4]],
    [[-4, -4, 5], [-4, 5, 5], [-4, -4, 5]],
    [[-4, -4, -4], [-4, 5, -4], [5, 5, 5]],
    [[5, -4, -4], [5, 5, -4], [5, -4, -4]]
] 

THRESHOLD = 5000

def templateMatching(img):
    # Get the size of the image
    img_height = np.shape(img)[0]
    img_width = np.shape(img)[1]

    # Transfer the result as three channel
    result_show = np.ndarray([img_height, img_width, 3])
    for i in range(img_height):
        for j in range(img_width):
            result_show[i][j] = img[i][j]

    template_num = np.shape(template)[0]
    for k in range(template_num):
        # Get the size of the mask
        mask_height = np.shape(template[k])[0]
        mask_width = np.shape(template[k])[1]

        # Get the stride of the mask
        slide_y = img_height - mask_height + 1
        slide_x = img_width - mask_width + 1

        # Do the template matching
        result = np.zeros([slide_y, slide_x])
        for i in range(slide_y):
            for j in range(slide_x):
                roi = img[i:i+mask_height, j:j+mask_width]
                result[i][j] = np.sum(roi * template[k])

        # Do the binary thresholding
        for i in range(np.shape(result)[0]):
            for j in range(np.shape(result)[1]):
                if result[i][j] >= THRESHOLD:
                    result[i][j] = 1
                else:
                    result[i][j] = 0

        # Graw the result
        for i in range(np.shape(result)[0]):
            for j in range(np.shape(result)[1]):
                if result[i][j] == 1:
                    #result_show[i+1][j+1]=[255, 0, 0]
                    cv2.rectangle(result_show, (j, i), (j+2, i+2), (255, 0, 0), 1)

        print "Finish ", k+1, "template matching..."
    return result_show