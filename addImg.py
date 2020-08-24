import cv2 as cv
import numpy as np


def addImg(img1, img2):
    i1 = img1.astype('float')
    i2 = img2.astype('float')

    img3 = np.add(i1, i2)
    img3[img3 > 255] = 255
    return img3.astype('uint8')

img1 = cv.imread("img/img1.jpg")
img2 = cv.imread("img/img2.jpg")

img3 = addImg(img1, img2)
cv.imwrite("img/img3.jpg", img3)
cv.imshow("Adding Image", img3)
cv.waitKey(0)

# img3 = np.zeros((height,width,3), np.uint8)
# for i in range(0, height):
#     for j in range(0, width):
#         for k in range(0, 3):
#             t = int(img1[i][j][k]) + int(img2[i][j][k])
#             if t > 255:
#                 t = 255
#             img3[i][j][k] = t
