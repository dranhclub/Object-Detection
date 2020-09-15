import numpy as np
import cv2 as cv
import time
import math
import matplotlib.pyplot as plt
from resize_img import resizeImg


print('Reading image...')
img1 = cv.imread('img/object0.jpg', cv.IMREAD_GRAYSCALE)     # queryImage
img2 = cv.imread('img/scene0.jpg', cv.IMREAD_GRAYSCALE)       # trainImage

# resize 20%
img1 = resizeImg(img1, 20)
img2 = resizeImg(img2, 20)


# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
print('KeyPoint matching...')
t = time.time()
bf = cv.BFMatcher()
matches = []

matches = bf.knnMatch(des1, des2, 2)

# for i in range(0, len(des1)):
#     minDistSq1 = minDistSq2 = 99999999
#     minIdx1 = minIdx2 = -1
#     for j in range(0, len(des2)):
#         diffVec = des2[j] - des1[i]
#         distSq = np.sum(np.square(diffVec))
#         if distSq < minDistSq1:
#             minDistSq2 = minDistSq1
#             minDistSq1 = distSq
#             minIdx2 = minIdx1
#             minIdx1 = j
#         elif distSq < minDistSq2:
#             minDistSq2 = distSq
#             minIdx2 = j
#
#     matches.append([
#         cv.DMatch(i, minIdx1, 0, math.sqrt(minDistSq1)),
#         cv.DMatch(i, minIdx2, 0, math.sqrt(minDistSq2))
#     ])

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

time = time.time() - t
print('({}s)'.format(time))
print('Done')

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()

