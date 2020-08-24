import numpy as np
import cv2 as cv
import time
import math
import matplotlib.pyplot as plt

def resizeImg(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)

print('Reading image...')
img1 = cv.imread('img/theguixe.jpg', cv.IMREAD_GRAYSCALE)     # queryImage
img2 = cv.imread('img/caiban.jpg', cv.IMREAD_GRAYSCALE)       # trainImage

# resize 30%
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

for i in range(0, len(des1)):
    minDist1 = minDist2 = 999999
    minIdx1 = minIdx2 = -1

    for j in range(0, len(des2)):
        dist = np.linalg.norm(des1[i] - des2[j])
        if dist < minDist1:
            minDist2 = minDist1
            minDist1 = dist
            minIdx2 = minIdx1
            minIdx1 = j
        elif dist < minDist2:
            minDist2 = dist
            minIdx2 = j

    matches.append([
        cv.DMatch(i, minIdx1, 0, minDist1),
        cv.DMatch(i, minIdx2, 0, minDist2)
    ])

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