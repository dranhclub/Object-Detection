import numpy as np
import cv2 as cv
import time
import math
import matplotlib.pyplot as plt
from resize_img import resizeImg


print('Reading image...')
img1 = cv.imread('img/object1.jpg', cv.IMREAD_GRAYSCALE)     # queryImage
img2 = cv.imread('img/scene1.jpg', cv.IMREAD_GRAYSCALE)       # trainImage

# resize 20%
img1 = resizeImg(img1, 20)
img2 = resizeImg(img2, 20)


sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

print('KeyPoint matching...')
t = time.time()
bf = cv.BFMatcher()
matches = []

matches = bf.knnMatch(des1, des2, 2);

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
        good.append(m)

time = time.time() - t
print('({}s)'.format(time))
print('Done')

matchesMask = None

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h, w = img1.shape
    pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1, 0]]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts, M)
    img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None



draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = None, # draw only inliers
                   flags = 2)
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()