import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import cv2 as cv
import math
import matplotlib.pyplot as plt
from resize_img import resizeImg
import random

print('Reading image...')
img1 = cv.imread('img/object1.jpg', cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread('img/scene1.jpg', cv.IMREAD_GRAYSCALE)  # trainImage

# resize 20%
img1 = resizeImg(img1, 20)
img2 = resizeImg(img2, 20)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

scene_points = np.float32([kp.pt for kp in kp2])

# #############################################################################
# Compute clustering with MeanShift

bandwidth = estimate_bandwidth(scene_points, quantile=0.2)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(scene_points)

labels = ms.labels_
cluster_centers = ms.cluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

for k in range(n_clusters_):
    mask_of_cluster_k = labels == k
    cluster_center = cluster_centers[k]
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cv.circle(img2, (cluster_center[0], cluster_center[1]), 10, color, thickness=10)
    for x, y in scene_points[mask_of_cluster_k]:
        cv.circle(img2, (x, y), 5, color)

# #############################################################################
# Matching features and locate object
bf = cv.BFMatcher()
img3 = np.array(img2, copy=True)
MIN_MATCH_COUNT = 100
instance_count = 0
for k in range(n_clusters_):
    print("processing cluster %d ..." % k)
    mask_of_cluster_k = labels == k
    matches = []

    # matching features
    for i in range(0, len(des1)):
        minDistSq1 = minDistSq2 = 99999999
        minIdx1 = minIdx2 = -1
        for j in range(0, len(des2)):
            if not mask_of_cluster_k[j]:
                continue
            diffVec = des2[j] - des1[i]
            distSq = np.sum(np.square(diffVec))
            if distSq < minDistSq1:
                minDistSq2 = minDistSq1
                minDistSq1 = distSq
                minIdx2 = minIdx1
                minIdx1 = j
            elif distSq < minDistSq2:
                minDistSq2 = distSq
                minIdx2 = j

        if minIdx1 > 0 and minIdx2 > 0:
            matches.append([
                cv.DMatch(i, minIdx1, 0, math.sqrt(minDistSq1)),
                cv.DMatch(i, minIdx2, 0, math.sqrt(minDistSq2))
            ])

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=None,  # draw only inliers
                       flags=2)
    imgK = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    fileName = "img/out/img" + str(k) + ".jpg"
    cv.imwrite(fileName, imgK)
    print("matching output: ", fileName)

    # locate object
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        img3 = cv.polylines(img3, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        instance_count += 1
        print("found instance in cluster ", k)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

print("number of instances: ", instance_count)
plt.imshow(img3, 'gray'), plt.show()

if instance_count > 0:
    cv.imwrite("img/out/img.jpg", img3)
    print("detecting output: img/out/img.jpg")