import scipy

XA = [
    [1,2],
    [3,4]
]

XB = [
    [1,1],
    [1,1]
]

Y = scipy.spatial.distance.cdist(XA, XB, 'euclidean')

print(Y)