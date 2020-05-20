# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('genome1.jpg')


img2 = img.reshape((-1,3))  #-1 reshape means, in this case MxN

#We convert the unit8 values to float as it is a requirement of the k-means method of OpenCV
img2 = np.float32(img2)

#Define criteria, number of clusters and apply k-means

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Number of clusters
k = 3

attempts = 10

#Two options, cv.KMEANS_PP_CENTERS and cv.KMEANS_RANDOM_CENTERS

ret,label,center=cv2.kmeans(img2, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)


center = np.uint8(center) 

#Next, we have to access the labels to regenerate the clustered image
res = center[label.flatten()]
res2 = res.reshape((img.shape)) #Reshape labels to the size of original image
cv2.imwrite("/home/nikhil/Downloads/NeuralNetwork/K Means Clustering/segmented.jpg", res2)


figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),
plt.imshow(img2, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(res2)
plt.title('Segmented Image when K = %i' % k), plt.xticks([]), plt.yticks([])
plt.show()
