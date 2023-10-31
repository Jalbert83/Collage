import cv2
import numpy as np
import matplotlib.pyplot as plt

kernel = np.ones((5,5), np.uint8)

img = cv2.imread('MaskAitor.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (62, 83))
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

'''
res = np.zeros(mask.shape, dtype=np.uint8)
r ,maskbin = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)
#_, maskbin = cv2.threshold(mask, 127,1,cv2.THRESH_BINARY)
maskdil = cv2.erode(maskbin, kernel)
masknp = np.asarray(maskbin)

plt.figure()
plt.imshow(maskbin, cmap='Greys')
plt.figure()
plt.imshow(maskdil, cmap='Greys')
plt.show()
cv2.imwrite("maskAitordil4_5.jpg", maskdil)

'''
