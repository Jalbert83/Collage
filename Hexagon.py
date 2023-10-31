import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def get_hexagon(im):
    l, _, _ = im.shape
    lm = l//2
    alp = 0.42
    p = []
    for row in range(lm):
        ci = int(np.round((lm-row)*np.tan(alp)))
        cf = l-ci
        for c in np.arange(ci, cf):
            p.append((row,c))
    for row in np.arange(lm,l):
        ci = int(np.round((row-lm)*np.tan(alp)))
        cf = l-ci
        for c in np.arange(ci, cf):
            p.append((row,c))
    return p

def get_hexagon_reg(l):
    #l, _, _ = im.shape
    lm = l//2
    alp = np.pi/6
    ri = int(np.round(lm-np.cos(alp)*(l/(1+2*np.sin(alp)))))
    p = []
    for row in np.arange(ri,lm):
        ci = int(np.round((lm-row)*np.tan(alp)))
        cf = l-ci
        for c in np.arange(ci, cf):
            p.append((row,c))
    for row in np.arange(lm,l-ri):
        ci = int(np.round((row-lm)*np.tan(alp)))
        cf = l-ci
        for c in np.arange(ci, cf):
            p.append((row,c))
    return p
'''
im = np.zeros((101,101,3), dtype=np.uint8)
p = get_hexagon_reg(im)
for r,c in p:
    im[r,c,:] = [255, 255, 255]

plt.imshow(im)
plt.show()'''

