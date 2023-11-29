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

def get_hexagon_mask_coordinates(im_size, mask_size_x, mask_size_y):
    alp = np.pi/6
    lhex = int(np.round(im_size/(1+2*np.sin(alp))))
    ri = int(np.round(im_size//2-np.cos(alp)*(im_size/(1+2*np.sin(alp)))))
    lres = int(np.round(lhex*np.sin(alp)))

    x_range1 = np.arange(0,mask_size_x,im_size+lhex)
    y_range1 = np.arange(0,mask_size_y,im_size-2*ri)
    coords1 = np.array([[x,y] for x in x_range1 for y in y_range1])
    x_range2 = np.arange(lhex+lres,(mask_size_x//(im_size+lhex))*(im_size+lhex)+lhex+lres,im_size+lhex)
    y_range2 = np.arange(im_size//2-ri,(mask_size_y//(im_size-2*ri))*(im_size-2*ri),im_size-2*ri)
    coords2 = np.array([[x,y] for x in x_range2 for y in y_range2])
    coords = np.concatenate((coords1, coords2), axis=0)
    return coords

'''
im = np.zeros((101,101,3), dtype=np.uint8)
p = get_hexagon_reg(im)
for r,c in p:
    im[r,c,:] = [255, 255, 255]

plt.imshow(im)
plt.show()'''

