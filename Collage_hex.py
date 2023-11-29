import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.linalg import norm
from hexagon import get_hexagon_reg, get_hexagon_mask_coordinates

'''
Returns a selection of images (within the data frame) filtered by brightness taking into account the mask brightness
'''
def bright_select(mask_sum, df, range, tolerance):
    amp = range[1]-range[0]-tolerance
    center = mask_sum*amp+range[0]+tolerance//2
    selection = df.loc[df.bright > center-tolerance//2-2].loc[df.bright <= center+tolerance//2+2]
    return selection


im_size = 128
mask_size_x = int(np.floor(20*im_size))#8000
mask_size_y = int(np.floor(30*im_size))#10666
dir_path = r'.\croped_images'
df = pd.DataFrame()
i= 0

for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        im = cv2.imread(os.path.join(dir_path, path))


        bright = np.average(norm(im, axis=2)) / np.sqrt(3)
        df.loc[i, 'path'] = os.path.join(dir_path, path)
        df.loc[i, 'bright'] = bright
        df.loc[i, 'used'] = 0
        i+=1



bright_range = [np.min(df.bright), np.max(df.bright)]
bright_tolerance = 40

mask = cv2.imread('mask.jpg')
mask = cv2.resize(mask, (mask_size_x, mask_size_y))
result = np.zeros(mask.shape, dtype=np.uint8)

hex_region = get_hexagon_reg(im_size)
coords = get_hexagon_mask_coordinates(im_size, mask_size_x, mask_size_y)

print(str(i) + ' Source images,' + str(len(coords)) + ' Mask spots')

mask_df = pd.DataFrame()


for i_spot, (x,y) in enumerate(coords):
    mask_region_bright = np.sum(mask[y:y+im_size,x:x+im_size,:])/(3*im_size**2*255)
    selected = bright_select(mask_region_bright, df, bright_range, bright_tolerance)
    if(len(selected) == 0):
        print ('zero results')
    selected_min_used = selected.loc[selected.used == np.min(selected.used)]
    selected = selected_min_used.iloc[np.random.randint(len(selected_min_used))]
    im = cv2.imread(selected.path)
    imr = cv2.resize(im,(im_size, im_size))
    for r, c in hex_region:
        result[min(y+r, mask_size_y-1),min(x+c,mask_size_x-1),::-1] = imr[r,c,:]
    df.loc[df.path == selected.path, 'used'] += 1
    mask_df.loc[i_spot, 'bright'] = mask_region_bright




c = df.loc[df.used == 0]
print(str(len(c)))
plt.figure()
c.hist("bright")
df.hist('bright')
mask_df.hist('bright')
plt.figure()
cv2.imwrite("result.jpg", result[:,:,::-1])
plt.imshow(result)

plt.figure()
plt.imshow(mask*255)
plt.show()

