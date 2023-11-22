import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.linalg import norm

while(True):
    imsize = 128
    MaskSizeX = int(np.floor(20*imsize))
    MaskSizeY = int(np.floor(30*imsize))
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
            #plt.imshow(im)
            #plt.show()

    mask = cv2.imread('mask.jpg')
    mask = cv2.resize(mask, (MaskSizeX, MaskSizeY))
    res = np.zeros(mask.shape, dtype=np.uint8)
    _, mask = cv2.threshold(mask, 127,1,cv2.THRESH_BINARY)



    for x in range(0,MaskSizeX,imsize):
        for y in range(0,MaskSizeY,imsize):
            Maskb = np.sum(mask[y:y+imsize,x:x+imsize,:])/(3*imsize**2)
            if(Maskb>0.7):
                c = df.loc[df.bright > 100]
    #        elif(Maskb>0.5):
    #            c = df.loc[df.bright > 120].loc[df.bright <= 150]
            elif(Maskb>0.1):
                c = df.loc[df.bright > 75].loc[df.bright <= 100]
            else:
                c = df.loc[df.bright <= 75]
            used = 0
            e = c.loc[df.used <= used]
            while(len(e) == 0):
                used += 1
                e = c.loc[df.used <= used]
            e = e.iloc[np.random.randint(len(e))]
            im = cv2.imread(e.path)
            res[y:y+imsize,x:x+imsize,::-1] = cv2.resize(im,(imsize, imsize))
            a = 0
            df.loc[df.path == e.path, 'used'] += 1
    c = df.loc[df.used == 0]
    print(str(len(c)))
    plt.figure()
    c.hist("bright")
    plt.figure()
    cv2.imwrite("res10.jpg", res[:,:,::-1])
    plt.imshow(res)
    plt.show()
