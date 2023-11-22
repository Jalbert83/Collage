import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.linalg import norm
from hexagon import get_hexagon_reg


def BrightSelect(MaskSum, df, range, var):
    Amp = range[1]-range[0]-var
    cenTb = MaskSum*Amp+range[0]+var//2
    c = df.loc[df.bright > cenTb-var//2-2].loc[df.bright <= cenTb+var//2+2]

    return c


imsize = 128
MaskSizeX = 8000#int(np.floor(20*imsize))
MaskSizeY = 10666#int(np.floor(30*imsize))
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


BrightRange = [np.min(df.bright), np.max(df.bright)]
BrightVar = 40

mask = cv2.imread('mask.jpg')
mask = cv2.resize(mask, (MaskSizeX, MaskSizeY))
res = np.zeros(mask.shape, dtype=np.uint8)

#_, mask = cv2.threshold(mask, 127,1,cv2.THRESH_BINARY)

alp = np.pi/6
lhex = int(np.round(imsize/(1+2*np.sin(alp))))
ri = int(np.round(imsize//2-np.cos(alp)*(imsize/(1+2*np.sin(alp)))))
lres = int(np.round(lhex*np.sin(alp)))
HexReg = get_hexagon_reg(imsize)

Xrange1 = np.arange(0,MaskSizeX,imsize+lhex)
Yrange1 = np.arange(0,MaskSizeY,imsize-2*ri)
Coords1 = np.array([[x,y] for x in Xrange1 for y in Yrange1])
Xrange2 = np.arange(lhex+lres,(MaskSizeX//(imsize+lhex))*(imsize+lhex)+lhex+lres,imsize+lhex)
Yrange2 = np.arange(imsize//2-ri,(MaskSizeY//(imsize-2*ri))*(imsize-2*ri),imsize-2*ri)
Coords2 = np.array([[x,y] for x in Xrange2 for y in Yrange2])
Coords = np.concatenate((Coords1, Coords2), axis=0)

print(str(i) + ' Source images,' + str(len(Coords)) + ' Mask spots')

Maskdf = pd.DataFrame()


for enu, (x,y) in enumerate(Coords):
    Maskb = np.sum(mask[y:y+imsize,x:x+imsize,:])/(3*imsize**2*255)
    c = BrightSelect(Maskb, df, BrightRange, BrightVar)
    if(len(c) == 0):
        print ('zero results')
    e = c.loc[c.used == np.min(c.used)]
    e = e.iloc[np.random.randint(len(e))]
    im = cv2.imread(e.path)
    imr = cv2.resize(im,(imsize, imsize))
    for r, c in HexReg:
        res[min(y+r, MaskSizeY-1),min(x+c,MaskSizeX-1),::-1] = imr[r,c,:]
    a = 0
    df.loc[df.path == e.path, 'used'] += 1
    Maskdf.loc[enu, 'bright'] = Maskb




c = df.loc[df.used == 0]
print(str(len(c)))
plt.figure()
c.hist("bright")
df.hist('bright')
Maskdf.hist('bright')
plt.figure()
cv2.imwrite("result.jpg", res[:,:,::-1])
plt.imshow(res)

plt.figure()
plt.imshow(mask*255)
plt.show()

