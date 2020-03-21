# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:19:15 2017

@author: momos
"""

import numpy as np  
from matplotlib import pyplot as plt
array = np.array
plot = plt.plot

def neiPairsOf(idim, trans=-1, shift=None):

    shift = shift or (lambda im,pairs:[np.roll(im,(d,r),axis=(0,1)) for d,r in pairs])
    
    def toNeiPairs(a, b):
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)
        pairs = np.hstack([a,b])
        pairs = pairs[pairs[:,0] != trans]
        pairs = pairs[pairs[:,1] != trans]
        return pairs
    
    neiPairs = [toNeiPairs(idim, neimage) for neimage in shift(idim, [(1,0),(0,1)])]
        
    return np.vstack(neiPairs)

def to_id_image(im, transOld=0, transNew=-1):
    idim = np.arange(im.ravel().shape[0]).reshape(im.shape)
    idim[im==transOld] = transNew
    return idim

def toCoords(ids: object, shape: object) -> object:
    ids = array(ids)
    m,n = shape
    r = ids // n
    c = ids % n
    return np.column_stack([r,c])

def toIDs(coords, shape):
    coords = array(coords)
    m, n = shape
    r, c = coords.T
    if isinstance(r, np.ndarray):
        r[r<0] += m
        c[c<0] += n
    else:
        r = r if r>=0 else r + m
        c = c if c>=0 else c + n
        
    return r * n + c

def plotGroup(ax, im, group):
    ax.imshow(im)
    
    yx = toCoords(list(group), im.shape)
    o = ax.plot(yx[:,1],yx[:,0],'r.')
    ax.axis("off")
#    plt.show()
    return o
    
def shift_neonespadding(arr, disp_pairs):
    h,w = arr.shape
    paded = np.empty((h+2,w+2), dtype=int)
    paded[(0,-1),:] = -1
    paded[:,(0,-1)] = -1
    paded[1:-1,1:-1] = arr
    
    shift = lambda d,r: paded[-d+1:-d+1+h, -r+1:-r+1+w]
    return [shift(d,r) for d,r in disp_pairs]