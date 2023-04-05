import numpy as np
import matplotlib.pyplot as plt

def fja(kvadrat,redak,stupac):
    crna = np.zeros((kvadrat,kvadrat))
    bijela = np.ones((kvadrat,kvadrat))*255
    redak1 = np.hstack([crna,bijela]*(stupac//2))
    if stupac%2 != 0:
        redak1 = np.hstack([redak1,crna])
    
    redak2 = np.hstack([bijela,crna]*(stupac//2))
    if stupac%2 != 0:
        redak2 = np.hstack([redak2, bijela])

    matrix = np.vstack([redak1,redak2] * (redak//2))
    if redak % 2 != 0:
        matrix = np.vstack([matrix,redak1])
    return matrix

img = fja(50,4,5)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()
