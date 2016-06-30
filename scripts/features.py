#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt

def extract_features(im, P, X, Y):
    density_symmetry = dsymmetry(P, X)

    def roughness_indices():
        for ind, proj_zone in enumerate(P):
            top = Y[ind]
            bottom = Y[ind+1]

            # Calculate (Vrib + Vlung)/2 for finding x1, x2, x3, x4
            vxrlung = proj_zone[X[ind][1]]
            vxrrib = proj_zone[X[ind][0]]

            vxllung = proj_zone[X[ind][3]]
            vxlrib = proj_zone[X[ind][4]]

            if vxrlung - vxrrib > vxllung - vxlrib:
                lung_field = (vxrlung + vxrrib)/2
            else:
                lung_field = (vxllung + vxlrib)/2

            RR = 0
            for i in range(top, bottom+1):
                # RR = RR + roughness(im, i)
                roughness(im, i, lung_field)

    def roughness(im, row, lung_field):
        # horizontal_sum = np.sum(im, axis=1)/np.shape(im)[0]
        # horizontal = im
        horizontal = np.zeros([1, np.shape(im)[1]])
        horizontal = im[row]
        horizontal = np.ndarray.tolist(horizontal)
        fig = plt.figure(0)
        fig.canvas.set_window_title('Horizontal Projection Profile - ' + str(row))
        plt.plot(horizontal)
        plt.show()

        # Find x1, x2, x3, x4
        print(find_nearest(horizontal, lung_field))
        # Does not return the desired index, as of now. Test fails.
        # Something is wrong with the horizontal projection profile.

        avg = np.zeros([1, np.shape(im)[1]])
        avg = []
        window = 10
        avg.append(moving_average(horizontal, window, row))


    NRR = roughness_indices()
    NRl = roughness_indices()


def find_nearest(horizontal, value):
    ind = (np.abs(np.asarray(horizontal)-value)).argmin()
    print(ind)
    return horizontal[ind]

def dsymmetry(P, X):
    '''
    dsymmetry  = [dsymmetry_zone1
                    .
                    .
                  dsymmetry_zone4]
    '''
    dsymmetry = np.zeros(4)
    for ind, zone in enumerate(P):
        vxrlung = zone[X[ind][1]]
        vxllung = zone[X[ind][3]]
        num = vxrlung - vxllung / max(vxrlung, vxllung)
        dsymmetry[ind] = num
    return dsymmetry

def moving_average(horizontal, window, row):
    low = math.floor(row - window/2)
    if low < 0:
        window = window + low
        low = 0

    high = math.floor(row + window/2)
    if high > len(horizontal):
        window = window + high - len(horizontal)
        high  = len(horizontal)

    summ = 0
    for i in range(low, high+1):
        summ = summ + horizontal[i]

    avg = summ/(window+1)
    return avg



if __name__ == '__main__':
    main()