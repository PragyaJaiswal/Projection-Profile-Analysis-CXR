#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt

def extract_features(im, P, X, Y):
    density_symmetry = dsymmetry(P, X)

    def roughness_indices():
        for ind, proj_zone in enumerate(P):
            print('Zone {0}'.format(ind+1))
            top = Y[ind]
            bottom = Y[ind+1]

            # Calculate (Vrib + Vlung)/2 for finding x1, x2, x3, x4
            vxrlung = proj_zone[X[ind][1]]
            vxrrib = proj_zone[X[ind][0]]

            vxllung = proj_zone[X[ind][3]]
            vxlrib = proj_zone[X[ind][4]]

            if vxrlung - vxrrib > vxllung - vxlrib:
                lung_field = (vxrlung + vxrrib)/2
                print(lung_field, 'right')
            else:
                lung_field = (vxllung + vxlrib)/2
                print(lung_field, 'left')

            # Calculate roughness of right and left sides, for each zone.
            print('Calculating roughness for zone {0}.'.format(ind+1))
            summ_RR = 0
            summ_RL = 0
            for i in range(top, bottom+1):
                # roughness(im, i, lung_field, X[ind])
                RR, RL = roughness(im, i, lung_field, X[ind])
                summ_RR += RR
                summ_RL += RL

            print('Finished roughness for zone {0}.'.format(ind+1))
            print(summ_RR, summ_RL, 'Zone {0}'.format(ind+1))
            input('Enter')


    def roughness(im, row, lung_field, positions):
        # horizontal_sum = np.sum(im, axis=1)/np.shape(im)[0]
        # horizontal = im
        horizontal = np.zeros([1, np.shape(im)[1]])
        horizontal = im[row]
        horizontal = np.ndarray.tolist(horizontal)

        # Find x1, x2, x3, x4
        print('Calculating lung field positions - x1, x2, x3, x4.')
        xrrib = positions[0]
        xrlung = positions[1]
        x1_init, val = find_nearest(horizontal[xrrib:xrlung], lung_field)
        x1 = x1_init + xrrib
        # print('x1 - {0}, y - {1}'.format(x1, val))

        xc = positions[2]
        x2_init, val = find_nearest(horizontal[xrlung:xc], lung_field)
        x2 = x2_init + xrlung
        # print('x2 - {0}, y - {1}'.format(x2, val))

        xllung = positions[3]
        x3_init, val = find_nearest(horizontal[xc:xllung], lung_field)
        x3 = x3_init + xc
        # print('x3 - {0}, y - {1}'.format(x3, val))

        xlrib = positions[4]
        x4_init, val = find_nearest(horizontal[xllung:xlrib], lung_field)
        x4 = x4_init + xllung
        # print('x4 - {0}, y - {1}'.format(x4, val))

        print('Finished calculating lung field positions.')
        '''
        fig = plt.figure(0)
        fig.canvas.set_window_title('Horizontal Projection Profile - ' + str(row))
        plt.plot(horizontal)
        plt.show()
        '''

        # Calculate moving average for horizontal profile.
        print('Finding moving average for horizontal profile')
        avg = []
        window = 10
        for i in range(0,np.shape(im)[1]):
            avg.append(moving_average(horizontal, window, row))
        
        # Calculate roughness for right side, for each horizontal profile in zone.
        print('Calculating roughness for right side for horizontal profile.')
        RR = 0
        for i in range(x1,x2):
            RR += abs(horizontal[i] - avg[i])
        RR = RR/(x2-x1+1)

        # Calculate roughness for left side, for horizontal profile in zone.
        print('Calculating roughness for left side for horizontal profile.\n')
        RL = 0
        for i in range(x3,x4):
            RL += abs(horizontal[i] - avg[i])
        RL = RL/(x4-x3+1)

        return RR, RL        
        

    roughness_indices()
    # NRL = roughness_indices()


def find_nearest(horizontal, value):
    ind = (np.abs(np.asarray(horizontal)-value)).argmin()
    return ind, horizontal[ind]


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

if __name__ == '__main__':
    main()