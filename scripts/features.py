#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

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

def roughness(im, row):
	# horizontal_sum = np.sum(im, axis=1)/np.shape(im)[0]
	# horizontal = im
	horizontal = np.zeros([1, np.shape(im)[1]])
	horizontal = im[row]
	horizontal = np.ndarray.tolist(horizontal)
	print(len(horizontal))
	# for i in range(0, np.shape(im)[1]):
	# 	horizontal[i] = im[row][i]

	avg = np.zeros([1, np.shape(im)[1]])
	avg = []
	window = 10
	avg.append(moving_average(horizontal, window, row))


def moving_average(horizontal, window, row):
	low = row - window/2
	if low < 0:
		window = window + low
		low = 0

	high = row + window/2
	if high > len(horizontal):
		window = window + high - len(horizontal)
		high  = len(horizontal)

	for i in range(low, high+1):
		summ = summ + horizontal[i]

	avg = summ/(window+1)
	return avg



if __name__ == '__main__':
	main()