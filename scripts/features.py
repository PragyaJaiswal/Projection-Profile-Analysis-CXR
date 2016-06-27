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

def roughness(im):
	# horizontal_sum = np.sum(im, axis=1)/np.shape(im)[0]
	pass

if __name__ == '__main__':
	main()