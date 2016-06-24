#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

data_dir = '../data/CXR_png/'

def profile_one_dim(im):
	gray_level(im)
	vertical_sum = np.sum(im, axis=0)/np.shape(im)[1]
	plt.plot(vertical_sum)
	plt.show()

def gray_level(im):
	num_of_gray_levels = len(np.unique(im))
	image_bit = math.log(num_of_gray_levels, 2)
	gray_level_hist = np.zeros(2**image_bit)
	print(np.shape(im))
	for x in im:
		for y in x:
			gray_level_hist[y]+=1
	print(gray_level_hist)


if __name__ == '__main__':
	for image in os.listdir(data_dir):
		im = scipy.ndimage.imread(data_dir + image)
		profile_one_dim(im)
		input('Enter')