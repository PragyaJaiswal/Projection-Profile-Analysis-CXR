#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

data_dir = '../data/CXR_png/'

def profile_one_dim(im):
	vertical_sum = np.sum(im, axis=0)/np.shape(im)[1]
	plt.plot(vertical_sum)
	plt.show()

if __name__ == '__main__':
	for image in os.listdir(data_dir):
		im = scipy.ndimage.imread(data_dir + image)
		profile_one_dim(im)