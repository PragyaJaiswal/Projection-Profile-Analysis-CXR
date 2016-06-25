#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import operator, collections
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

data_dir = '../data/CXR_png/'

# 0 - black, 255 - white

def profile_one_dim(im):
	im = gray_level(im)
	print(np.shape(im))
	vertical_sum = np.sum(im, axis=0)/np.shape(im)[1]
	fig = plt.figure(0)
	fig.canvas.set_window_title('Projection Profile')
	plt.plot(vertical_sum)
	plt.show()
	zone_division(im, vertical_sum)

def gray_level(im):
	num_of_gray_levels = len(np.unique(im))
	image_bit = math.log(num_of_gray_levels, 2)
	'''
	Initialise a gray_level_hist list with all zeros. Indices
	denote the grap level, value at index denote the count.
	'''
	gray_level_hist = np.zeros(2**image_bit)
	# # VERY SLOW
	# for x in im:
	# 	for y in x:
	# 		gray_level_hist[y]+=1
	# print(gray_level_hist)

	unique, counts = np.unique(im, return_counts=True)
	gray_level_hist_dict = dict(zip(unique, counts))
	background_value = max(gray_level_hist_dict.items(), key=operator.itemgetter(1))[0]
	# print(background_value, gray_level_hist_dict[background_value])
	normalized_im = np.divide(im, background_value)
	return normalized_im

def zone_division(im, vertical_sum):
	low = math.floor(0.25*len(vertical_sum))
	high = math.floor(0.50*len(vertical_sum))
	'''
	mini = min(vertical_sum[low:high])
	ind = list(vertical_sum).index(mini)
	x_right = []
	for x in im:
		# print(x[ind])
		x_right.append(255 - x[ind])
	# print(x_right)
	'''

	'''
	x_right = math.floor(len(vertical_sum)/4)
	print(x_right, 'div')
	vertical_profile(im, x_right)
	'''

	x_right = list(vertical_sum).index(max(vertical_sum[low:high]))
	print(x_right, 'x_right')
	vert_prof = vertical_profile(im, x_right)

	# For ytop
	def ytop():
		low = math.floor(0.05*len(vert_prof))
		high = math.floor(0.50*len(vert_prof))
		ytopv = min(vert_prof[low:high])
		# ytopi = vert_prof.index(ytopv)
		ytopi = np.argmin(np.asarray(vert_prof[low:high])) + low

		print(ytopi, 'y-top index')
		fig = plt.figure(0)
		fig.canvas.set_window_title('Vertical Profile at x_right')
		plt.plot(vert_prof)
		plt.show()
		return ytopi


	# For ybottom
	def ybottom():
		low = math.floor(0.51*len(vert_prof))
		high = math.floor(0.95*len(vert_prof))
		vert_prof_derivative = np.zeros(len(vert_prof))
		'''Calculate derivative using finite difference
			f'(x) = f(x+h) - f(x)/h'''
		h = 20
		for i in range(0, len(vert_prof)-h):
			vert_prof_derivative[i] = (vert_prof[i+h] - vert_prof[i])/h
		ybottomv = min(vert_prof_derivative[low:high])
		# ybottomi = list(vert_prof_derivative[low:high]).index(ybottomv) + low
		ybottomi = np.argmin(np.asarray(vert_prof_derivative[low:high])) + low
		print(ybottomi, 'y-bottom index')

		fig = plt.figure(0)
		fig.canvas.set_window_title('Vertical Profile Derivative at x_right')
		plt.plot(vert_prof_derivative)
		plt.show()

	ytopi = ytop()
	ybottomi = ybottom()

	y1 = ytopi + 0.25*(ybottomi-ytopi)
	y2 = ytopi + 0.5*(ybottomi-ytopi)
	y3 = ytopi + 0.75*(ybottomi-ytopi)
	Y = [ytopi, y1, y2, y3, ybottomi]


def vertical_profile(im, x_right):
	vert_prof = []
	for x in im:
		vert_prof.append(x[x_right])
	return vert_prof

if __name__ == '__main__':
	for image in os.listdir(data_dir):
		im = scipy.ndimage.imread(data_dir + image)
		profile_one_dim(im)
		input('Enter')