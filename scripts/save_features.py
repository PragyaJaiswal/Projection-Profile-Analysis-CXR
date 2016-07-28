#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import json, pickle

out_dir = '../sample_output/pickles/'

def feature_vector(density_symmetry, roughness_max, roughness_symmetry, filename):
	# print(density_symmetry, roughness_max, roughness_symmetry)
	arr = np.append(density_symmetry, roughness_max, axis = 0)
	arr = np.append(arr, roughness_symmetry, axis = 0)
	label = filename[-5]
	arr = np.append(arr, label)
	return arr

def label_vec(all_vector):
	labels = (np.asarray(all_vector))[:, [12]]
	print(len(labels))
	dump(labels, 'labels.pkl')

def dump(all_vector, out_filename):
	with open(out_dir + out_filename, 'wb') as outfile:
		pickle.dump(all_vector, outfile)

def load(filename):
	# print(pickle.load(open('features.pkl', 'rb')))
	feat_vector = np.load(out_dir + filename)
	feat_vector = cleaned(feat_vector)
	return feat_vector

def cleaned(feat_vector):
	pos = []
	neg = []
	for i in range(len(feat_vector)):
		label = feat_vector[i][-1]
		if label == '1':
			pos.append(feat_vector[i])
		else:
			neg.append(feat_vector[i])
	print(len(pos), len(neg))
	neg = replace_nan_or_inf(neg)
	pos = replace_nan_or_inf(pos)
	
	fv = np.concatenate((neg, pos), axis=0)
	print(np.shape(fv))
	return fv

def replace_nan_or_inf(arr):
	fine = []
	nanorinf = []
	for i in range(0, 12):
		for j in range(0, len(arr)):
			if np.isnan(float(arr[j][i])) or np.isinf(float(arr[j][i])):
				nanorinf.append(j)
			else:
				fine.append(float(arr[j][i]))
		val = np.median(fine)
		
		for ind in nanorinf:
			arr[ind][i] = val

		fine = []
		nanorinf = []

	return arr

