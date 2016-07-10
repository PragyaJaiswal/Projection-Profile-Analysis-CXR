#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import json, pickle

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
	with open(out_filename, 'wb') as outfile:
		pickle.dump(all_vector, outfile)

def load(filename):
	# print(pickle.load(open('features.pkl', 'rb')))
	feat_vector = np.load(filename)
	print(len(feat_vector))