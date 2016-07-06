#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import json, pickle

def feature_vector(density_symmetry, roughness_max, roughness_symmetry, filename):
	# print(density_symmetry, roughness_max, roughness_symmetry)
	arr = np.append(density_symmetry, roughness_max, axis = 0)
	arr = np.append(arr, roughness_symmetry, axis = 0)
	arr = np.append(arr, str(filename))
	return arr

def dump(all_vector):
	with open('features.pkl', 'wb') as outfile:
		pickle.dump(all_vector, outfile)

def load():
	# print(pickle.load(open('features.pkl', 'rb')))
	vector = np.load('features.pkl')
	print(vector)