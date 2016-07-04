#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import numpy as np

def feature_vector(density_symmetry, roughness_max, roughness_symmetry, filename, vector):
	# val = hashlin.md5(filename)
	# print(val, filename)
	print(density_symmetry, roughness_max, roughness_symmetry)
	arr = np.append(density_symmetry, roughness_max, axis = 0)
	arr = np.append(arr, roughness_symmetry, axis = 0)
	arr = np.append(arr, str(filename))
	print(arr)
	return arr