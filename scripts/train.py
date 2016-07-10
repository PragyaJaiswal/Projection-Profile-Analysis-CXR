#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from save_features import load

def train_RFC():
    fv = load('features.pkl')
    labels = np.ravel(load('labels.pkl'))
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(fv, labels)

if __name__ == '__main__':
    train_RFC()