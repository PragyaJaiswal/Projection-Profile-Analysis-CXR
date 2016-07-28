#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from save_features import load

def train_RFC():
    fv = load('features_complete.pkl')
    
    labels = []
    for i in range(len(fv)):
        labels.append(fv[i][-1])

    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(fv, labels)
    

if __name__ == '__main__':
    train_RFC()