#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from save_features import load

def train_RFC():
    fv = load('features_montg_complete.pkl')
    a = [[] for i in range(len(fv))]
    for row_num, row in enumerate(fv):
        a[row_num] = fv[row_num][0:12]

    print(np.shape(a))

    # # 85% of 662 = 562.7
    # train_set = a[0:562]
    # test_set = a[563:]
    
    # 80% of 662 = 530
    train_set = a[0:530]
    test_set = a[531:]
    '''
    # 85% of 530 = 450
    train_set = a[0:450]
    test_set = a[451:]

    # 80% of 530 = 424
    train_set = a[0:424]
    test_set = a[425:]
    '''
    
    # Montogomery Set
    # 80% of 138 = 110
    # train_set = a[0:117]
    # test_set = a[118:]

    labels = []
    for i in range(len(fv)):
        labels.append(fv[i][-1])
    labels_train_set = labels[0:530]
    labels_test_set = labels[531:]

    clf = RandomForestClassifier(n_estimators=1000)
    clf = clf.fit(train_set, labels_train_set)

    test_set_results = []
    count = 0
    for ind, each in enumerate(test_set):
        predicted_label = clf.predict(each)
        
        if predicted_label == labels_test_set[ind]:
            count += 1
        test_set_results.append(predicted_label)

    print(count/len(test_set))

if __name__ == '__main__':
    train_RFC()