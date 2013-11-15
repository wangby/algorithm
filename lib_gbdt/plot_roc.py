#! /usr/bin/python
"""
=======================================
Receiver operating characteristic (ROC)
=======================================

"""
print __doc__

import sys
import numpy as np
import pylab as pl

from sklearn.metrics import roc_curve, auc

if len(sys.argv) < 2:
    print "Usage: python plot_roc.py result_file_path"
    exit()

file_path = sys.argv[1]
data = []
data_ = []
f = open(file_path)
line = 'l'
while line:
    line = f.readline()
    if line:
        line = line.strip()
        items = line.split('\t')
        if len(items) != 2:
            continue
        y = int(items[0])
        prob = float(items[1])
        data.append([y,prob])
if not data:
    print 'out file is empty or format wrong!'
    exit()

data = np.array(data)
# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(data[:,0], data[:, 1])
roc_auc = auc(fpr, tpr)
print thresholds[0:10]
print "AUC : %f" % roc_auc

