#!/bin/bash

python ../text-train.py -f -A train_feats1 -A train_feats2 train_file
python ../text-predict.py -f -A test_feats1 -A test_feats2 test_file train_file.model predict_result
python demo.py
