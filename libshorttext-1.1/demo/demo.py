#!/usr/bin/env python

import sys, os
sys.path += ['..']
from libshorttext.analyzer import *

if __name__ == '__main__':
    predict_result = InstanceSet('predict_result')
    analyzer = Analyzer('train_file.model')
    insts = predict_result.select(wrong, with_labels(['Books', 'Music', 'Art', 'Baby']), sort_by_dec, subset(100))
    analyzer.info(insts)
    analyzer.gen_confusion_table(insts)
    insts.load_text()
    print(insts[61])
    analyzer.analyze_single(insts[61], 3)
    analyzer.analyze_single('beatles help longbox sealed usa 3 cd single', 3)
