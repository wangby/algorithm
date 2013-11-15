#!/usr/bin/env python

#from __future__ import print_function
from sys import argv, stderr
from os import system, path
from libshorttext.classifier import *
from libshorttext.converter import Text2svmConverter

def exit_with_help():
	stderr.write("""Usage: text-train.py [options] training_file [model]

options: 
    -P {0|1|2|3|4|5|6|7|converter_directory}
        Preprocessor options. The options include stopwrod removal, 
        stemming, and bigram. (default 1)
        0   no stopword removal, no stemming, unigram
        1   no stopword removal, no stemming, bigram
        2   no stopword removal, stemming, unigram
        3   no stopword removal, stemming, bigram
        4   stopword removal, no stemming, unigram
        5   stopword removal, no stemming, bigram
        6   stopword removal, stemming, unigram
        7   stopword removal, stemming, bigram
        If a preprocssor directory is given instead, then it is assumed
        that the training data is already in LIBSVM format. The preprocessor
        will be included in the model for test. 
    -G {0|1}
        Grid search for the penalty parameter in linear classifiers. (default 0)
        0   disable grid search (faster)
        1   enable grid search (slightly better results)
    -F {0|1|2|3}
        Feature representation. (default 0)
        0   binary feature
        1   word count 
        2   term frequency
        3   TF-IDF (term frequency + IDF)
    -N {0|1}
        Instance-wise normalization before training/test.
        (default 1 to conduct normalization)
    -L {0|1|2|3}
        Classifier. (default 0)
        0   support vector classification by Crammer and Singer
        1   L1-loss support vector classification
        2   L2-loss support vector classification
        3   logistic regression
    -A extra_svm_file
        Append extra libsvm-format data. This parameter can be applied many 
        times if more than one extra svm-format data set need to be appended.
    -f
        Overwrite the existing model file.
Examples:
    text-train.py -L 3 -F 1 -N 1 raw_text_file model_file 
    text-train.py -P text2svm_converter -L 1 converted_svm_file
""")
	exit(1)

if __name__ == '__main__':
	
	if len(argv) < 2:
		exit_with_help()

	text_converter      = None
	converter_arguments = ''
	grid_arguments      = '0'
	feature_arguments   = '' 
	liblinear_arguments = '' # default is -s 4
	data                = None
	force               = False
	model_path          = None
	extra_svm_files     = []

	i = 1
	while(True):
		if i >= len(argv): break

		if not argv[i].startswith('-'):
			if data is None:
				data = argv[i]
			elif model_path is None:
				model_path = argv[i]
			else:
				stderr.write('Error: Wrong usage.\n')
				exit_with_help()
			i += 1
			continue

		if argv[i] == '-f':
			force = True
			i += 1
			continue

		if i+1 >= len(argv):
			stderr.write('Error: Invalid usage of option ' + argv[i] + '\n')
			exit_with_help()
		
		value = argv[i+1]
		if argv[i] == '-P':
			if value in ['0', '1', '2', '3', '4', '5', '6', '7']:
				opt = int(value)
				stopword = (opt & 4) >> 2
				stemming = (opt & 2) >> 1
				feature = opt & 1
				converter_arguments = '-stopword {0} -stemming {1} -feature {2}'.format(stopword, stemming, feature)
			elif value.startswith('@'):
				converter_arguments += value[1:]
			else:
				text_converter = value
		elif argv[i] == '-G':
			if value not in ['0', '1'] and not value.startswith('@'):
				stderr.write('Error: Invalid usage of option -G.\n')
				exit_with_help()
			elif value.startswith('@'):
				grid_arguments = value[1:]
			else:
				grid_arguments = value
		elif argv[i] == '-F':
			if value == '0':
				feature_arguments += ' -D 1'
			elif value == '1':
				feature_arguments += ' -D 0'
			elif value == '2':
				feature_arguments += ' -D 0 -T 1'
			elif value == '3':
				feature_arguments += ' -D 0 -T 1 -I 1'
			elif value.startswith('@'):
				feature_arguments += value[1:]
			else:
				stderr.write('Error: Invalid usage of option -F.\n')
				exit_with_help()
		elif argv[i] == '-N':
			if value not in ['0', '1']: 
				stderr.write('Error: Invalid usage of option -N.\n')
				exit_with_help()
			feature_arguments += ' -N ' + value
		elif argv[i] == '-L':
			if value == '0': 
				liblinear_arguments = '-s 4'
			elif value == '1':
				liblinear_arguments = '-s 3'
			elif value == '2':
				liblinear_arguments = '-s 1'
			elif value == '3':
				liblinear_arguments = '-s 7'
			elif value.startswith('@'):
				liblinear_arguments = value[1:]
			else:
				stderr.write('Error: Invalid usage of option -L.\n')
				exit_with_help()
		elif argv[i] == '-A':
			extra_svm_files += [value]
		elif argv[i] == '-x':
			if value.lower() == 'grid':
				system(path.dirname(LIBLINEAR_HOME) + '/../grid.py')
			elif value.lower() == 'liblinear':
				system(LIBLINEAR_HOME + '/train')
			else:
				stderr.write('Error: Invalid usage of option -x. No command ' + value + '\n')
				exit_with_help()
			exit(1)
		else:
			stderr.write('Error: No option ' + argv[i] + '\n')
			exit_with_help()
		
		i += 2


	if not data:
		stderr.write('Error: Training data path is not given.\n')
		exit_with_help()

	model_path = model_path or path.split(data)[1] + '.model'
	if path.exists(model_path) and not force:
		stderr.write('{0} already exists. Use -f to overwrite the existing file.\n'.format(model_path))
		exit(1)
		
	if not text_converter:
		m, svm_file = train_text(data, converter_arguments=converter_arguments, grid_arguments=grid_arguments, feature_arguments=feature_arguments, train_arguments=liblinear_arguments, extra_svm_files = extra_svm_files)
		m.save(model_path, force)
	else:
		if len(extra_svm_files) > 0:
			stderr.write('Warning: Extra svm files are ignored.\n')
		text_converter = Text2svmConverter().load(text_converter)
		m = train_converted_text(data, text_converter, grid_arguments=grid_arguments, feature_arguments=feature_arguments, train_arguments=liblinear_arguments)
		m.save(model_path, force)

