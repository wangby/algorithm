#!/usr/bin/env python

#from __future__ import print_function
from sys import argv, stderr
from os import path
from libshorttext.classifier import *


def exit_with_help():
	stderr.write("""Usage: text-predict.py [options] test_file model output

options:
    -f
        Overwrite the existing output file.
    -a {0|1}
        Output options. (default 1)
        0   Store only predicted labels. The information is NOT sufficient 
            for interactive analysis. Use this option if you would like to get 
            only accuracy.
        1   More information is stored. The output provides information for 
            interactive analysis, but the size of output can become much larger.
    -A extra_svm_file
        Append extra libsvm-format data. This parameter can be applied many 
        times if more than one extra svm-format data set need to be appended.
""")
	exit(1)



if __name__ == '__main__':
	if len(argv) < 4:
		exit_with_help()

	data                = None
	model               = None
	output              = None
	liblinear_arguments = ''
	force               = False
	analyzable          = True
	extra_svm_files     = []
	
	i = 1
	while(True):
		if i >= len(argv): break

		if not argv[i].startswith('-'):
			if data is None:
				data = argv[i]
			elif model is None:
				model = argv[i]
			elif output is None:
				output = argv[i]
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
		if argv[i] == '-a':
			if value == '0':
				analyzable = False
			elif value == '1':
				analyzable = True
			else:
				stderr.write('Error: Invalid usage of option -a\n')
				exit_with_help()
		elif argv[i] == '-A':
			extra_svm_files += [value]
		else:
			stderr.write('Error: No option ' + argv[i] + '\n')
			exit_with_help()
		
		i += 2

	if output is None:
		stderr.write('Output path is not given.\n')
		exit_with_help()

	if path.exists(output) and not force:
		stderr.write('{0} already exists. Use -f to overwrite the existing file.\n'.format(output))
		exit(1)

	m = TextModel()
	m.load(model)
	predict_result = predict_text(data, m, svm_file=None, predict_arguments = liblinear_arguments, extra_svm_files = extra_svm_files)

	print("Accuracy = {0:.4f}% ({1}/{2})".format(
		predict_result.get_accuracy()*100, 
		sum(ty == py for ty, py in zip(predict_result.true_y, predict_result.predicted_y)),
		len(predict_result.true_y)))

	predict_result.save(output, analyzable)
