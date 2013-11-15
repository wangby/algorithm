#! /usr/local/bin/python

thresh = []

for i in range(100):
    thresh.append(float(i)/100)


for i in thresh:
    file_p = open("test.result")
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for line in file_p:
        flds = line[:-1].split("\t")
        label = int(flds[0])
        score = float(flds[1])
        if score > i:
            predict_score = 1
        else:
            predict_score = 0
        if predict_score == 1 and label == 1:
            tp += 1
        elif predict_score == 1 and label == 0:
            fp += 1
        elif predict_score == 0 and label ==0:
            tn += 1
        elif predict_score == 0 and label == 1:
            fn += 1
    if (tp+fp) != 0 and (tp+fn) != 0:
        print "%s\t%s\t%s" %(i, float(tp)/(tp+fp), float(tp)/(tp+fn))
    file_p.close()


        
       
