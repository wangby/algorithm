#usage: python numqueries.py data.txt
import os,sys
FILE=sys.argv[1]
qs=[]
for line in open(FILE,'r'):
    qid=line.split()[1].split(':')[1]
    if qid not in qs:
        qs.append(qid)
print len(qs)
