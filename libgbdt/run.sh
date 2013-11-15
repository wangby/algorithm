
./gbdt_train -P 0.5 -r 0.8 -t 100 -s 0.03 -n 30 -d 5 -m test.model -f train
cat test | ./gbdt_predict test.model >test.result
