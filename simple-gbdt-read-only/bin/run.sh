#! /bin/sh

D=14
TRAIN_DATE=20131116
TEST_DATE=20131117

if [ $# -eq 3 ];then
  TRAIN_DATE=$1
  TEST_DATE=$2
  D=$3
elif [ $# -eq 1 ];then
  TRAIN_DATE=$1
  TEST_DATE=$1
elif  [ $# -eq 2 ];then
  TRAIN_DATE=$1
  TEST_DATE=$2
fi

TRAIN_FILE=search_log_${TRAIN_DATE}_train_log_norm_14
TEST_FILE=search_log_${TEST_DATE}_train_log_norm_14
MODEL_FILE=${TRAIN_FILE}.model
OUT_FILE=${TEST_FILE}.prediction
SCORE_FILE=${TEST_FILE}.score

CMD_TRAIN="./gbrt -f ctr -t -i ${TRAIN_FILE} -m ${MODEL_FILE} -d ${D}"
echo ${CMD_TRAIN}
${CMD_TRAIN}

echo -----------------------------
CMD_PREDICT="./gbrt -f ctr -p -i ${TEST_FILE} -m ${MODEL_FILE} -d ${D}"
echo ${CMD_PREDICT}
${CMD_PREDICT}

echo -----------------------------
CMD_AUC="./auc $OUT_FILE"
echo ${CMD_AUC}
${CMD_AUC} | tee ${SCORE_FILE}

