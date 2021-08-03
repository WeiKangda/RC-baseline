NUM_SPLITS=8
for ((SPLIT_NUM = 0 ; SPLIT_NUM < $NUM_SPLITS ; SPLIT_NUM++ ));
do
    python3 COVID_QA_zh_Annotation.py --num_splits=$NUM_SPLITS --split_num=$SPLIT_NUM &
done