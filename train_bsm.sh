python3 /path/to/learn/training.py \
/path/to/mimicdata/mimic3/train_50.csv \
/path/to/mimicdata/mimic3/vocab.csv \
50 \
bsm_maxpooling \
100 \
--num-filter-maps 500 \
--filter-size 4 \
--dropout 0.2 \
--batch-size 8 \
--patience 6 \
--criterion f1_micro \
--lr 0.001 \
--lstm-hidden-size 64 \
--lambda-p 0.25 \
--lambda-sel 0.02 \
--lambda-cont 0.05 \
--tau 0.8 \
--embed-file /path/to/mimicdata/mimic3/processed_full.embed \
--gpu