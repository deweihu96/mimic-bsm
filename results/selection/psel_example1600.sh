python3 \
/path/to/extractor.py \
/path/to/mimicdata/mimic3/test_50.csv \
/path/to/mimicdata/mimic3/vocab.csv \
50 \
bsm_maxpooling \
--embed-file /path/to/mimicdata/mimic3/processed_full.embed \
--test-model /path/to/saved_models/bsm/model.pth \
--lstm-hidden-size 64 \
--filter-size 4 \
--dropout 0.2 \
--num-filter-maps 500 \
--lambda-p 0.25 \
--tau 0.8 \
--gpu \
--index 1600