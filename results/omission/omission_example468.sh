python3 \
/path/to/omission_on_text.py \
/path/to/mimicdata/mimic3/test_50.csv \
/path/to/mimicdata/mimic3/vocab.csv \
50 \
caml \
--embed-file /path/to/mimicdata/mimic3/processed_full.embed \
--test-model /path/to/saved_models/caml/CAML_mimic3_50.pth \
--filter-size 10 \
--num-filter-maps 50 \
--dropout 0.2 \
--index 468 \
--gpu 