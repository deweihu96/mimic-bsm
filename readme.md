Some of the code is referred from this [repo](https://github.com/jamesmullenbach/caml-mimic). 

# BSM-MIMICIII
Code for the paper A Binary Soft Mask Approach for ICD Coding from Clinical Text

## Model
![image](https://user-images.githubusercontent.com/48980790/146659577-ab4c97f9-f5b7-47e6-b3a6-f98ca43202df.png)


## Dependencies
- Python 3.7
- PyTorch 1.9.0 
- tqdm 4.41.1
- scikit-learn 0.22.2
- numpy 1.19.5
- scipy 1.4.1
- pandas 1.1.5
- gensim 3.6.0
- nltk 3.2.5  

All of the work has been done on Google Colab with GPU, and these packages are directly imported from the Colab. Other versions may also work.

## Data Processing
The Data Processing method is referred from the paper [Explainable Prediction of Medical Codes from Clinical Text](https://arxiv.org/abs/1802.05695) and this [repo](https://github.com/jamesmullenbach/caml-mimic).

First, edit the data and model directory in the file `constant.py` , and place the data into the  `mimicdata` like this:

```
mimicdata
|   D_ICD_DIAGNOSES.csv
|   D_ICD_PROCEDURES.csv
|   ICD9_descriptions (already in repo)
└───mimic3/
|   |   NOTEEVENTS.csv
|   |   DIAGNOSES_ICD.csv
|   |   PROCEDURES_ICD.csv
|   |   *_hadm_ids.csv (already in repo)
```

Run all cells in `notebooks/dataproc_mimic_III.ipynb`.  It will take about more than 10 minutes.

## Train a new model
To train a new BSM model, first modify the file `constants.py`, and run `train_bsm.sh`.

## Test a model

We provide a trained BSM model in `saved_models` folder, by running `test_bsm.sh` you can check the metrics. There is also a `CAML` model in this folder, published by Mullenbach: https://github.com/jamesmullenbach/caml-mimic, the original paper: [Explainable Prediction of Medical Codes from Clinical Text](https://arxiv.org/abs/1802.05695).


## Results on MIMIC-III top 50 labels test set
![image](https://user-images.githubusercontent.com/48980790/146659648-3355bf92-bf01-445a-a5d3-ff2bc1c3b988.png)


## Full Results

In `results/omission` and `results/selection` , there are analysis we have done. By running these shell commands you will get some files used for the further analysis, and the examples in the appendix. In `results/saved_results`, these are results we got.

For comparison of the explainability between CAML and BSM, run all cells in  `notebooks/comparison.ipynb` and you'll get figures.

