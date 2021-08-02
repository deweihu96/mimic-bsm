"""
    Data loading methods
"""
import pdb
from collections import defaultdict
import csv
import math
import numpy as np
import sys
# add the path to the repo, if needed
# sys.path.append('path')
from constants import *

class Batch:
    """
        This class and the data_generator could probably be replaced with a PyTorch DataLoader
    """
    def __init__(self):
        self.docs = []
        self.labels = []
        self.hadm_ids = []
        self.code_set = set()
        self.length = 0
        self.max_length = MAX_LENGTH
        self.descs = []


        ## add something for half target
        self.half_labels = []
        self.half_code_set = set()



    def add_instance(self, row, ind2c, c2ind, w2ind, num_labels):
        hadm_id = int(row[1])
        text = row[2]
        length = int(row[4])

        labels = set()
        cur_code_set = set()      
        labels_idx = np.zeros(num_labels)
        labelled = False
        desc_vecs = []

        #get codes as a multi-hot vector
        for l in row[3].split(';'):
            if l in c2ind.keys():
                code = int(c2ind[l])
                labels_idx[code] = 1
                cur_code_set.add(code)
                labelled = True
        if not labelled:
            return

        #OOV words are given a unique index at end of vocab lookup
        text = [int(w2ind[w]) if w in w2ind else len(w2ind)+1 for w in text.split()]
        #truncate long documents
        if len(text) > self.max_length:
            text = text[:self.max_length]

        #build instance
        self.docs.append(text)
        self.labels.append(labels_idx)
        self.hadm_ids.append(hadm_id)
        self.code_set = self.code_set.union(cur_code_set)
        #reset length
        self.length = min(self.max_length, length)


    def pad_docs(self):

        #pad all docs to have self.length
        padded_docs = []
        for doc in self.docs:
            if len(doc) < self.length:
                doc.extend([0] * (self.length - len(doc)))
            padded_docs.append(doc)
        self.docs = padded_docs

            

    def to_ret(self):

        
        return np.array(self.docs), np.array(self.labels), np.array(self.hadm_ids), self.code_set,\
                np.array(self.descs)

def pad_desc_vecs(desc_vecs):
    #pad all description vectors in a batch to have the same length
    desc_len = max([len(dv) for dv in desc_vecs])
    pad_vecs = []
    for vec in desc_vecs:
        if len(vec) < desc_len:
            vec.extend([0] * (desc_len - len(vec)))
        pad_vecs.append(vec)
    return pad_vecs

def data_generator(filename, dicts, batch_size, num_labels, desc_embed=False, version='mimic3',hierarchical=False):
    """
        Inputs:
            filename: holds data sorted by sequence length, for best batching
            dicts: holds all needed lookups
            batch_size: the batch size for train iterations
            num_labels: size of label output space
            desc_embed: true if using DR-CAML (lambda > 0)
            version: which (MIMIC) dataset
        Yields:
            np arrays with data for training loop.
    """
    if hierarchical is not True:
        ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
        
    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        #header
        next(r)
        cur_inst = Batch()
        for row in r:
            #find the next `batch_size` instances
            if len(cur_inst.docs) == batch_size:
                cur_inst.pad_docs()
                yield cur_inst.to_ret()
                #clear
                cur_inst = Batch()
            cur_inst.add_instance(row, ind2c, c2ind, w2ind, num_labels)
        cur_inst.pad_docs()
        yield cur_inst.to_ret()

def load_vocab_dict(args, vocab_file):
    #reads vocab_file into two lookups (word:ind) and (ind:word)
    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for i,line in enumerate(vocabfile):
            line = line.rstrip()
            if line != '':
                vocab.add(line.strip())
    

    ind2w = {i+1:w for i,w in enumerate(sorted(vocab))}
    w2ind = {w:i for i,w in ind2w.items()}
    return ind2w, w2ind



def load_lookups(args):
    """
        Inputs:
            args: Input arguments
            desc_embed: true if using DR-CAML
        Outputs:
            vocab lookups, ICD code lookups, description lookup, description one-hot vector lookup
    """
    #get vocab lookups
    ind2w, w2ind = load_vocab_dict(args, args.vocab)

    #get code and description lookups
    if args.Y == 'full': 
        ind2c = load_full_codes(args.data_path, version=args.version)
    else:
        codes = set()
        with open("%s/TOP_%s_CODES.csv" % (MIMIC_3_DIR, str(args.Y)), 'r') as labelfile:
            lr = csv.reader(labelfile)
            for i,row in enumerate(lr):
                codes.add(row[0])
        ind2c = {i:c for i,c in enumerate(sorted(codes))}
        
    c2ind = {c:i for i,c in ind2c.items()}
    dicts = {'ind2w': ind2w, 'w2ind': w2ind, 'ind2c': ind2c, 'c2ind': c2ind}


    return dicts

def load_full_codes(train_path, version='mimic3'):
    """
        Inputs:
            train_path: path to train dataset
            version: which (MIMIC) dataset
        Outputs:
            code lookup
    """
    #build code lookups from appropriate datasets
    if version == 'mimic2':
        ind2c = defaultdict(str)
        codes = set()
        with open('%s/proc_dsums.csv' % MIMIC_2_DIR, 'r') as f:
            r = csv.reader(f)
            #header
            next(r)
            for row in r:
                codes.update(set(row[-1].split(';')))
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i:c for i,c in enumerate(sorted(codes))})
    else:
        codes = set()
        for split in ['train', 'dev', 'test']:
            with open(train_path.replace('train', split), 'r') as f:
                lr = csv.reader(f)
                next(lr)
                for row in lr:
                    for code in row[3].split(';'):
                        codes.add(code)
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i:c for i,c in enumerate(sorted(codes))})
    return ind2c



def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits, 
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code




