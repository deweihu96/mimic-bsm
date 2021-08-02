"""
    Various utility methods
"""
import pdb
import csv
import json
import math
import os
import pickle

import torch
from torch.autograd import Variable


from constants import *
import datasets
import persistence
import numpy as np

import sys
# add the path to the repo, if needed
# sys.path.append('path')

from learn import bsm
from learn import hdwmodels



def pick_model(args, dicts):
    """
        Use args to initialize the appropriate model
    """
    # pdb.set_trace()
    Y = len(dicts['ind2c'])
    if "bsm" in args.model:
        loss_weights = [1, args.lambda_sel, args.lambda_cont]
    # baseline models
    if args.model == "caml":
        filter_size = int(args.filter_size)
        model = hdwmodels.CAML(Y, args.embed_file, filter_size, args.num_filter_maps,  args.gpu, dicts,
                                    embed_size=args.embed_size, dropout=args.dropout)

    elif args.model == "laat":
        model = hdwmodels.LAAT(Y, args.embed_file, dicts, args.lstm_hidden_size,args.lstm_project_size,
                            args.dropout,args.gpu,args.embed_size)

    elif args.model == "cnnmaxpooling":
        filter_size = int(args.filter_size)
        model = hdwmodels.CNNMaxPooling(Y, args.embed_file, filter_size, 
                            args.num_filter_maps, args.gpu, dicts, args.embed_size, args.dropout)

    elif args.model == "encaml":
        model = hdwmodels.EnCAML(Y, args.embed_file, dicts,args.num_filter_maps,
                                    args.dropout,args.gpu,args.embed_size)
    
    ## BSM models
    elif args.model == "bsm_caml":
        filter_size = int(args.filter_size)
        
        model = bsm.CAML(Y, args.embed_file, dicts, args.dropout, 
                        args.lstm_hidden_size,
                        args.num_filter_maps,
                        filter_size,
                        args.gpu,
                        args.lambda_p,
                        args.tau,
                        loss_weights)

    elif args.model == "bsm_maxpooling":
        filter_size = int(args.filter_size)

        model = bsm.CNNMaxPooling(Y, args.embed_file, dicts,
                                args.dropout,
                                args.lstm_hidden_size,
                                args.num_filter_maps,
                                filter_size,
                                args.gpu,
                                args.lambda_p,
                                args.tau,
                                loss_weights)


    elif args.model == "bsm_encaml":
        model = bsm.EnCAML(Y ,args.embed_file, dicts, 
                        args.dropout,
                        args.lstm_hidden_size,
                        args.num_filter_maps,
                        args.gpu,
                        args.lambda_p,
                        args.tau,
                        loss_weights)

    elif args.model == "bsm_laat":  
        model = bsm.LAAT(Y,
                        args.embed_file,
                        dicts,
                        args.dropout,
                        args.lstm_hidden_size,
                        args.lstm_classifier_hidden_size,
                        args.lstm_project_size,
                        args.gpu,
                        args.lambda_p,
                        args.tau,
                        loss_weights)

    if args.test_model:
        sd = torch.load(args.test_model)
        model.load_state_dict(sd)
    if args.gpu:
        model.cuda()
    
    return model

def make_param_dict(args):
    """
        Make a list of parameters to save for future reference
    """
    param_vals = [args.Y, args.filter_size, args.dropout, args.num_filter_maps, 
                 args.command, args.weight_decay, args.version, args.data_path, args.vocab, args.embed_file, args.lr, args.lambda_p, args.lambda_sel, args.lambda_cont, args.tau]
    param_names = ["Y", "filter_size", "dropout", "num_filter_maps", "command",
                   "weight_decay", "version", "data_path", "vocab", "embed_file", "lr", "lambda_p", "lambda_sel", "lambda_cont", "tau"]
    params = {name:val for name, val in zip(param_names, param_vals) if val is not None}
    return params

def build_code_vecs(code_inds, dicts):
    """
        Get vocab-indexed arrays representing words in descriptions of each *unseen* label
    """
    code_inds = list(code_inds)
    ind2w, ind2c, dv_dict = dicts['ind2w'], dicts['ind2c'], dicts['dv']
    vecs = []
    for c in code_inds:
        code = ind2c[c]
        if code in dv_dict.keys():
            vecs.append(dv_dict[code])
        else:
            #vec is a single UNK if not in lookup
            vecs.append([len(ind2w) + 1])
    #pad everything
    vecs = datasets.pad_desc_vecs(vecs)
    return (torch.cuda.LongTensor(code_inds), vecs)

