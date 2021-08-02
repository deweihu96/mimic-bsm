import torch
from torch.autograd import Variable
import torch.nn.functional as F

import pandas as pd
import argparse
import sys
import datasets
import time
import numpy as np
from termcolor import colored
import pdb
from tqdm import tqdm
import learn.tools as tools
from constants import *
import json
import statistics

# highlight text
RESET = '\033[0m'
def get_rgb(color,background=True):
    r = color[0]
    g = color[1]
    b = color[2]
    return '\033[{};2;{};{};{}m'.format(48 if background else 38, r, g, b)

def latexhl(words):
    return "\hl{"+words+"}"

def init(args):
    # init the model; dicts; 
    print("loading lookups...")
    dicts = datasets.load_lookups(args)
    # ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']

    print("loading model...")
    model = tools.pick_model(args, dicts) #remember to set as test mode
    print(model)


    if args.index>0:
        pass
    else:
        return dicts,model


def generator(args):
    print("loading lookups...")
    dicts = datasets.load_lookups(args)
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']


    print("loading model...")
    model = tools.pick_model(args, dicts) #remember to set as test mode
    print(model)

    print("loading data...")
    data = pd.read_csv(args.data_path)

    text = data.iloc[args.index,2]
    print("The discharge summary you select is: ")
    print(text) #string


    print("The ICD codes it corresponds to:")
    target = data.iloc[args.index,3]
    print(target)

    # transform text and code to tensor
    tensor_text = [int(w2ind[w]) if w in w2ind else len(w2ind)+1 for w in text.split()]
    tensor_text = Variable(torch.LongTensor(tensor_text)).view(1,-1)
    if len(tensor_text) > MAX_LENGTH:
        tensor_text = tensor_text[:MAX_LENGTH]


    num_labels = len(ind2c)
    labels_idx = np.zeros(num_labels)
    for l in target:
        if l in c2ind.keys():
            code = int(c2ind[l])
            labels_idx[code] = 1
    labels_idx = Variable(torch.FloatTensor(labels_idx)).view(1,-1)
    
    if args.gpu:
        tensor_text = tensor_text.cuda()
        labels_idx = labels_idx.cuda()

    return dicts,model,text,tensor_text,target,labels_idx

def main(args):
    pseles = dict()
    stat = dict()
    rationale_percent = dict()
    if args.index < 0: #  the whole data set

        dicts, model = init(args)
        ind2c = dicts['ind2c']
        num_labels = len(dicts["ind2c"])
        gen = datasets.data_generator(args.data_path, dicts, 1, num_labels, version="mimic3")
        model.eval()
        for bacth_idx,tup in tqdm(enumerate(gen)):
            data, target, hadm_ids, _, descs = tup
            with torch.no_grad():
                data, target = Variable(torch.LongTensor(data)), Variable(torch.FloatTensor(target))
                    
                if args.gpu:
                    data = data.cuda()
                    target = target.cuda()

                output,mask,loss,mask_loss,psel = model(data,target)

                if args.rationale_percentage:
                    y_hat = torch.sigmoid(y_hat)
                    y_hat = (y_hat>0.5).float()

                    codes = [ind2c[i] for i in np.where(y_hat.cpu()==1)[1]]
                    # pdb.set_trace()
                    rationale = torch.argmax(mask,dim=2) # [1,L]
                    rationale_indices = np.where(rationale.cpu()==1)[1]
                    rationale_percent[bacth_idx] = len(rationale_indices)/data.shape[1]
                else:
                    pseles[bacth_idx] = psel.item()
        if args.rationale_percentage:
            c = rationale_percent.values()
            stat["mean"] = statistics.mean(c)
            stat["median"] = statistics.median(c)
            stat["stdev"] = statistics.stdev(c)
            stat["var"] = statistics.variance(c)
            with open(args.output_json,"w") as fd:
                fd.write(json.dumps([rationale_percent,stat]))
        else:
            c = pseles.values()
            stat["mean"] = statistics.mean(c)
            stat["median"] = statistics.median(c)
            stat["stdev"] = statistics.stdev(c)
            stat["var"] = statistics.variance(c)
            with open(args.output_json, 'w') as fd:
                fd.write(json.dumps([pseles, stat])) 
            print("selection probability has been written to the file: " + args.output_json)

    else:
        # extract on the single file (the given index)

        
        dicts, model, raw_text,data, raw_target,target = generator(args)
        ind2c = dicts['ind2c']

        # make predictions 
        with torch.no_grad():
            y_hat,mask,loss,mask_loss,psel = model(data,target)
        y_hat = torch.sigmoid(y_hat)
        y_hat = (y_hat>0.5).float()

        codes = [ind2c[i] for i in np.where(y_hat.cpu()==1)[1]]
        # pdb.set_trace()
        rationale = torch.argmax(mask,dim=2) # [1,L]
        rationale_indices = np.where(rationale.cpu()==1)[1]
        text_list = raw_text.split()
        if args.latex:
            rationale_result = " ".join(latexhl(text_list[t]) if t in rationale_indices else text_list[t] for t in range(len(text_list)))
        else:
            rationale_result = " ".join(colored(text_list[t],"white","on_blue") if t in rationale_indices else text_list[t] for t in range(len(text_list)))
        print(rationale_result)

        print("Predicted ICD codes:",codes)

        # percent of rationales
        print("Proportion of extracted words: ",len(rationale_indices)/len(text_list))
















if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a neural network on some clinical documents")
    parser.add_argument("data_path", type=str,
                        help="path to a file containing sorted train data. dev/test splits assumed to have same name format with 'train' replaced by 'dev' and 'test'")
    parser.add_argument("vocab", type=str, help="path to a file holding vocab word list for discretizing words")
    parser.add_argument("Y", type=str, help="size of label space")

    parser.add_argument("model", type=str, choices=["cnn_vanilla", "laat","jointlaat", "rlaat","rnn", "conv_attn", 
                        "multi_conv_attn", "logreg", "saved","encaml","hencaml","rcnn","rcaml","caml","cnnmaxpooling",
                        "bsm_maxpooling","bsm_caml","bsm_encaml","bsm_laat","mtm"], help="model")

    parser.add_argument("--embed-file", type=str, required=False, dest="embed_file",
                        help="path to a file holding pre-trained embeddings")
    parser.add_argument("--embed-size", type=int, required=False, dest="embed_size", default=100,
                        help="size of embedding dimension. (default: 100)")       
    # LSTM
    parser.add_argument("--lstm-hidden-size",type=int,required=False,dest="lstm_hidden_size",default=512)
    parser.add_argument("--lstm-classifier-hidden-size",type=int,required=False,dest="lstm_classifier_hidden_size",default=512)
    parser.add_argument("--lstm-project-size",type=int,required=False,dest="lstm_project_size",default=128)
    
    # CNN

    parser.add_argument("--filter-size", type=str, required=False, dest="filter_size", default=3,
                        help="size of convolution filter to use. (default: 3) For multi_conv_attn, give comma separated integers, e.g. 3,4,5")
    parser.add_argument("--num-filter-maps", type=int, required=False, dest="num_filter_maps", default=50,
                        help="size of conv output (default: 50)")

    parser.add_argument("--weight-decay", type=float, required=False, dest="weight_decay", default=0,
                        help="coefficient for penalizing l2 norm of model weights (default: 0)")
    parser.add_argument("--lr", type=float, required=False, dest="lr", default=1e-3,
                        help="learning rate for Adam optimizer (default=1e-3)")
    parser.add_argument("--optimizer",type=str,choices=["Adam","AdamW"],required=False,dest="optimizer",default="Adam",help="optimizer")
    parser.add_argument("--batch-size", type=int, required=False, dest="batch_size", default=16,
                        help="size of training batches")
                        
    parser.add_argument("--dropout", dest="dropout", type=float, required=False, default=0.5,
                        help="optional specification of dropout (default: 0.5)")
    parser.add_argument("--dataset", type=str, choices=['mimic2', 'mimic3'], dest="version", default='mimic3', required=False,
                        help="version of MIMIC in use (default: mimic3)")
    parser.add_argument("--test-model", type=str, dest="test_model", required=False, help="path to a saved model to load and evaluate")
    parser.add_argument("--criterion", type=str, default='f1_micro', required=False, dest="criterion",
                        help="which metric to use for early stopping (default: f1_micro)")
    parser.add_argument("--patience", type=int, default=3, required=False,
                        help="how many epochs to wait for improved criterion metric before early stopping (default: 3)")
    parser.add_argument("--gpu", dest="gpu", action="store_const", required=False, const=True,
                        help="optional flag to use GPU if available")
    parser.add_argument("--quiet", dest="quiet", action="store_const", required=False, const=True,
                        help="optional flag not to print so much during training")
    
    # argument for BSM
    parser.add_argument("--lambda-p",dest="lambda_p",required=False,type=float, default=0.30, help="limitation of percent of words")
    parser.add_argument("--lambda-sel",dest="lambda_sel",required=False,type=float, default=0.04, help="selection regularizer")
    parser.add_argument("--lambda-cont",dest="lambda_cont",required=False,type=float, default=0.04, help="continuity regularizer")
    parser.add_argument("--tau",dest="tau",required=False,type=float,default=0.8,help="temperature in gumble softmax")


    parser.add_argument("--output-json",dest="output_json",type=str,required=False)
    parser.add_argument("--index",dest="index",default=-1,type=int,required=False,help="-1: the whole file; others: the index of document")
    parser.add_argument("--highlight-less",dest="highligh_less",action="store_const", required=False, const=False,help="whether highlight words corresponds to the predicted codes")
    parser.add_argument("--RGB",dest="RGB",type=str,required=False,help="the color of hilights, json file")
    parser.add_argument("--rationale-percentage",dest="rationale_percentage",action="store_const",const=False,required=False,help="while runing on the whole file, if true, get the rationale percentage")
    parser.add_argument("--latex",dest="latex",action="store_const",const=True,required=False,help="print latex highlight codes")
    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    main(args)