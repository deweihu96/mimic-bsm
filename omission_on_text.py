"""
Given an index of the dataset, Open one model, and find the words it removed to make all predicted labels change (the switching point) for that discharge summary.
"""
from numpy.core.numeric import indices
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

import learn.tools as tools
from constants import *

def latexhl(words):
    return "\hl{"+words+"}"

def init(args):
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


def omission(attn,indices,ind2c,w2ind,text,data,target,model,gpu=True):
    """
        To find the omission words for all predicted labels in the text.
    Input:
        attn: attention weights
        indices: the predicted labels for the text; (indices over the dict ind2c)
        ind2c, ind2w: dicts
        text: raw text, list of words
        data: convert raw text to tensor 
        target: target labels, tensor
        model: model to evaluate 
    Output:
        omission_words: set, all words deleted for the predicted labels;
        percent: percentage of deleted words over the length of input text

    """
    omission_words = set() # record the words deleted 
    omission_indices = set() # record the indices of deleted words
    if type(attn) == list: # for encaml
        pass 

    else:

        for index in indices:
            print("The code:",ind2c[index],"is now under evaluating......")

            attn_words = attn[index,:].view(-1) # get the attention weights for each label.

            attn_rank = list(attn_words.topk(len(attn_words))[1].cpu().numpy())  # get the indices of each word's weight from high to low
            
            
            data_ = data.detach().clone() # due to overlapps, we need to initialize the data for each label;

            # start to delete words in text:
            for i in range(len(text)):

                
                # deleted the words
                data_ = torch.cat((data_[:,:attn_rank[i]-i], data_[:,attn_rank[i]+1-i:]),dim=1)

                if gpu:
                    data_ = data_.cuda()
                # pdb.set_trace()
                if attn_rank[i] < len(text):
                    omission_words.add(text[attn_rank[i]] if text[attn_rank[i]] in list(w2ind.keys()) else "unk")
                    omission_indices.add(attn_rank[i])

                with torch.no_grad():
                    output,alpha_,loss,a,b,c = model(data_,target)
                
                indices_ = list(np.where(torch.sigmoid(output).cpu()>0.5)[1])

                if index not in indices_:
                    print("Finished evaluation of code:",ind2c[index])
                    break
        # pdb.set_trace()
        percent = len(omission_indices)/len(text)
        
    return omission_words,omission_indices, percent

def main(args):

    start = time.time()
    dicts, model, raw_text, data, raw_target,target = init(args)
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']

    # make predictions 
    with torch.no_grad():
        y_hat, attn, loss,a,b,c = model(data,target)
    y_hat = torch.sigmoid(y_hat)
    y_hat = (y_hat>0.5).float()

    codes = [ind2c[i] for i in np.where(y_hat.cpu()==1)[1]] 

    ## variables used for omission
    list_text = raw_text.split()
    indices = list(np.where(y_hat.cpu()==1)[1])
    # pdb.set_trace()
    attn = attn.squeeze(dim=0)


    ## omission
    omission_words, omission_indices, percent = omission(attn,indices,ind2c,w2ind,list_text,data,target,model,gpu=args.gpu)

    ## highlight on raw_text
    splited_text = raw_text.split()
    omission_words = list(omission_words)
    if args.latex:
        omission_result = " ".join(latexhl(splited_text[t]) if t in omission_indices else splited_text[t] for t in range(len(splited_text)) )

    else:
        omission_result = " ".join(colored(splited_text[t],"white","on_blue") if t in omission_indices else splited_text[t] for t in range(len(splited_text)) )
    print(omission_result)

    print("Predicted ICD codes:",codes)

    # percent of rationales
    print("Proportion of rationale: ", percent)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a neural network on some clinical documents")

    parser.add_argument("data_path", type=str,
                        help="path to a file containing sorted train data. dev/test splits assumed to have same name format with 'train' replaced by 'dev' and 'test'")
    parser.add_argument("vocab", type=str, help="path to a file holding vocab word list for discretizing words")
    parser.add_argument("Y", type=str, help="size of label space, 50 or full")

    parser.add_argument("model", type=str, choices=["caml"], help="model")

    parser.add_argument("--embed-file", type=str, required=False, dest="embed_file",
                        help="path to a file holding pre-trained embeddings")
    parser.add_argument("--test-model", type=str, dest="test_model", required=False, help="path to a saved model to load and evaluate")


    # CNN
    parser.add_argument("--embed-size", type=int, required=False, dest="embed_size", default=100, help="size of embedding dimension. (default: 100)")

    parser.add_argument("--num-filter-maps", type=int, required=False, dest="num_filter_maps", default=50,
                        help="size of conv output (default: 50)")

    parser.add_argument("--filter-size", type=str, required=False, dest="filter_size", default=3,
                        help="size of convolution filter to use. (default: 3) For multi_conv_attn, give comma separated integers, e.g. 3,4,5")
    parser.add_argument("--dropout", dest="dropout", type=float, required=False, default=0.5,
                        help="optional specification of dropout (default: 0.5)") 
       
    parser.add_argument("--index",dest="index",required=False,type=int,default=100,help="which row of text you want to infer")
    parser.add_argument("--gpu", dest="gpu", action="store_const", required=False, const=True,
                        help="optional flag to use GPU if available")

    parser.add_argument("--latex",dest="latex",action="store_const",const=False,required=False,help="print highlight texts in latex")
    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    main(args)


