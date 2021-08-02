"""
Given a data set, open one model, and find the words it removed to 
make all predicted labels change (the switching point) for each discharge summary in that data set
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
from tqdm import tqdm
import numpy as np
from termcolor import colored
import pdb
import json
import learn.tools as tools
from constants import *


def init(args):
    """
    init the model, dicts, data set
    """

    print("loading lookups...")
    dicts = datasets.load_lookups(args)
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    num_labels = len(ind2c)


    print("loading model...")
    model = tools.pick_model(args, dicts) #remember to set as test mode
    print(model)

    
    print("loading data...")
    dataset = pd.read_csv(args.omission_data_path)
    dataset = dataset.iterrows() # generator
    return dicts, model, dataset,num_labels


def omission_file(dicts,model,dataset,num_labels):
    avrg_percent = 0
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    result_dict = dict()



    for index, row in tqdm(enumerate(dataset)):
        
        text = row[1]["TEXT"]
        target = row[1]["LABELS"]

        # convert to tensor
        tensor_text = [int(w2ind[w]) if w in w2ind else len(w2ind)+1 for w in text.split()]
        tensor_text = Variable(torch.LongTensor(tensor_text)).view(1,-1)
        if len(tensor_text) > MAX_LENGTH:
            tensor_text = tensor_text[:MAX_LENGTH]
        labels_idx = np.zeros(num_labels)
        for l in target:
            if l in c2ind.keys():
                code = int(c2ind[l])
                labels_idx[code] = 1
        labels_idx = Variable(torch.FloatTensor(labels_idx)).view(1,-1)

        if args.gpu:
            tensor_text = tensor_text.cuda()
            labels_idx = labels_idx.cuda()

            # make predictions 
        with torch.no_grad():
            y_hat, attn, loss,a,b,c = model(tensor_text,labels_idx)
        y_hat = torch.sigmoid(y_hat)
        y_hat = (y_hat>0.5).float()

        codes = [ind2c[i] for i in np.where(y_hat.cpu()==1)[1]] 

        ## variables used for omission
        list_text = text.split()
        indices = list(np.where(y_hat.cpu()==1)[1])
        # pdb.set_trace()
        attn = attn.squeeze(dim=0)
        # pdb.set_trace()

        omission_words,omission_indices, percent = omission(attn,indices,ind2c,w2ind,list_text,tensor_text,labels_idx,model,gpu=args.gpu)
        result_dict[index] = percent
        avrg_percent += percent
        with open(args.output_json,"w") as save:
            json.dump(result_dict,save)

    # print("saving all the omission percentage...")
    # with open(args.output_json,"w") as file:
    #     json.dump(result_dict,file)

    return avrg_percent/(index+1)

def main(args):
    dicts, model, dataset,num_labels = init(args)

    avrg_percent = omission_file(dicts, model, dataset,num_labels)
    print("The average omission percentage is: "+str(avrg_percent))



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
    if attn.shape[1] == 50:
        attn = attn.view(50,-1)
    omission_words = set() # record the words deleted 
    omission_indices = set() # record the indices of deleted words
    if type(attn) == list: # for encaml
        pass 

    else:
        
        for index in indices:
            # print("The code:",ind2c[index],"is now under evaluating......")
            # pdb.set_trace()

            attn_words = attn[index,:].view(-1) # get the attention weights for that label.

            attn_rank = list(attn_words.topk(len(attn_words))[1].cpu().numpy())  # get the indices of each word's weights from high to low
            # if len(text) != len(attn_rank):
            #     print("there is a bug")
            
            data_ = data.detach().clone() # due to overlapps, we need to initialize the data for each label;

            # start to delete words in text:
            for i in range(len(text)):

                
                # deleted the words
                # pdb.set_trace()
                # print(i)
                data_ = torch.cat((data_[:,:attn_rank[i]-i], data_[:,attn_rank[i]+1-i:]),dim=1)

                if gpu:
                    data_ = data_.cuda()
                
                if attn_rank[i] < len(text):
                    omission_words.add(text[attn_rank[i]] if text[attn_rank[i]] in list(w2ind.keys()) else "unk")
                    omission_indices.add(attn_rank[i])


                with torch.no_grad():
                    output,alpha_,loss,a,b,c = model(data_,target)
                
                indices_ = list(np.where(torch.sigmoid(output).cpu()>0.5)[1])

                if index not in indices_:
                    # print("Finished evaluation of code:",ind2c[index])
                    break
        # pdb.set_trace()
        percent = len(omission_indices)/len(text)

        
    return omission_words,omission_indices, percent



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a neural network on some clinical documents")
    parser.add_argument("data_path",type=str,help="the training data set for full/top 50 labels")
    parser.add_argument("omission_data_path", type=str,
                        help="the file you want to omit")
    parser.add_argument("vocab", type=str, help="path to a file holding vocab word list for discretizing words")
    parser.add_argument("Y", type=str, help="size of label space, 50 or full")

    parser.add_argument("model", type=str, choices=["caml","laat"], help="model")

    parser.add_argument("--embed-file", type=str, required=False, dest="embed_file",
                        help="path to a file holding pre-trained embeddings")

    parser.add_argument("--output-json",dest="output_json",type=str,required=True)
    parser.add_argument("--dataset", type=str, choices=['mimic2', 'mimic3'], dest="version", default='mimic3', required=False,
                        help="version of MIMIC in use (default: mimic3)")

    # CNN
    parser.add_argument("--embed-size", type=int, required=False, dest="embed_size", default=100, help="size of embedding dimension. (default: 100)")

    parser.add_argument("--test-model", type=str, dest="test_model", required=False, help="path to a saved model to load and evaluate")

    parser.add_argument("--num-filter-maps", type=int, required=False, dest="num_filter_maps", default=50,
                        help="size of conv output (default: 50)")

    parser.add_argument("--filter-size", type=str, required=False, dest="filter_size", default=3,
                        help="size of convolution filter to use. (default: 3) For multi_conv_attn, give comma separated integers, e.g. 3,4,5")
    
    parser.add_argument("--dropout", dest="dropout", type=float, required=False, default=0.5,
                        help="optional specification of dropout (default: 0.5)")    
    parser.add_argument("--index",dest="index",required=False,type=int,default=100,help="which row of text you want to infer")
    parser.add_argument("--gpu", dest="gpu", action="store_const", required=False, const=True,
                        help="optional flag to use GPU if available")
    
    
    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    main(args)