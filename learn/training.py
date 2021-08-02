"""
    Main training code. Loads data, builds the model, trains, tests, evaluates, writes outputs, etc.
"""
import sys
# add the path to the repo, if needed
# sys.path.append('path')
import pdb
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import csv
import argparse
import os 
import numpy as np
import operator
import random

import time
from tqdm import tqdm
from collections import defaultdict
from torch.optim.lr_scheduler import ReduceLROnPlateau

from constants import *
import datasets
import evaluation
import interpret
import persistence
import learn.tools as tools


def main(args):
    start = time.time()
    args, model, optimizer, params, dicts, scheduler = init(args)
    epochs_trained = train_epochs(args, model, optimizer, params, dicts, scheduler)
    print("TOTAL ELAPSED TIME FOR %s MODEL AND %d EPOCHS: %f" % (args.model, epochs_trained, time.time() - start))

def init(args):
    """
        Load data, build model, create optimizer, create vars to hold metrics, etc.
    """
    #need to handle really large text fields
    csv.field_size_limit(sys.maxsize)

    #load vocab and other lookups
    print("loading lookups...")
    dicts = datasets.load_lookups(args)

    model = tools.pick_model(args, dicts)
    print(model)
    
    if not args.test_model:
        if args.optimizer == "Adam":        
            optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        elif args.optimizer == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        # learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, 'max',patience=5,factor=0.1)
    else:
        optimizer = None
        scheduler = None

    params = tools.make_param_dict(args)
    
    return args, model, optimizer, params, dicts, scheduler

def train_epochs(args, model, optimizer, params, dicts,scheduler):
    """
        Main loop. does train and test
    """
    metrics_hist = defaultdict(lambda: [])
    metrics_hist_te = defaultdict(lambda: [])
    metrics_hist_tr = defaultdict(lambda: [])
    metrics_hist_reg_tr = defaultdict(lambda: []) 
    metrics_hist_psel = defaultdict(lambda: [])
    metrics_hist_wordloss = defaultdict(lambda: [])

    test_only = args.test_model is not None
    evaluate = args.test_model is not None
    #train for n_epochs unless criterion metric does not improve for [patience] epochs
    for epoch in range(args.n_epochs):
        #only test on train/test set on very last epoch
        if epoch == 0 and not args.test_model:
            model_dir = os.path.join(MODEL_DIR, '_'.join([args.model, time.strftime('%b_%d_%H:%M:%S', time.localtime())]))
            os.mkdir(model_dir)
        elif args.test_model:
            model_dir = os.path.dirname(os.path.abspath(args.test_model))
 
        metrics_all = one_epoch(model, optimizer, args.Y, epoch, args.n_epochs, args.batch_size, args.data_path,
                                                  args.version, test_only, dicts, model_dir, 
                                                  args.gpu, args.quiet)
        
        score = metrics_all[0]["f1_micro"]
        if scheduler is not None:
            scheduler.step(score)

        for name in metrics_all[0].keys():
            metrics_hist[name].append(metrics_all[0][name])
        for name in metrics_all[1].keys():
            metrics_hist_te[name].append(metrics_all[1][name])
        for name in metrics_all[2].keys():
            metrics_hist_tr[name].append(metrics_all[2][name])
        for name in metrics_all[3].keys():
            metrics_hist_reg_tr[name].append(metrics_all[3][name])


        metrics_hist_all = (metrics_hist, metrics_hist_te, metrics_hist_tr,metrics_hist_reg_tr,metrics_hist_psel)
      
        #save metrics, model, params
        persistence.save_everything(args, metrics_hist_all, model, model_dir, params, args.criterion, evaluate)

        if test_only:
            #we're done
            break

        if args.criterion in metrics_hist.keys():
            if early_stop(metrics_hist, args.criterion, args.patience):
                #stop training, do tests on test and train sets, and then stop the script
                print("%s hasn't improved in %d epochs, early stopping..." % (args.criterion, args.patience))
                test_only = True
                args.test_model = '%s/model_best_%s.pth' % (model_dir, args.criterion)
                model = tools.pick_model(args, dicts)
    return epoch+1

def early_stop(metrics_hist, criterion, patience):
    if not np.all(np.isnan(metrics_hist[criterion])):
        if len(metrics_hist[criterion]) >= patience:
            if criterion == 'loss_dev': 
                return np.nanargmin(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
            else:
                return np.nanargmax(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
    else:
        #keep training if criterion results have all been nan so far
        return False
        
def one_epoch(model, optimizer, Y, epoch, n_epochs, batch_size, data_path, version, testing, dicts, model_dir, 
              gpu, quiet):
    """
        Wrapper to do a training epoch and test on dev
    """
    if not testing:
        losses, pseles, unseen_code_inds = train(model, optimizer, Y, epoch, batch_size, data_path, gpu, version, dicts, quiet)
        # pdb.set_trace()
        loss = torch.mean(torch.stack(losses))
        
        psel = torch.mean(torch.stack(pseles))


        print("epoch loss: " + str(loss))
        # print("epoch regularizer loss: "+ str(reg_loss))
        print("epoch selection probability: " + str(psel))

    else:
        loss = np.nan
        reg_loss = np.nan
        psel = np.nan
        word_loss = np.nan
        unseen_code_inds = set()


    fold = 'test' if version == 'mimic2' else 'dev'
    if epoch == n_epochs - 1:
        print("last epoch: testing on test and train sets")
        testing = True
        quiet = False

    #test on dev
    metrics = test(model, Y, epoch, data_path, fold, gpu, version, unseen_code_inds, dicts, model_dir,
                   testing)
                   

                   
    if testing or epoch == n_epochs - 1:
        print("\nevaluating on test")
        metrics_te = test(model, Y, epoch, data_path, "test", gpu, version, unseen_code_inds, dicts, 
                          model_dir, True)
    else:
        metrics_te = defaultdict(float)
        fpr_te = defaultdict(lambda: [])
        tpr_te = defaultdict(lambda: [])

    if type(loss) == float:
        metrics_tr = {"loss":loss}

    else:
        metrics_tr = {'loss': loss.item()}

    if type(psel) == float:
        metrics_psel_tr = {"psel_loss":psel}

    else:
        metrics_psel_tr = {'psel_loss': psel.item()}


    metrics_all = (metrics, metrics_te, metrics_tr, metrics_psel_tr)
    return metrics_all


def train(model, optimizer, Y, epoch, batch_size, data_path, gpu, version, dicts, quiet):
    """
        Training loop.
        output: losses for each example for this iteration
    """


    print("EPOCH %d" % epoch)

    model.train()

    num_labels = len(dicts['ind2c'])

    losses = [] 
    pseles = []


    #how often to print some info to stdout
    print_every = 25
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    unseen_code_inds = set(ind2c.keys())

    gen = datasets.data_generator(data_path, dicts, batch_size, num_labels, version=version)


    for batch_idx, tup in tqdm(enumerate(gen)):
        data, target, _, code_set, descs = tup
        data, target = Variable(torch.LongTensor(data)), Variable(torch.FloatTensor(target))
        # pdb.set_trace()
        unseen_code_inds = unseen_code_inds.difference(code_set)

        if gpu:
            data = data.cuda()
            target = target.cuda()
        
           
        output,mp,loss,mask_loss,psel = model(data,target)
        psel = torch.mean(psel)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)
        pseles.append(psel)


        if not quiet and batch_idx % print_every == 0:            
            print("Train epoch: {} [batch #{}, batch_size {}, seq length {}]\tTotal Loss: {:.6f}\tpsel: {:.6f}".format(
            epoch, batch_idx, data.size()[0], data.size()[1], torch.mean(torch.stack(losses[-10:])),
            torch.mean(torch.stack(pseles[-10:])) ))
                    
    
    return losses,pseles,unseen_code_inds 

def unseen_code_vecs(model, code_inds, dicts, gpu):
    """
        Use description module for codes not seen in training set.
    """
    code_vecs = tools.build_code_vecs(code_inds, dicts)
    code_inds, vecs = code_vecs
    #wrap it in an array so it's 3d
    desc_embeddings = model.embed_descriptions([vecs], gpu)[0]
    #replace relevant final_layer weights with desc embeddings 
    model.final.weight.data[code_inds, :] = desc_embeddings.data
    model.final.bias.data[code_inds] = 0

def test(model, Y, epoch, data_path, fold, gpu, version, code_inds, dicts, model_dir, testing):
    """
        Testing loop.
        Returns metrics
    """

    filename = data_path.replace('train', fold)
    print('file for evaluation: %s' % filename)
    num_labels = len(dicts['ind2c'])



    y, yhat, yhat_raw, hids, losses,pseles = [], [], [], [], [], []
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']


    model.eval()


    gen = datasets.data_generator(filename, dicts, 1, num_labels, version=version)
    for batch_idx, tup in tqdm(enumerate(gen)):
        data, target, hadm_ids, _, descs = tup
        with torch.no_grad():
            data, target = Variable(torch.LongTensor(data)), Variable(torch.FloatTensor(target))
                
            if gpu:
                data = data.cuda()
                target = target.cuda()
            
            
            output,attn,loss,mask_loss,psel = model(data,target)


            output = torch.sigmoid(output)
            output = output.data.cpu().numpy()


            losses.append(loss)
            pseles.append(psel.data)

            target_data = target.data.cpu().numpy()

            #save predictions, target, hadm ids
            yhat_raw.append(output)
            output = np.round(output)
            y.append(target_data)
            yhat.append(output)
            hids.extend(hadm_ids)

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)

    #write the predictions
    preds_file = persistence.write_preds(yhat, model_dir, hids, fold, ind2c, yhat_raw)
    #get metrics
    k = 5 if num_labels == 50 else [8,15]
    metrics = evaluation.all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)
    evaluation.print_metrics(metrics)
    metrics['loss_%s' % fold] = torch.mean(torch.stack(losses)).item()
    metrics['psel_%s' % fold] = torch.mean(torch.stack(pseles)).item()
    return metrics


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a neural network on some clinical documents")
    parser.add_argument("data_path", type=str,
                        help="path to a file containing sorted train data. dev/test splits assumed to have same name format with 'train' replaced by 'dev' and 'test'")
    parser.add_argument("vocab", type=str, help="path to a file holding vocab word list for discretizing words")
    parser.add_argument("Y", type=str, help="size of label space")

    parser.add_argument("model", type=str, choices=["cnn_vanilla", "laat","jointlaat", "conv_attn", 
                        "multi_conv_attn", "logreg","encaml","caml","cnnmaxpooling",
                        "bsm_maxpooling","bsm_caml","bsm_encaml","bsm_laat"], help="model")

    parser.add_argument("n_epochs", type=int, help="number of epochs to train")


    parser.add_argument("--embed-file", type=str, required=False, dest="embed_file",
                        help="path to a file holding pre-trained embeddings")
                    
    parser.add_argument("--embed-size", type=int, required=False, dest="embed_size", default=100,
                        help="size of embedding dimension. (default: 100)")

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
    parser.add_argument("--dataset", type=str, choices=['mimic3'], dest="version", default='mimic3', required=False,
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

    # LSTM
    parser.add_argument("--lstm-hidden-size",type=int,required=False,dest="lstm_hidden_size",default=512)
    parser.add_argument("--lstm-classifier-hidden-size",type=int,required=False,dest="lstm_classifier_hidden_size",default=512)
    parser.add_argument("--lstm-project-size",type=int,required=False,dest="lstm_project_size",default=128)

    # argument for BSM
    parser.add_argument("--lambda-p",dest="lambda_p",required=False,type=float, default=0.30, help="limitation of percent of words")
    parser.add_argument("--lambda-sel",dest="lambda_sel",required=False,type=float, default=0.04, help="selection regularizer")
    parser.add_argument("--lambda-cont",dest="lambda_cont",required=False,type=float, default=0.04, help="continuity regularizer")
    parser.add_argument("--tau",dest="tau",required=False,type=float,default=0.8,help="temperature in gumble softmax")

    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    main(args)










