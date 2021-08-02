"""
    Holds PyTorch models: CAML, EnCAML, LAAT, MaxPooling
"""
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ 
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
import pdb
from math import floor


from constants import *
from dataproc import extract_wvs

class Base(nn.Module):

    def __init__(self, Y, embed_file, dicts,  dropout=0.5, gpu=True, embed_size=100):
        super(Base, self).__init__()
        torch.manual_seed(1337)
        self.gpu = gpu
        self.Y = Y
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=dropout)


        #make embedding layer
        if embed_file:
            print("loading pretrained embeddings...")
            W = torch.Tensor(extract_wvs.load_embeddings(embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embed.weight.data = W.clone()
        else:
            #add 2 to include UNK and PAD
            vocab_size = len(dicts['ind2w'])
            self.embed = nn.Embedding(vocab_size+2, embed_size, padding_idx=0)
            

    def _get_loss(self, yhat, target):
        #calculate the BCE
        loss = F.binary_cross_entropy_with_logits(yhat, target)

        return loss


class CAML(Base):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, gpu, dicts, embed_size=100, dropout=0.5):
        super(CAML, self).__init__(Y, embed_file, dicts, dropout=dropout, gpu=gpu, embed_size=embed_size)

        #initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=int(floor(kernel_size/2)))
        xavier_uniform_(self.conv.weight)

        #context vectors for computing attention as in 2.2
        self.U = nn.Linear(num_filter_maps, Y)
        xavier_uniform_(self.U.weight)

        #final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(num_filter_maps, Y)
        xavier_uniform_(self.final.weight)

        
    def forward(self, x, target):
        #get embeddings and apply dropout
        # pdb.set_trace()
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        #apply convolution and nonlinearity (tanh)
        x = torch.tanh(self.conv(x).transpose(1,2))
        #apply attention
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1,2)), dim=2)
        #document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)
        #final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        
        #final sigmoid to get predictions
        yhat = y
        loss = self._get_loss(yhat, target)
        return yhat,alpha, loss, torch.zeros(1),torch.zeros(1),torch.zeros(1)



class EnCAML(Base):
    
    def __init__(self,num_codes,embed_file,dicts,num_filter_maps=64,dropout=0.5,gpu=False,embed_size=100):
        super(EnCAML, self).__init__(num_codes, embed_file, dicts, dropout=dropout, gpu=gpu, embed_size=embed_size)

        torch.manual_seed(1234)

        # we're gonna use four different kernel size: 3, 5, 7, and 9
        self.conv3 = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=3, padding=int(floor(3/2)))
        self.conv5 = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=5, padding=int(floor(5/2)))
        self.conv7 = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=7, padding=int(floor(7/2)))
        self.conv9 = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=9, padding=int(floor(9/2)))

        xavier_uniform_(self.conv3.weight)
        xavier_uniform_(self.conv5.weight)
        xavier_uniform_(self.conv7.weight)
        xavier_uniform_(self.conv9.weight)

        ## four conv1d layer share the same per-label attention weights
        self.U3 = nn.Linear(num_filter_maps,num_codes)
        xavier_uniform_(self.U3.weight)
        self.U5 = nn.Linear(num_filter_maps,num_codes)
        xavier_uniform_(self.U5.weight)
        self.U7 = nn.Linear(num_filter_maps,num_codes)
        xavier_uniform_(self.U7.weight)
        self.U9 = nn.Linear(num_filter_maps,num_codes)
        xavier_uniform_(self.U9.weight)
        ## final feed forward layer
        self.final = nn.Linear(num_filter_maps*4,num_codes)
        xavier_uniform_(self.final.weight)

    def forward(self,x,y):
        # embedding
        # pdb.set_trace()
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1,2)

        # convolution
        x3 = torch.tanh(self.conv3(x).transpose(1,2))
        x5 = torch.tanh(self.conv5(x).transpose(1,2))
        x7 = torch.tanh(self.conv7(x).transpose(1,2))
        x9 = torch.tanh(self.conv9(x).transpose(1,2))

        # per-label attention, share the attention weights
        alpha3 = F.softmax(self.U3.weight.matmul(x3.transpose(1,2)),dim=2)
        alpha5 = F.softmax(self.U5.weight.matmul(x5.transpose(1,2)),dim=2)
        alpha7 = F.softmax(self.U7.weight.matmul(x7.transpose(1,2)),dim=2)
        alpha9 = F.softmax(self.U9.weight.matmul(x9.transpose(1,2)),dim=2)

        # attention on the convolution results
        m3 = alpha3.matmul(x3)
        m5 = alpha5.matmul(x5)
        m7 = alpha7.matmul(x7)
        m9 = alpha9.matmul(x9)

        m = torch.cat((m3,m5,m7,m9),dim=2)
        y_hat = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        loss = self._get_loss(y_hat,y)

        alpha = [alpha3, alpha5, alpha7, alpha9]
        return y_hat, alpha, loss, torch.zeros(1),torch.zeros(1),torch.zeros(1)

class LAAT(Base):
    def __init__(self,num_codes,embed_file, dicts, hidden_size, project_size,dropout,gpu=True,embed_size=100):
        super(LAAT, self).__init__(num_codes, embed_file, dicts, dropout=dropout, gpu=gpu, embed_size=embed_size)

        torch.manual_seed(1234)

        self.bidirectional = True # bool
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.num_directions = int(self.bidirectional) + 1

        self.project_size = project_size

        # self.batch_size = batch_size
        self.dropout = dropout
        self.gpu = gpu


        self.rnn = nn.LSTM(self.embed_size,hidden_size,self.num_layers,
                        bidirectional=self.bidirectional,
                        dropout=self.dropout if self.num_layers >1 else 0,
                        batch_first=True)

        ## attention part
        self.W = nn.Linear(self.hidden_size*2,self.project_size)
        self.U = nn.Linear(self.project_size,num_codes) # remember here, see if we could predict first three then predict the last two
        self.O = nn.Linear(self.hidden_size*2,num_codes) # final output

        xavier_uniform_(self.W.weight)
        xavier_uniform_(self.U.weight)
        xavier_uniform_(self.O.weight)


    def init_hidden(self,batch_size):
        h = Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))
        c = Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))
        # if self.rnn_model.lower() == "gru":
        #     return h
        if self.gpu:
            h = h.cuda()
            c = c.cuda()
        return h, c


    def forward(self,x,y):
        # pdb.set_trace()
        batch_size = x.shape[0]
        with torch.no_grad():
            lengths = torch.count_nonzero(x,dim=-1).cpu()
        x = self.embed(x)
        x = self.embed_drop(x)
        pack = pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
        
        h0,c0 = self.init_hidden(batch_size)
        output, hidden = self.rnn(pack,(h0,c0))
        output = pad_packed_sequence(output)[0].transpose(0,1)
        
        Z = torch.tanh(self.W(output))
        A = torch.softmax(self.U(Z),dim=1)
        V = output.transpose(1,2).matmul(A)
        y_hat = self.O.weight.mul(V.transpose(1,2)).sum(dim=2).add(self.O.bias)
        loss = self._get_loss(y_hat,y)

        loss = self._get_loss(y_hat,y)
      
        return y_hat,A, loss, torch.zeros(1),torch.zeros(1),torch.zeros(1)


class CNNMaxPooling(Base):
    
    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, gpu=True, dicts=None, embed_size=100, 
                    dropout=0.5):
        super(CNNMaxPooling, self).__init__(Y, embed_file, dicts, dropout=dropout, embed_size=embed_size) 
        #initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size)
        xavier_uniform_(self.conv.weight)

        #linear output
        self.fc = nn.Linear(num_filter_maps, Y)
        xavier_uniform_(self.fc.weight)

    def forward(self, x, target):
        #embed
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        #conv/max-pooling
        c = self.conv(x)

        # x = F.max_pool1d(torch.tanh(c), kernel_size=c.size()[2])
        # attn = None
        # x = x.squeeze(dim=2)

        # #linear output
        # x = self.fc(x)


        # another way
        attn = None
        # pdb.set_trace()
        x = self.fc(c.transpose(1,2)) 
        yhat = F.max_pool1d(torch.tanh(x.transpose(1,2)),kernel_size=x.size()[1])
        yhat = yhat.squeeze(dim=2)

        #final sigmoid to get predictions
        # yhat = x
        loss = self._get_loss(yhat, target)
        return yhat, attn, loss, torch.zeros(1),torch.zeros(1),torch.zeros(1)

    

