import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import pdb
from math import floor
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from constants import *
from dataproc import extract_wvs

from learn import hdwmodels


class Base(nn.Module):
    def __init__(self, 
                num_codes, 
                embed_file, 
                dicts,
                dropout=0.5, 
                gpu=False, 
                embed_size=100,
                lambda_p=0.5,
                lstm_hidden_size=64,
                loss_weights=[1,0.02,0.02]):

        super(Base,self).__init__()

        self.weights = loss_weights
        

        self.num_codes = num_codes
        self.gpu = gpu
        self.embed_drop = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lambda_p = lambda_p
        
        self.weights = torch.FloatTensor(loss_weights) # tune the loss of different parts 

        ## make embedding layer
        if embed_file:
            print("loading pretrained embeddings...")
            W = torch.Tensor(extract_wvs.load_embeddings(embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embed.weight.data = W.clone()
        else:
            #add 2 to include UNK and PAD
            vocab_size = len(dicts['ind2w'])
            self.embed = nn.Embedding(vocab_size+2, embed_size, padding_idx=0)

        if self.gpu:
            self.weights = self.weights.cuda()

        self.lstm = nn.LSTM(self.embed_size, lstm_hidden_size,num_layers=1,batch_first=True,bidirectional=True)

        self.masker = nn.Linear(2*self.lstm_hidden_size,2) # binary, need or not
        xavier_uniform_(self.masker.weight)
        

    def _get_loss(self, yhat, target, diffs=None):
        #calculate the BCE
        # for safety, we make sure that all data on the same device
        if self.gpu:
            yhat = yhat.cuda()
            target = target.cuda()
        else:
            yhat = yhat.cpu()
            target = target.cpu()
            
        loss = F.binary_cross_entropy_with_logits(yhat, target)
        return loss


 

    def disc_loss(self,mask):
        # pdb.set_trace()
        max_len = mask.shape[1]

        disc = torch.abs(mask[:,1:,:]-mask[:,:-1,:])
        disc = disc.sum(dim=-1).sum(dim=-1)/(max_len*2)

        batch_size = mask.shape[0]
        zero = torch.zeros(batch_size)
        if self.gpu:
            zero = zero.cuda()
        return F.binary_cross_entropy(disc,zero)

    def soft_mask_loss(self,mask,lambda_p):

        batch_size = mask.shape[0]
        max_len = mask.shape[1]

        t0 = mask[:,:,0]
        psel = (1-t0).sum(dim=-1)/max_len

        lambda_p = [lambda_p for i in range(batch_size)]

        lambda_p = torch.FloatTensor(lambda_p)
        
        if self.gpu:
            lambda_p = lambda_p.cuda()

        lsel = F.binary_cross_entropy(psel,lambda_p)

        return lsel, psel

    def init_hidden(self,batch_size): # init the hidden in masker if lstm
        h = Variable(torch.zeros(2, batch_size, self.lstm_hidden_size))
        c = Variable(torch.zeros(2, batch_size, self.lstm_hidden_size))
        if self.gpu:
            h = h.cuda()
            c = c.cuda()
        return h, c


class CAML(Base):
    def __init__(self, label_space, embed_file, dicts, 
                dropout, 
                lstm_hidden_size,
                num_filter_maps, 
                kernel_size=9, 
                gpu=False,
                lambda_p=0.50, 
                tau=0.8, 
                loss_weights=[1,0.02,0.02]):

        super(CAML,self).__init__(label_space,embed_file,dicts,dropout,gpu,
                                    lambda_p=lambda_p,
                                    lstm_hidden_size=lstm_hidden_size,
                                    loss_weights=lossl_weights)
        self.label_space = label_space
        self.num_filter_maps = num_filter_maps
        self.tau = tau

        self.conv = nn.Conv1d(self.embed_size,num_filter_maps,kernel_size,padding=int(kernel_size//2))
        ## attentin layer
        self.U = nn.Linear(num_filter_maps,label_space)
        ## per-label output layer
        self.final = nn.Linear(num_filter_maps,label_space)

        xavier_uniform_(self.conv.weight)
        xavier_uniform_(self.U.weight)
        xavier_uniform_(self.final.weight)
    
    def forward(self,x_,target):
        batch_size = x_.shape[0]
        with torch.no_grad():
            lengths = torch.count_nonzero(x_,dim=-1).cpu()
        x = self.embed(x_)
        x = self.embed_drop(x)

        # lstm
        pack = pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
        h0,c0 = self.init_hidden(batch_size)
        hidden, _ = self.lstm(pack,(h0,c0))
        hidden = pad_packed_sequence(hidden)[0].transpose(0,1)

        # soft mask
        mask = F.log_softmax(self.masker.weight.matmul(hidden.transpose(1,2)),dim=-1)
        mask = mask.transpose(1,2)
        # gumble-softmax sample 
        mask = F.gumbel_softmax(mask,tau=self.tau,hard=False,dim=-1) #[B,L,2]
        mask_ = mask.detach().clone()  # use for the loss


        x1 = torch.cat((x,x),dim=-1)
        mask1 = mask[:,:,0]
        mask2 = mask[:,:,1]
        mask1 = mask1.unsqueeze(dim=-1).repeat(1,1,self.embed_size)
        mask2 = mask2.unsqueeze(dim=-1).repeat(1,1,self.embed_size)
        mask = torch.cat((mask1,mask2),dim=-1)
        embed_mask = x1*mask
        embed_mask = embed_mask[:,:,self.embed_size:]
        embed_mask = embed_mask.transpose(1,2)

        # CAML-like structure
        conv = torch.tanh(self.conv(embed_mask).transpose(1,2)) #[B,L,D]
        # attention
        alpha = F.softmax(self.U.weight.matmul(conv.transpose(1,2)),dim=2)
        # m
        m = alpha.matmul(conv)
        y_hat = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        ## mask loss 
        mask_loss,psel = self.soft_mask_loss(mask_,self.lambda_p)
        dist_loss = self.disc_loss(mask_)   

        loss = self._get_loss(y_hat,target)*self.weights[0]+mask_loss*self.weights[1]+dist_loss*self.weights[2] # total loss
        return y_hat,mask_,loss,mask_loss,psel
      


class EnCAML(Base):
    def __init__(self,label_space, embed_file,dicts, dropout,
                lstm_hidden_size,
                num_filter_maps,
                gpu=False,
                lambda_p=0.50,
                tau=0.8,
                lambda_s=0.4,
                loss_weights=[1,0.02,0.02]):

        super(EnCAML,self).__init__(label_space,embed_file,dicts,dropout,gpu,lambda_p=lambda_p,
                                    lambda_s=lambda_s,lstm_hidden_size=lstm_hidden_size,loss_weights=loss_weights)
        self.label_space = label_space
        self.num_filter_maps = num_filter_maps
        self.tau = tau


        # multi convolution
        self.conv3 = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=3, padding=int(floor(3/2)))
        self.conv5 = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=5, padding=int(floor(5/2)))
        self.conv7 = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=7, padding=int(floor(7/2)))
        self.conv9 = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=9, padding=int(floor(9/2)))


        ## attentin layer
        self.U3 = nn.Linear(num_filter_maps,label_space)
        self.U5 = nn.Linear(num_filter_maps,label_space)
        self.U7 = nn.Linear(num_filter_maps,label_space)
        self.U9 = nn.Linear(num_filter_maps,label_space)

        ## final feed forward layer
        self.final = nn.Linear(num_filter_maps*4,label_space)

        xavier_uniform_(self.conv3.weight)
        xavier_uniform_(self.conv5.weight)
        xavier_uniform_(self.conv7.weight)
        xavier_uniform_(self.conv9.weight)    
        xavier_uniform_(self.U3.weight)    
        xavier_uniform_(self.U5.weight)    
        xavier_uniform_(self.U7.weight)
        xavier_uniform_(self.U9.weight)
        xavier_uniform_(self.final.weight)

    def forward(self,x_,target):
        # pdb.set_trace()
        batch_size = x_.shape[0]
        with torch.no_grad():
            lengths = torch.count_nonzero(x_,dim=-1).cpu()
        x = self.embed(x_)
        x = self.embed_drop(x)

        # lstm
        pack = pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
        h0,c0 = self.init_hidden(batch_size)
        hidden, _ = self.lstm(pack,(h0,c0))
        hidden = pad_packed_sequence(hidden)[0].transpose(0,1)

        # soft mask
        mask = F.log_softmax(self.masker(hidden),dim=-1)
        # gumble-softmax sample 
        mask = F.gumbel_softmax(mask,tau=self.tau,hard=False,dim=-1) #[B,L,2]
        mask_ = mask.detach().clone()  # use for the loss


        # mask word embedding
        x1 = torch.cat((x,x),dim=-1)
        mask1 = mask[:,:,0]
        mask2 = mask[:,:,1]
        mask1 = mask1.unsqueeze(dim=-1).repeat(1,1,self.embed_size)
        mask2 = mask2.unsqueeze(dim=-1).repeat(1,1,self.embed_size)
        mask = torch.cat((mask1,mask2),dim=-1)
        embed_mask = x1*mask
        embed_mask = embed_mask[:,:,self.embed_size:]
        embed_mask = embed_mask.transpose(1,2)

        # EnCAML
        x3 = torch.tanh(self.conv3(embed_mask).transpose(1,2))
        x5 = torch.tanh(self.conv5(embed_mask).transpose(1,2))
        x7 = torch.tanh(self.conv7(embed_mask).transpose(1,2))
        x9 = torch.tanh(self.conv9(embed_mask).transpose(1,2))

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
        alpha = [alpha3, alpha5, alpha7, alpha9]

      
        ## mask loss 
        mask_loss,psel = self.soft_mask_loss(mask_,self.lambda_p)
        dist_loss = self.disc_loss(mask_)   

        loss = self._get_loss(y_hat,target)*self.weights[0]+mask_loss*self.weights[1]+dist_loss*self.weights[2] # total loss
        return y_hat,mask_,loss,mask_loss,psel



class LAAT(Base):
    def __init__(self,
                label_space,
                embed_file,
                dicts,
                dropout,
                masker_lstm_hidden_size,
                classifier_lstm_hidden_size,
                project_size,
                gpu=False,
                lambda_p=0.50,
                tau=0.8,
                loss_weights=[1,0.02,0.02]):
        super(LAAT,self).__init__(label_space,embed_file,dicts,dropout,gpu,
                                lambda_p=lambda_p,
                                lstm_hidden_size=masker_lstm_hidden_size,
                                loss_weights=loss_weights)


        self.label_space = label_space
        self.tau = tau
        self.classifier_lstm_hidden_size = classifier_lstm_hidden_size

        self.classifier_lstm = nn.LSTM(self.embed_size,classifier_lstm_hidden_size,1,
                                        bidirectional=True, batch_first=True)
         ## attention part
        self.W = nn.Linear(classifier_lstm_hidden_size*2,project_size)
        self.U = nn.Linear(project_size,self.label_space) # remember here, see if we could predict first three then predict the last two
        self.O = nn.Linear(classifier_lstm_hidden_size*2,self.label_space) # final output


        xavier_uniform_(self.W.weight)
        xavier_uniform_(self.U.weight)
        xavier_uniform_(self.O.weight)


    def init_classifier_hidden(self,batch_size):
        h = Variable(torch.zeros(2, batch_size, self.classifier_lstm_hidden_size))
        c = Variable(torch.zeros(2, batch_size, self.classifier_lstm_hidden_size))
        if self.gpu:
            h = h.cuda()
            c = c.cuda()
        return h, c      

    
    def forward(self,x_,target):
        # pdb.set_trace()
        batch_size = x_.shape[0]
        with torch.no_grad():
            lengths = torch.count_nonzero(x_,dim=-1).cpu()
        x = self.embed(x_)
        x = self.embed_drop(x)

        # lstm
        pack = pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
        h0,c0 = self.init_hidden(batch_size)

        hidden, _ = self.lstm(pack,(h0,c0))
        hidden = pad_packed_sequence(hidden)[0].transpose(0,1)

        # soft mask
        mask = F.log_softmax(self.masker(hidden),dim=-1)
        # gumble-softmax sample 
        mask = F.gumbel_softmax(mask,tau=self.tau,hard=False,dim=-1) #[B,L,2]
        mask_ = mask.detach().clone()  # use for the loss


        # mask word embedding
        # mask core
        x1 = torch.cat((x,x),dim=-1)
        mask1 = mask[:,:,0]
        mask2 = mask[:,:,1]
        mask1 = mask1.unsqueeze(dim=-1).repeat(1,1,self.embed_size)
        mask2 = mask2.unsqueeze(dim=-1).repeat(1,1,self.embed_size)
        mask = torch.cat((mask1,mask2),dim=-1)
        embed_mask = x1*mask
        embed_mask = embed_mask[:,:,self.embed_size:]
        
        emb_pack = pack_padded_sequence(embed_mask,lengths,batch_first=True,enforce_sorted=False)
        emb_h0,emb_c0 = self.init_classifier_hidden(batch_size)
        emb_output, emb_hidden = self.classifier_lstm(emb_pack,(emb_h0,emb_c0))
        emb_output = pad_packed_sequence(emb_output)[0].transpose(0,1)

        # LAAT
        Z = torch.tanh(self.W.weight.matmul(emb_output.transpose(1,2))).transpose(1,2)
        A = self.U.weight.matmul(Z.transpose(1,2)).transpose(1,2)
        V = emb_output.transpose(1,2).matmul(A).transpose(1,2)
        y_hat = self.O.weight.mul(V).sum(dim=2).add(self.O.bias)

        ## mask loss 
        mask_loss,psel = self.soft_mask_loss(mask_,self.lambda_p)
        dist_loss = self.disc_loss(mask_)   

        loss = self._get_loss(y_hat,target)*self.weights[0]+mask_loss*self.weights[1]+dist_loss*self.weights[2] # total loss
        return y_hat,mask_,loss,mask_loss,psel


class CNNMaxPooling(Base):
    def __init__(self, label_space, embed_file, dicts, dropout, 
                lstm_hidden_size, 
                num_filter_maps, 
                kernel_size=9, 
                gpu=False,
                lambda_p=0.50, 
                tau=0.8, 
                loss_weights=[1,0.02,0.02]):

        super(CNNMaxPooling,self).__init__(label_space,embed_file,dicts,dropout,gpu,lambda_p=lambda_p,
                                    lstm_hidden_size=lstm_hidden_size,
                                    loss_weights=loss_weights)
        self.label_space = label_space
        self.num_filter_maps = num_filter_maps
        self.tau = tau

        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size)
        

        #linear output
        self.fc = nn.Linear(num_filter_maps, label_space)

        xavier_uniform_(self.conv.weight)
        xavier_uniform_(self.fc.weight)


    def forward(self,x_,target):
        # pdb.set_trace()
        batch_size = x_.shape[0]
        with torch.no_grad():
            lengths = torch.count_nonzero(x_,dim=-1).cpu()
        x = self.embed(x_)
        x = self.embed_drop(x)

        # lstm
        pack = pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
        h0,c0 = self.init_hidden(batch_size)
        hidden, _ = self.lstm(pack,(h0,c0))
        hidden = pad_packed_sequence(hidden)[0].transpose(0,1)

        # soft mask
        mask = F.log_softmax(self.masker(hidden),dim=-1)
        # gumble-softmax sample 
        mask = F.gumbel_softmax(mask,tau=self.tau,hard=False,dim=-1) #[B,L,2]
        mask_ = mask.detach().clone()  # use for the loss


        # mask word embedding
        x1 = torch.cat((x,x),dim=-1)
        mask1 = mask[:,:,0]
        mask2 = mask[:,:,1]
        mask1 = mask1.unsqueeze(dim=-1).repeat(1,1,self.embed_size)
        mask2 = mask2.unsqueeze(dim=-1).repeat(1,1,self.embed_size)
        mask = torch.cat((mask1,mask2),dim=-1)
        embed_mask = x1*mask
        embed_mask = embed_mask[:,:,self.embed_size:]
        embed_mask = embed_mask.transpose(1,2)

        # CNN
        c = self.conv(embed_mask)

        mp = F.max_pool1d(torch.tanh(c), kernel_size=c.size()[2])
        attn = None
        y_hat = mp.squeeze(dim=2)

        #linear output
        y_hat = self.fc(y_hat)

        ## mask loss 
        mask_loss,psel = self.soft_mask_loss(mask_,self.lambda_p)
        dist_loss = self.disc_loss(mask_)
        loss = self._get_loss(y_hat,target)*self.weights[0]+mask_loss*self.weights[1]+dist_loss*self.weights[2] # total loss

        return y_hat,mask_,loss,mask_loss,psel

class Baseline(Base):
    def __init__(self, label_space, embed_file, dicts, dropout, 
                lstm_hidden_size, 
                num_filter_maps, 
                kernel_size=9, 
                gpu=False,
                lambda_p=0.50, 
                tau=0.8, 
                loss_weights=[1,0.02,0.02]):

        super(Baseline,self).__init__(label_space,embed_file,dicts,dropout,gpu,lambda_p=lambda_p,
                                    lstm_hidden_size=lstm_hidden_size,
                                    loss_weights=loss_weights)
        self.label_space = label_space
        self.num_filter_maps = num_filter_maps
        self.tau = tau

        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size)
        

        #linear output
        self.fc = nn.Linear(num_filter_maps, label_space)

        xavier_uniform_(self.conv.weight)
        xavier_uniform_(self.fc.weight)


    def forward(self,x_,target):
        # pdb.set_trace()
        batch_size = x_.shape[0]
        max_len = x_.shape[1]
        with torch.no_grad():
            lengths = torch.count_nonzero(x_,dim=-1).cpu()
        x = self.embed(x_)
        x = self.embed_drop(x)


        mask = torch.randn(batch_size,max_len,2)
        mask_ = mask.detach().clone()  # use for the loss


        # mask word embedding
        x1 = torch.cat((x,x),dim=-1)
        mask1 = mask[:,:,0]
        mask2 = mask[:,:,1]
        mask1 = mask1.unsqueeze(dim=-1).repeat(1,1,self.embed_size)
        mask2 = mask2.unsqueeze(dim=-1).repeat(1,1,self.embed_size)
        mask = torch.cat((mask1,mask2),dim=-1)
        embed_mask = x1*mask
        embed_mask = embed_mask[:,:,self.embed_size:]
        embed_mask = embed_mask.transpose(1,2)

        # CNN
        c = self.conv(embed_mask)

        mp = F.max_pool1d(torch.tanh(c), kernel_size=c.size()[2])
        attn = None
        y_hat = mp.squeeze(dim=2)

        #linear output
        y_hat = self.fc(y_hat)

        ## mask loss 
        mask_loss,psel = self.soft_mask_loss(mask_,self.lambda_p)
        dist_loss = self.disc_loss(mask_)
        loss = self._get_loss(y_hat,target)*self.weights[0]+mask_loss*self.weights[1]+dist_loss*self.weights[2] # total loss

        return y_hat,mask_,loss,mask_loss,psel
