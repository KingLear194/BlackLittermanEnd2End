#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 18:37:41 2023

@author: jed169
"""

import torch
import torch.nn as nn


class FFNStandardTorch(nn.Module):
    
    '''
    Creates the architecture of a Standard, fully connected FFN
    As an intermediate step it creates an ordered dict of model type, which is fed to the forward method of nn.
    
    Args:
        num_features (int): number of features for every observation, needs to be the same for 
        every observation
                
        hidden_units (list): number of units for each consecutive hidden layer; it has length equal to the number
        of hidden layers
        
        output_size (int): the dimension of the output of the network
        
        activation (str): activation used for all layers, default = 'SiLU'
        
        batchnormalization (bool): if True, adds a layer of Batchnormalization (BN) right after
        every activation. Note: even though the original paper about BN had BN before activation, since then,
        people have been mostly using it right after the activation.
        
        dropout (float): if not None, adds a dropout layer right after the last activation
        /right before the output layer, default = None
        
    Functions:
        create: returns the network, created from passing the argument; first creates an 
        ordered dict that is then passed to the forward method
        
        forward: does the forward pass 
    
        get_architecture: prints the architecture of the model on the console
        
    Note: Elastic-Net can be modeled with this if we do one layer without biases
        
    '''
    
    def __init__(self,num_features, 
                 hidden_units,
                 output_size,
                 activation = 'SiLU', 
                 bias = True,
                 batchnormalization = True, 
                 dropout = None,
                 world_size = 1):
        
        super(FFNStandardTorch, self).__init__()

        self.num_features = num_features
        self.hidden_units = hidden_units  #this will be a list which will contain the number of hidden units for each 
        self.output_size = output_size
        self.activation = getattr(nn, activation)
        self.bias = bias
        self.batchnormalization = batchnormalization
        self.dropout = dropout
        self.world_size = world_size
        
        self.od = nn.ModuleDict()
        self.od['input layer'] = nn.Linear(self.num_features,self.hidden_units[0], bias = self.bias)
        self.od[f'activation_{1}'] = self.activation()
        if self.batchnormalization:
                self.od[f'batchnorm_{1}'] = nn.BatchNorm1d(self.hidden_units[0])
        
        for i in range(1,len(self.hidden_units)):
            self.od[f'linear_{i}'] = nn.Linear(self.hidden_units[i-1],self.hidden_units[i], bias = self.bias)
            self.od[f'activation_{i+1}'] = self.activation()
            # optional dropout layer   
            if self.dropout is not None:
                self.od['dropout_{i+1}'] = nn.Dropout(self.dropout)
            # optional batchnorm layer
            if self.batchnormalization:
                if self.world_size >1:
                    self.od[f'batchnorm_{i+1}'] = nn.SyncBatchNorm(hidden_units[i])
                else:
                    self.od[f'batchnorm_{i+1}'] = nn.BatchNorm1d(hidden_units[i])
        # add final dense, linear layer aggregating the outcomes so far
        self.od['output layer'] = nn.Linear(self.hidden_units[-1],self.output_size)       
        
    def forward(self, x):
        
        x = x.view(-1, self.num_features)
        
        for step, layer in self.od.items():
           # switch off batch normalization when batch size = 1
           if x.shape[0] == 1 and (isinstance(layer, nn.SyncBatchNorm) or isinstance(layer, nn.BatchNorm1d)):
               continue
           x = layer(x)
        
        return x
    
    def get_architecture(self):
                
        print(nn.Sequential(self.od))


'''
Standard RNN that takes sequences of views and macro_ts features and predicts views.
So input size is seq_len x (nr_views+nr_macro_ts) and the output size (output_size) is nr_views.
num_features = nr_views+nr_macro_ts.
batch_size here is the number of dates, because features are only sensitive to date_id.
'''

class RecurrentNNStandardTorch(nn.Module):
    
    '''
    Recurrent GRU/LSTM cells. Allows for both types (GRU and LSTM), allows for intialization through an 
    outside state, for state propagation after each call of the forward method,
    allows for bidirectional RNN.
    
    Initialization:
        - kind (str): GRU or LSTM
        - num_features (int): nr of dimensions of each element of the batch
        - seq_len (int): seq_len of the past which is used as features for each observation
        - hidden_dim (int): the dimension of hidden state 
        - num_layers (int): the number of hidden layers (stacked RNN)
        - output_size (int or 'none'): if int, dimension of output after FC layer, otherwise 
        outputs state
        - outside_state (torch.Tensor or tuple of torch.Tensors): initializing state for the model,
            default = None (initializes with zero tensors)
        - propagate_state (False/True): whether last state will be propagated whenever forward is called
        or the state will be re-initialized from zero for each new forward
        - dropout (float): dropout rate for both encoder and decoder, default = None
        - output_only (True, False): 
    
    On Output: 
        - if output_size = 'none', tensor outputs contains state for all the past, i.e. has shape (batch_size, seq_len, hidden_dim)
          if output_size is int, output is from a FC layer, shape (batch_size, output_size)
        - if output_only: final_output is outputs, otherwise it's (outputs, current_state)
    
    Other parameters:
        - self.current_state: initialized through method init_state, gets propagated after every forward call
          if propagate_state; this is the state coming from last element of a batch
        - self.current_batch_states: final states after forward for the whole batch 
          (used for case where need both output and states for whole batch)
    
    Note: here we assume that whenever input changes within a batch, running state needs to be updated, otherwise not updated.
          This is not the default behavior from pytorch. For fin. time series, the current code covers both case where batch comes from
          a dataframe with index (time, asset) as well as only time. Only estimation issue would arise if for two consecutive periods 
          all time series have exactly the same values. 
    
    Note: - To get final states of a batch: load model, do forward on the batch and call 
            .current_batch_states
          - use LSTM without projections here
    '''

    def __init__(self,kind, 
                 num_features,
                 seq_len,
                 hidden_dim,
                 num_layers = 1,
                 output_size = 'none',
                 bidirectional = False,
                 dropout = 0,
                 output_only = True,
                 device = 'cpu'):
        
        super().__init__()
        self.kind = kind # str: LSTM, GRU
        self.num_features = num_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.directions = 2 if bidirectional else 1
        self.output_size = output_size
        self.dropout = dropout
        self.output_only = output_only
        self.device = device
        
        if self.kind == 'GRU':
            fun = nn.GRU
            
        elif self.kind == 'LSTM':
            fun = nn.LSTM

        else:
            raise ValueError('Class does not recognize the recurrent model type.')
            
        self.recurrent = fun(
                input_size = self.num_features,
                hidden_size = self.hidden_dim,
                num_layers = self.num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout = self.dropout)
        
        if isinstance(self.output_size,int):
            self.linear = nn.Linear(in_features=hidden_dim, out_features=self.output_size)
        
        # initialize state 
        self.init_state()    
            
    def init_state(self):
        
        if self.kind == 'LSTM':
            h_0 = torch.zeros(self.directions*self.num_layers, 1, self.hidden_dim)
            c_0 = torch.zeros(self.directions*self.num_layers, 1, self.hidden_dim)
            self.current_state = (h_0.detach(),c_0.detach())
        elif self.kind == 'GRU':
            h_0 = torch.zeros(self.directions*self.num_layers, 1, self.hidden_dim)
            self.current_state = h_0.detach()
                
    def forward(self, x):
        # x must be in the shape Nxseq_lenxnum_features, N here is the batch_size
        N = len(x)
        outputs = []
        running_states = []
        if x.shape != (N, self.seq_len, self.num_features):
            x = x.view((N, self.seq_len, self.num_features))
        
        out = -999*torch.ones(N, 1, self.directions*self.hidden_dim).detach()
        previous_input = -999*torch.ones(1, self.seq_len, self.num_features).detach()
        previous_input = previous_input.to(self.device)
        if self.kind == 'GRU':
            running_state = self.current_state.detach().clone()
        else:
            running_state = (self.current_state[0].detach().clone(),self.current_state[1].detach().clone())
        running_state = running_state.to(self.device)
        for i, input_t in enumerate(x.chunk(chunks = N, dim = 0)):
            # slow though not for our typical (stationarized) time series of macro data where there is considerable variation over time
            if not torch.all(input_t.eq(previous_input)):
                out, running_state = self.recurrent(input_t, running_state)
                previous_input = input_t.detach().clone()
                previous_input = previous_input.to(self.device)
            # register out and running state
            outputs += [out]
            if self.kind == 'GRU':
                running_states +=[running_state]
            elif self.kind == 'LSTM':
                running_states +=[running_state[0]]
       
        self.init_state()
        
        outputs = torch.stack(outputs, dim = 0).squeeze(dim=1)
        if self.directions>1:
            # make sure out is of shape (batch_size, seq_len, hidden_dim) always by eleminating the 
            # D dimension if bidirectional
            # delete the D dimension of the output by summing over it
            outputs = outputs.view(N,self.seq_len, self.directions, self.hidden_dim)
            outputs = torch.sum(outputs, axis = 2)
            
        self.current_batch_states = torch.stack(running_states, dim = 1).squeeze(dim = 2)
        if not isinstance(self.output_size, str):
            outputs = self.linear(outputs[:,-1,:])
        # if no FC layer in the end outputs has shape (batch_size, seq_len, hidden_dim)
        # otherwise it has shape (batch_size, output_size)
        final_output  =  outputs if self.output_only else (outputs, self.current_state)
        return final_output 


