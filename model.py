# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable



class RBM():
    def __init__(self , nv , nh):
        self.W = torch.randn((nh , nv)) # weights
        self.a = torch.randn((1 , nh)) # bias for hiddens
        self.b = torch.randn((1 , nv)) # bias for visibles      
        
    def sample_h(self , x): # x: the visible node v in probablity of p
        wx = torch.mm(x , self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v , torch.bernoulli(p_h_given_v)
    
    def sample_v(self , y): # y: the hidden node v in probablity of p
        wy = torch.mm(y , self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h , torch.bernoulli(p_v_given_h)
    def train(self, v0 , vk , ph0 , phk): # input vector/ visible nodes after k sampling/ 
                                          # vactor of probabilties(at first iteration=0)/proba... after k sampling 
        self.W += (torch.mm(v0.t() , ph0) - torch.mm(vk.t() , phk)).t()
        self.b += torch.sum((v0 - vk) , 0)
        self.a += torch.sum((ph0 - phk) , 0) 
        
        
        