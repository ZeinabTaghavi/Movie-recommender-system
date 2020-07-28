# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from model import RBM
import torch
from train_test import train , test


def convert(data):
    new_data = []
    for id_users in range(1 , nb_users+1):
        id_movies = data[:,1][data[:,0]==id_users]
        id_rating = data[:,2][data[:,0]==id_users]
        rating = np.zeros(nb_movies)
        rating[id_movies - 1 ] = id_rating
        new_data.append(list(rating))
        
    return new_data



if __name__=='__main__':
    
    # download dataset if not extists
    
    if not(os.path.exists('./ml-1m/movies.dat') and
           os.path.exists('./ml-1m/users.dat') and
           os.path.exists('./ml-1m/ratings.dat')):
        os.system('wget "http://files.grouplens.org/datasets/movielens/ml-1m.zip"')
        os.system('unzip ml-1m.zip')
        os.system('ls')
        
        
    if not(os.path.exists('./ml-100k/u1.base') and
           os.path.exists('./ml-100k/u1.test')):
        os.system('wget "http://files.grouplens.org/datasets/movielens/ml-100k.zip"')
        os.system('unzip ml-100k.zip')
        os.system('ls')


    # data preprocessing
    
    movies = pd.read_csv('ml-1m/movies.dat' , sep='::' , header=None , engine='python' , encoding='latin-1')
    users =  pd.read_csv('ml-1m/users.dat' , sep='::' , header=None , engine='python' , encoding='latin-1')
    ratings =  pd.read_csv('ml-1m/ratings.dat' , sep='::' , header=None , engine='python' , encoding='latin-1')
    
    training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t' , header=None)
    training_set = np.array(training_set , dtype = int)
    testing_set = pd.read_csv('ml-100k/u1.test', delimiter='\t' , header=None)
    testing_set = np.array(testing_set , dtype = int)
    
    nb_users = int(max(max(training_set[:,0]),max(testing_set[:,0])))
    nb_movies = int(max(max(training_set[:,1]),max(testing_set[:,1])))
    
    training_set = convert(training_set)
    testing_set = convert(testing_set)
    
    
    training_set = torch.FloatTensor(training_set)
    testing_set = torch.FloatTensor(testing_set)
    
    training_set[training_set==0]= -1
    training_set[training_set==1]= 0
    training_set[training_set==2]= 0
    training_set[training_set>=3]= 1
    
    
    testing_set[testing_set==0]= -1
    testing_set[testing_set==1]= 0
    testing_set[testing_set==2]= 0
    testing_set[testing_set>=3]= 1
    
    
    # model
    
    nv = len(training_set[0])
    nh = 100 # we want to detect 100 features
    rbm = RBM(nv, nh)


    # train
    
    batch_size = 100
    epochs = 10
    train(rbm,nv,nh,batch_size,epochs,training_set,testing_set,nb_users)
    
    
    # test
    
    test(rbm,training_set,testing_set,nb_users)

    
    
    



    
    
