# -*- coding: utf-8 -*-
import torch

def train(rbm,nv,nh,batch_size,epochs,training_set,testing_set,nb_users):
    for epoch in range(1 , epochs + 1):
        train_loss = 0
        s = 0. # counter
        for id_user in range(0, nb_users - batch_size , batch_size):
            vk = training_set[id_user: id_user+batch_size]
            v0 = training_set[id_user: id_user+batch_size]
            ph0,_ = rbm.sample_h(v0)
            
            for k in range(10):
                _, hk = rbm.sample_h(vk)
                _, vk = rbm.sample_v(hk)
                vk[v0<0] = v0[v0<0]
            
            phk,_ = rbm.sample_h(vk)
            rbm.train(v0, vk, ph0, phk)
            train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
            s += 1.
        print('tarin epoch:' + str(epoch) + 'loss: ' , str(train_loss/s))
        
        
def test(rbm,training_set,testing_set,nb_users):
    test_loss = 0
    s = 0.
    for id_user in range(nb_users):
        v = training_set[id_user:id_user+1]
        vt = testing_set[id_user:id_user+1]
        if len(vt[vt>=0]) > 0:
            _,h = rbm.sample_h(v)
            _,v = rbm.sample_v(h)
            test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
            s += 1.
    print('test loss: '+str(test_loss/s))
    pass
