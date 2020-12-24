# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 18:23:26 2020

@author: sushanthsgradlaptop2
"""

import numpy as np
import tensorflow as tf
from matplotlib import pyplot
mnist=tf.keras.datasets.mnist
(x_train, y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test=x_train/255.0,x_test/255.0
def OMP_iter(genmtx, d_i, spp):
    x = d_i
    loc = []
    res={}
    for i in range(spp):
        am = np.matmul(np.transpose(genmtx), d_i).reshape([genmtx.shape[1], 1])
        l1 = np.argmax(np.abs(am))
        
        loc.append(l1)
        act_set = genmtx[:, loc]
        a = np.matmul(np.linalg.pinv(act_set), x)
        d_i = x-np.matmul(act_set, a)
        if(np.sum(d_i**2) < 1e-6):
            break
    res['vals']=a.reshape([a.shape[0],1])
    res['locs']=loc
    return(res)




def dict_update(coef_mat,tr_dat):
    d_up=np.matmul(tr_dat,np.linalg.pinv(coef_mat))
    return(d_up)

def sp_code(genmtx,tr_dat,spp):
    z=np.zeros([genmtx.shape[1],tr_dat.shape[1]])
    for i in range(tr_dat.shape[1]):
        d_i=tr_dat[:,i]
        g=OMP_iter(genmtx,d_i,spp)
        z[g['locs'],i]=g['vals'].ravel()
    return(z)

def whiten_zca(dat,eps_v):
    d_m=dat-np.mean(dat)
    d_m_co=np.matmul(d_m,np.transpose(d_m))/dat.shape[1]
    [u,s,vh]=np.linalg.svd(d_m_co) # s not returned as diag matrix
    d_w1=np.matmul(u,np.diag(1/np.sqrt(s+eps_v)))
    d_w2=np.matmul(d_w1,np.transpose(u))
    w_dat=np.matmul(d_w2,d_m)
    return(w_dat)

def disp_dict(W1,figdims):
    sm=np.zeros([figdims[0]*28,figdims[1]*28])
    k=0
    for i in range(figdims[0]):
        for j in range(figdims[1]):
            sm[i*28:(i+1)*28,j*28:(j+1)*28]=np.reshape(W1[:,k],[28,28])
            k=k+1
    pyplot.imshow(sm,vmin=-0.1,vmax=0.1,cmap='gray')

def train_dict_ILS_MOD(dict_p,tr_dat,spp,iter_v,mean_flag):
    err=np.zeros([iter_v,1])
    dict_train={}
    for i in range(iter_v):
        print(i)
        if(i==0 and mean_flag==1):
            coef_m=sp_code(dict_p,tr_dat,spp)
        else:
            if(mean_flag==1):
                d_c=tr_dat-np.matmul(dict_p[:,0][:,None], coef_m[0,:][None,:])
                coef_temp=sp_code(dict_p[:,1:],d_c,spp)
                coef_m=np.append(coef_m[0,:][None,:],coef_temp,axis=0)
            else:
                coef_m=sp_code(dict_p,tr_dat,spp)
        res=np.matmul(dict_p,coef_m)-tr_dat
        err[i]=np.mean(np.mean(res**2))
        print(err[i])          
        if(mean_flag==1):
             d_c=tr_dat-np.matmul(dict_p[:,0][:,None], coef_m[0,:][None,:])
             d_up=dict_update(coef_m[1:,:],d_c)
             d_up=np.append(np.ones([dict_p.shape[0],1]),d_up-np.mean(d_up),axis=1)
             d_up=d_up/(np.sqrt(np.sum(d_up**2,axis=0)))
        else:
            d_up=dict_update(coef_m,tr_dat)
            d_up=d_up/(np.sqrt(np.sum(d_up**2,axis=0)))
        dict_p=d_up
       
    dict_train['dict']=d_up
    dict_train['err']=err
    return(dict_train)
#%%
flat_dat=np.zeros([x_train.shape[1]*x_train.shape[2],x_train.shape[0]])
for i in range(x_train.shape[0]):
    flat_dat[:,i]=np.ndarray.flatten(x_train[i,:,:])
d_w=whiten_zca(flat_dat,1e-1)
#disp_dict(d_m1,[16,16])
rp=np.random.permutation(d_w.shape[1])
n_atoms=255
spp=5
iter_v=40
mean_flag=1
d_init1=d_w[:,rp[0:n_atoms]]
d_init=np.append(np.ones([d_w.shape[0],1]),d_init1-np.mean(d_init1),axis=1)
d_init=d_init/np.sqrt(np.sum(d_init**2,axis=0))
disp_dict(d_init,[16,16])
#%%
d_train_n=train_dict_ILS_MOD(d_init,d_w[:,rp[0:25000]],spp,iter_v,mean_flag)
#%%
disp_dict(d_train_n['dict'],[16,16])