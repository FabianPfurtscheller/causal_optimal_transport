# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stats

def sample_cond_mu(pi, batch_size):
    while 1:
        mu = next(pi)[:,0,:]
        bw = batch_size**(-1./(2+4))*mu.std() #Scott's Rule for bandwith, check references here https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html#ra3a8695506c7-1
        kde = stats.gaussian_kde(mu.T) #not necessarily gaussian, just estimated using gaussians
        sample = kde.resample(8*batch_size) #ensures interval between bandwiths is not empty
        cond_mu = np.empty((batch_size, int(batch_size/2), 2))
        for i in range(batch_size):
            cond = conditional_sampling(sample, mu[i,0], bw, int(batch_size/2))
            cond_mu[i,:,0] = np.repeat(mu[i,0],int(batch_size/2)) #first column is x_1 
            cond_mu[i,:,1] = np.random.choice(cond, int(batch_size/2), replace = True) #second to last column are sampled x_2 given x_1
        yield cond_mu
        
def sample_cond_T_theta(T_theta, pi, batch_size):
    cond_batch_size = int(batch_size/2)
    while 1:
        T_theta_1 = T_theta[:,0,:]
        mu = next(pi)[:,0,:]
        bw = batch_size**(-1./(2+4))*mu.std() #Scott's Rule for bandwith, check references here https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html#ra3a8695506c7-1
        kde = stats.gaussian_kde(mu.T) #not necessarily gaussian, just estimated using gaussians
        sample = kde.resample(8*batch_size) #ensures interval between bandwiths is not empty
        cond_theta = np.empty((batch_size, cond_batch_size, 2))
        for i in range(batch_size):
            cond = conditional_sampling(sample, mu[i,0], bw, int(batch_size/2))
            cond_theta[i,:,0] = np.repeat(T_theta_1[i,0],int(batch_size/2)) #first column is x_1 
            cond_theta[i,:,1] = np.random.choice(cond, int(batch_size/2), replace = True) #second to last column are sampled x_2 given x_1
        yield cond_theta


def conditional_sampling(sample, x_1, bw, batch_size):
    lower = x_1 - bw
    upper = x_1 + bw
    bw_indices = np.where((lower < sample[0,:]) & (sample[0,:] < upper))
    return sample[1,bw_indices].flatten()


