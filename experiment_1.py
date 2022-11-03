# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from ff_nets import ff_net
import time

# Specifying problem parameters
BATCH_SIZE = 2 ** 10

N_RUNS = 5
N = 10000 # total number of training steps for generator
N_REPORT = 100
N_SAMPS_TOTAL = 10000  # number of samples generated
N_r = 500
N_s = 500  # number of first generator steps before adversary - helps convergence

# network architecture
LAYERS_H = 4
HIDDEN_H = 64
LAYERS_T = 4
HIDDEN_T = 64
ACT_T = 'tanh'
ACT_H = 'ReLu'

LIP_CONST = 1
LIP_LAM = 10 ## according to Gulrajanai et al, 2014

TOL = 0.01 ### break loop if average change over last N_r iterations <= TOL


# theta Unif(0,1)

def sample_theta(batch_size):
    while 1:
        theta_x = np.random.uniform(0, 1, [batch_size,2]) 
        theta_y = np.random.uniform(0, 1, [batch_size,2])
        theta = np.stack((theta_x,theta_y), axis = 1)
        yield theta

### pi = (mu, nu), mu ~ N(0,1), nu ~ N(0,4), 2D-gaussians

def sample_pi(batch_size):
    while 1:
        mu = np.random.multivariate_normal([0,0],np.eye(2),[batch_size])
        nu = np.random.multivariate_normal([0,0], 4*np.eye(2),[batch_size])
        pi = np.stack((mu,nu), axis = 1)
        yield pi
    
### objective function - squared distance (-> Wasserstein distance)
### then, theoretically: objective value should be 2

def c_objective(u):
    return -tf.reduce_sum(tf.square(u[:,0,:] - u[:,1,:]), axis = 1)

#BUILD GRAPH

obj_values_total = []
plot_values_total = []

for k_RUN in range(N_RUNS):

    t0 = time.time()
    tf.reset_default_graph()
    
    ##shape: dimension 1: number of samples, dimension 2: x&y marginals, dimension 3: time step 1 & 2
    x_pi = tf.compat.v1.placeholder(shape=[None, 2, 2], dtype=tf.float32)  # samples from pi -> distribution with marginals mu & nu
    x_theta = tf.compat.v1.placeholder(shape=[None, 2, 2], dtype=tf.float32)  # samples from theta, same structure as above
    
    T_theta = ff_net(x_theta, 'T', input_dim=2, output_dim=2, activation=ACT_T, n_layers=LAYERS_T, hidden_dim=HIDDEN_T)
    
    ### OT constraint (evaluation of functions from set H_1)
    
    h_pi = 0  # sum over h evaluated at samples of pi and samples of mu given x_1
    h_T_theta = 0  # sum over h evaluated at samples of theta and samples of mu given x_1
    for j in range(2): #evaluation of h_1, h_2 from H_1
        h_pi += ff_net(x_pi[:,j:(j + 1),:], 'h_' + str(j), input_dim=2, output_dim=1, activation=ACT_H,
                       n_layers=LAYERS_H, hidden_dim=HIDDEN_H)
        h_T_theta += ff_net(T_theta[:, j:(j + 1),:], 'h_' + str(j), input_dim=2, output_dim=1, activation=ACT_H,
                            n_layers=LAYERS_H, hidden_dim=HIDDEN_H)
    
    obj = tf.reduce_mean(c_objective(T_theta)) - tf.reduce_mean(h_T_theta) + tf.reduce_mean(h_pi)        
         
    ### Lipschitz constraint
    
    ## 1) gradients
    
    x_pi_1 = x_pi[:,0:1,:]
    grad_h_pi_1 = tf.gradients(ff_net(x_pi_1, 'h_0', n_layers=LAYERS_H, 
                                      hidden_dim=HIDDEN_H, input_dim=2,
                                      output_dim=1, activation=ACT_H), [x_pi_1])[0]
    x_pi_2 = x_pi[:,1:2,:]
    grad_h_pi_2 = tf.gradients(ff_net(x_pi_2, 'h_1', n_layers=LAYERS_H, 
                                      hidden_dim=HIDDEN_H, input_dim=2,
                                      output_dim=1, activation=ACT_H), 
                               [x_pi_2])[0]
    
    T_theta_1 = T_theta[:,0:1,:]
    grad_h_T_theta_1 = tf.gradients(ff_net(T_theta_1, 'h_0', n_layers=LAYERS_H, 
                                      hidden_dim=HIDDEN_H, input_dim=2,
                                      output_dim=1, activation=ACT_H), 
                               [T_theta_1])[0]
    T_theta_2 = T_theta[:,1:2,:]
    grad_h_T_theta_2 = tf.gradients(ff_net(T_theta_2, 'h_1', n_layers=LAYERS_H, 
                                      hidden_dim=HIDDEN_H, input_dim=2,
                                      output_dim=1, activation=ACT_H), 
                               [T_theta_2])[0]
    
    ## 2) norm of gradients
    
    sqrt_eps = 10 ** (-9) # this is added to avoid infinite gradients of the square root at 0
    
    norm_h_pi_1 = tf.sqrt(sqrt_eps + tf.reduce_sum(tf.square(grad_h_pi_1), 
                                                   reduction_indices=[1]))
    norm_h_pi_2 = tf.sqrt(sqrt_eps + tf.reduce_sum(tf.square(grad_h_pi_2), 
                                                   reduction_indices=[1]))
    
    norm_h_T_theta_1 = tf.sqrt(sqrt_eps + tf.reduce_sum(tf.square(grad_h_T_theta_1), 
                                                        reduction_indices=[1]))
    norm_h_T_theta_2 = tf.sqrt(sqrt_eps + tf.reduce_sum(tf.square(grad_h_T_theta_2), 
                                                        reduction_indices=[1]))
    
    ## 3) add to objective function
    
    h_pi_LIP = h_pi + LIP_LAM * (tf.nn.relu(norm_h_pi_1 - LIP_CONST) ** 2 +
                                 tf.nn.relu(norm_h_pi_2 - LIP_CONST) ** 2)
    
    h_T_theta_LIP = h_T_theta - LIP_LAM * (tf.nn.relu(norm_h_T_theta_1 - LIP_CONST) ** 2 +
                                            tf.nn.relu(norm_h_T_theta_2 - LIP_CONST) ** 2)
    
    ## new objective function for h with penalty to ensure Lipschitz regularity
    
    obj_h = tf.reduce_mean(c_objective(T_theta)) - tf.reduce_mean(h_T_theta_LIP) + tf.reduce_mean(h_pi_LIP)
    
    T_vars = [v for v in tf.compat.v1.global_variables() if ('T' in v.name)]
    h_vars = [v for v in tf.compat.v1.global_variables() if ('h' in v.name)]
    
    train_op_h = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, 
                                                  beta1=0.5, beta2=0.9, 
                                                  epsilon=1e-08).minimize(
            obj_h, var_list=h_vars)
    train_op_T = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, 
                                                  beta1=0.5, beta2=0.9, 
                                                  epsilon=1e-08).minimize(
            -obj, var_list=T_vars) 
        
    # run session
    
    objective_values = []
    integral_values = []
    samp_pi = sample_pi(BATCH_SIZE)
    samp_theta = sample_theta(BATCH_SIZE)
    
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        plot_vals = []
        for t in range(1,N+1):
            if t > N_s:
                s_pi = next(samp_pi)
                s_theta = next(samp_theta)
                (_,) = sess.run([train_op_h], feed_dict={x_pi: s_pi, x_theta: s_theta})
            s_pi = next(samp_pi)
            s_theta = next(samp_theta)
            (_,ov) = sess.run([train_op_T, obj], feed_dict={x_pi: s_pi, x_theta: s_theta})
            objective_values.append(-ov)
            if t % 100 == 0:
                print(t, f"objective value = {np.mean(objective_values[-N_r:])}")
                plot_vals.append(np.mean(objective_values[-N_r:]))
                if (t > N_r) and (np.mean(np.abs(np.diff(objective_values))[-N_r:]) < TOL):
                    break
        print(f"final objective value run {k_RUN} = {np.mean(objective_values[-N_r:])}")
        print(f"runtime run {k_RUN}: + {time.time() - t0}")
    obj_values_total.append(objective_values)
    plot_values_total.append(plot_vals)