# -*- coding: utf-8 -*-
"""
Neuromorphic Setup (Nonlinear Transformation from Linear Wave Scattering)

Code for training a neuromorphic device implementing nonlinear
information processing via the parameter dependence of the
scattering matrix.

Attributes implemented here (see manuscript):
- Transmission Setup
- Convolutional Structure
- Data set: Fashion MNIST

Requirements: jax, optax, tensorflow [only for data loading]

Train the parameters in a scattering matrix using jax.

NOTE:
Directory 'outputs' needs to exist in the current directory!

C. Wanjura and F. Marquardt
2023/24, MIT License
"""

# Parameters

scale_down_images=False # if True, really use 14x14 instead of 28x28
input_scale=5 # scaling of input values physically injected into detunings of optical modes
random_scale=1 # global scaling of random initialization of couplings (fully connected layers)
random_kernel_scale=0.1 # scaling of random kernel weight initialization (conv layers)
random_layer0_detuning_scale = 5 # random init of layer 0 detunings
random_detuning_scale = 0.05 # random init scale of detunings

# 28x28x1 --> (4x4 kernel) --> 14x14xchannels_1 --> (4x4 kernel) --> 7x7xchannels_2 --> N3 --> 10
channels_0=1
channels_1=10
channels_2=16
N3=100

batch_size=20 # number of samples per batch
train_batches=300_000 # how many training batches to do (note 1 epoch = 60_000 samples)
max_learning_rate=0.01 # for other details, look below for schedule
end_learning_rate=0.0

import optax

#schedule=optax.linear_onecycle_schedule(transition_steps=300_000, peak_value=max_learning_rate, pct_start=0.3, pct_final=0.85, div_factor=25.0, final_div_factor=10000.0)


optimizersched = optax.chain(
    optax.amsgrad(learning_rate=max_learning_rate),
    optax.clip(0.001)
)

DIRNAME="./outputs/" # intermediate and final results go inside here



# Import necessary libraries

import numpy as np
from numpy import loadtxt

import jax.numpy as jnp
import jax
from jax import jit, vmap, grad, value_and_grad

from jax import random
key = random.PRNGKey(0)

import optax

import tensorflow as tf
import keras

# Do not waste GPU memory on tensorflow:
tf.config.experimental.set_visible_devices([], "GPU")

import matplotlib.pyplot as plt
import time

import os

"""# Definition of Network"""

if scale_down_images:
    scale_image_factor=4 # reduction factor for scale for number of pixels
    img_scaling=2 # linear scale reduction
else:
    scale_image_factor=1
    img_scaling=1

def invSuscept(gamma, omega):
    return jnp.diag(gamma/2 - 1j*omega)

def suscept(gamma, omega):
    return jnp.diag(1/(gamma/2 - 1j*omega))

# initialise gamma and omega vectors and JJ with random values
# Nj is the size of the jth layer
def randomVector(Nj):
    return np.random.uniform(size=(Nj,))

def randomJJ(N1, N2):
    return np.random.uniform(size=(N1,N2))

# Produce matrices that reflect the convolutional structure.
# These are sparse matrices that will be multiplied by the learnable
# parameters for the convolutional kernels.

def mat_idx(pos_target,pos_input,M_target,M_input):
    '''
    Return tuple of idx_target, idx_input that 
    addresses the right element in the N_target x N_input matrix.

    Where: pos=(jx,jy,ch) and we are dealing with MxM images for target and input.
    '''
    return (pos_target[0]+pos_target[1]*M_target + pos_target[2]*M_target**2), (pos_input[0]+pos_input[1]*M_input + pos_input[2]*M_input**2)


def create_conv_stride_matrices_centered_fixed_channel(M_input,stride,channels_target,channels_input,
                                         kernel_max_shift):
    '''
    Create a set of matrices to describe the convolutional mapping. These matrices
    then can be multiplied by trainable parameters to obtain the actual coupling matrix.

    This version is centered, i.e. for a stride of 2, the result inside jx=jy=0 is taken
    from the input pixels (jx,jy) at (0,0),(0,1),(1,0) and (1,1). For that purpose, stride must
    be even.

    This particular version 'fixed_channel' is setting up a matrix of full size,
    but only produces the matrix for input and target channel indices both equal to zero.
    The idea is that actually needed matrices will be generated on-the-fly by jnp.roll,
    shifting the matrix as needed. This saves memory (a bit), so one can go to larger
    channel numbers, at the expense of longer run time.

    Output: matrix, where

        matrix[dx+kernel_max_shift,dy+kernel_max_shift]

    is the matrix that maps an input vector (2d image with channels) to the output vector,
    with the shifts dx,dy. Here dx,dy range between -kernel_max_shift and +kernel_max_shift-1

    This is a matrix of size (channels_target*M_target**2, channels_input*M_input**2).

    It can be multiplied by the trainable parameter for this part of the kernel:

        parameter[dx+kernel_max_shift,dy+kernel_max_shift]*matrix[above]
    
    See the code for network() below to see how this is used!

    '''
    M_target=int(M_input//stride)

    matrix=np.zeros((2*kernel_max_shift,2*kernel_max_shift,channels_target*M_target**2,channels_input*M_input**2),dtype='int8')

    cht=0
    ch=0
    for dx in range(-kernel_max_shift,kernel_max_shift):
        for dy in range(-kernel_max_shift,kernel_max_shift):
            for jtx in range(M_target):
                for jty in range(M_target):
                    jx=jtx*stride+dx+stride//2
                    jy=jty*stride+dy+stride//2
                    if jx>=0 and jx<M_input and jy>=0 and jy<M_input:
                        idx_matrix=mat_idx((jtx,jty,cht),(jx,jy,ch),M_target,M_input)
                        matrix[dx+kernel_max_shift,dy+kernel_max_shift,idx_matrix[0],idx_matrix[1]]=1.0

    return matrix

@jit
def network(x, parameters10, parameters21, JJ2, JJ3, omega0, omega1, omega2, omega3, omega4,
            gamma0, gamma1, gamma2, gamma3, gamma4):
    """
    The main function of the whole program. Evaluate the output of a neuromorphic network,
    with input x: The transmission scattering matrix.

    The structure of this network is of the type

    28x28x1 -- (conv) -- 14x14x(channels_1) -- (conv) -- 7x7x(channels_2) -- N3 -- 10

    (or replace 28x28 by 14x14 if small images are selected)

    where the (conv) represents convolutional couplings. Please note, as explained in
    the manuscript, that in spite of this seemingly conventional structure, there is backscattering
    of waves back and forth through the whole device, so the dynamics is significantly more
    complicated than for a feedforward network.

    Trainable parameters are given inside parameters10, parameters21 (for the 
    convolutional kernels) and JJ2,JJ3 (for the connection weights of subsequent layers),
    as well as omega0..4 (for the vectors denoting the tuneable frequency offsets).
    gamma0..4 denote fixed parameters, namely vectors containing the decay rates for the modes
    in the layers 0 to 4.
    """

# Calculate the various susceptibilities:
    chi4 = suscept(gamma4, -omega4)
    chi3_inv = invSuscept(gamma3, -omega3)
    chi2_inv = invSuscept(gamma2, -omega2)
    chi1_inv = invSuscept(gamma1, -omega1)
    chi0_inv = invSuscept(gamma0, -(x+omega0) )
    
    JJ3_tr=jnp.transpose(JJ3)
    JJ2_tr=jnp.transpose(JJ2)


# First step: generate the actual coupling matrix JJ0 (layers 0 to 1) from
# the convolutional parameters.
#
# In the following scan loop, we go through all trainable parameters
# indexed by the integer j, where
#    j = channel_target*big_shift + channel_input*small_shift + kernel_part
# Here:
# big_shift = num_channels_input * num_kernel_params
# small_shift = num_kernel_params
# thus: channel_target = j//big_shift
# thus: channel_input = (j%big_shift)//num_kernel_params

    big_shift_10=channels_0 * num_kernel_params_10
    small_shift_10=num_kernel_params_10

    def f_loop_JJ0(carry,j):
        return carry+parameters10[j]*jnp.roll(matrix_10[j%num_kernel_params_10],
         (pixels1*(j//big_shift_10),pixels0*((j%big_shift_10)//num_kernel_params_10)),
                                              axis=(0,1)), None
    
    JJ0_tr,_ = jax.lax.scan(f_loop_JJ0, jnp.zeros((N1,N0)), jnp.arange(num_parameters10))
    JJ0=jnp.transpose(JJ0_tr)

# Second step: generate the actual coupling matrix JJ1 (layers 1 to 2) from
# the convolutional parameters.

    big_shift_21=channels_1 * num_kernel_params_21
    small_shift_21=num_kernel_params_21

    def f_loop_JJ1(carry,x):
        return carry+parameters21[x]*jnp.roll(matrix_21[x%num_kernel_params_21],
         (pixels2*(x//big_shift_21),pixels1*((x%big_shift_21)//num_kernel_params_21)),
                                              axis=(0,1)), None
    
    JJ1_tr,_ = jax.lax.scan(f_loop_JJ1, jnp.zeros((N2,N1)), jnp.arange(num_parameters21))
    JJ1=jnp.transpose(JJ1_tr)

# Calculate the modified susceptibilities of the layers:
    chi3_tilde = jnp.linalg.inv( chi3_inv + jnp.matmul(jnp.matmul(JJ3, chi4), JJ3_tr) )
    chi2_tilde = jnp.linalg.inv( chi2_inv + jnp.matmul(jnp.matmul(JJ2, chi3_tilde), JJ2_tr) )
    chi1_tilde = jnp.linalg.inv( chi1_inv + jnp.matmul(jnp.matmul(JJ1, chi2_tilde), JJ1_tr) )
    chi0_tilde = jnp.linalg.inv( chi0_inv + jnp.matmul(jnp.matmul(JJ0, chi1_tilde), JJ0_tr) )

# Return the full transmission scattering matrix, going from layer 0 to layer 4:
    return jnp.dot(chi4, jnp.dot( JJ3_tr, jnp.dot(chi3_tilde, jnp.dot( JJ2_tr, jnp.dot(chi2_tilde, jnp.dot( JJ1_tr, jnp.dot(chi1_tilde, jnp.dot(JJ0_tr, jnp.dot(chi0_tilde, jnp.sqrt(gamma0))))))))))


@jit
def costF(x, y, parameters10, parameters21, JJ2, JJ3, omega0, omega1, omega2, omega3, omega4,
            gamma0, gamma1, gamma2, gamma3, gamma4):
    prob = jnp.log(jax.nn.softmax(8 * jnp.imag(network(x, parameters10, parameters21, JJ2, JJ3,
            omega0, omega1, omega2, omega3, omega4,
            gamma0, gamma1, gamma2, gamma3, gamma4))))
    return - jnp.dot(prob, y)

@jit
def costF_vmap(x, y, parameters10, parameters21, JJ2, JJ3, omega0, omega1, omega2, omega3, omega4,
            gamma0, gamma1, gamma2, gamma3, gamma4):
    return jnp.mean(vmap(costF, in_axes = (0,0)+(None,)*14, out_axes=0)(x, y, parameters10, parameters21, JJ2, JJ3,
            omega0, omega1, omega2, omega3, omega4,
            gamma0, gamma1, gamma2, gamma3, gamma4),axis=0)

@jit
def real_costF_grad_all_mean_and_value(x, y, params, fixed_params):
    return value_and_grad(costF_vmap, argnums=[2,3,4,5,6,7,8,9,10])(x, y, *params, *fixed_params)

"""# Loading and rescaling fashion MNIST"""

# loading fashion mnist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)
x_train_vect = jnp.float32(jnp.array(input_scale * (1 - jnp.array(x_train)/256)))
x_test_vect = jnp.float32(jnp.array(input_scale * (1 - jnp.array(x_test)/256)))


if scale_down_images:
    # scaling down pictures
    n_train=np.shape(x_train)[0]
    x_train_new = np.empty((n_train,int(784//4)))
    n_test=np.shape(x_test)[0]
    x_test_new = np.empty((n_test,int(784//4)))

    row = np.arange(0,28,2)
    idx_00=[]
    for k in range(0,28,2):
        idx_00.extend(list(k*28+row))

    # indices for all 4 pixels to be grouped
    idx_00 = np.array(idx_00)
    idx_01 = idx_00 + 1
    idx_10 = idx_00 + 28
    idx_11 = idx_00 + 29

    # average over groups of 2x2
    x_train_new[:,:]=0.25*(1.0*x_train[:,idx_00]+1.0*x_train[:,idx_01]+
                    1.0*x_train[:,idx_10]+1.0*x_train[:,idx_11])

    x_test_new[:,:]=0.25*(1.0*x_test[:,idx_00]+1.0*x_test[:,idx_01]+
                    1.0*x_test[:,idx_10]+1.0*x_test[:,idx_11])

    # data vectors 'real'

    x_train_vect = jnp.float32(jnp.array(input_scale * (1 - jnp.array(x_train_new)/256)))

    x_test_vect = jnp.float32(jnp.array(input_scale * (1 - jnp.array(x_test_new)/256)))


# fill correct one-hot output
y_train = np.array(y_train, dtype = int)
y_train_vect = np.zeros((len(y_train),10))

for jj in range(len(y_train)):
    y_train_vect[jj, int(y_train[jj])]=1.0
y_train_vect = jnp.float32(y_train_vect)

y_test = np.array(y_test, dtype = int)
y_test_vect = np.zeros((len(y_test),10))
for jj in range(len(y_test)):
    y_test_vect[jj, int(y_test[jj])]=1.0
y_test_vect = jnp.float32(y_test_vect)

"""# Visualization and analysis routines"""

# visualize_test_batch
# to be used during training

# create a test batch that is nicely ordered
# first samples_per_fig will be '0' etc
samples_per_fig=5
pixels_per_fig=int(784//scale_image_factor)
x_test_batch=np.empty((samples_per_fig*10,pixels_per_fig),dtype='complex')

for j in range(10):
    indices=np.where(y_test==j)[0]
    x_test_batch[j*samples_per_fig:(j+1)*samples_per_fig,:]=x_test_vect[indices[:samples_per_fig],:]

x_test_batch=jnp.complex64(x_test_batch)

def visualize_test_batch_params(params,fixed_params):
    samples_per_fig=5
    out_network = jnp.ravel(jnp.imag(vmap(network, in_axes = (0,)+(None,)*14, out_axes = 0)(x_test_batch, *params, *fixed_params)))
    out_network = out_network.reshape(len(x_test_batch), 10)
    plt.figure(dpi=200)
    plt.imshow(np.transpose(out_network[:50]))
    for s in range(10):
        plt.plot((s*samples_per_fig-0.5,s*samples_per_fig-0.5),
                 (-0.5,9.5),color="white")
    plt.axis('off')

# get the whole confusion matrix
# on a larger part of the data set!

network_vmap=jax.jit(vmap(network, in_axes = (0,)+(None,)*14, out_axes = 0))

conf_samples_per_fig=1000
pixels_per_fig=int(784/scale_image_factor)
x_test_batch_large=np.empty((conf_samples_per_fig*10,pixels_per_fig),dtype='complex')

for j in range(10):
    indices=np.where(y_test==j)[0]
    x_test_batch_large[j*conf_samples_per_fig:(j+1)*conf_samples_per_fig,:]=x_test_vect[indices[:conf_samples_per_fig],:]

x_test_batch_large=jnp.complex64(x_test_batch_large)

def get_confusion_matrix(params,fixed_params,samplesize=None):

    if samplesize is None:
        samplesize=conf_samples_per_fig

    acc_test_digit=np.zeros((10,10))
    for j in range(10):
        n_sub_samples=int(conf_samples_per_fig/samplesize)
        for sample_idx in range(n_sub_samples):
            the_set = x_test_batch_large[j*conf_samples_per_fig+sample_idx*samplesize:j*conf_samples_per_fig+sample_idx*samplesize+samplesize]
            out_network_test = jnp.ravel(jnp.imag(network_vmap(the_set, *params, *fixed_params)))
            out_network_test = out_network_test.reshape(len(the_set), 10)
            for k in range(10):
                contrib=sum(jnp.argmax(out_network_test, axis = 1) == k)/len(the_set)
                #print(j,sample_idx,k, contrib)
                acc_test_digit[j][k]+=contrib
        acc_test_digit[j]/=n_sub_samples
        print(acc_test_digit[j][j])

    return acc_test_digit

# plot the confusion matrix
def plot_confusion_matrix(name,confm):
    plt.matshow(confm)
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    plt.xlabel("detected")
    plt.ylabel("true")
    plt.title(name+" "+f"{np.mean(np.diag(confm)*100):2.1f}%")
    plt.colorbar()

"""# Training routines"""

def save_params(NAME, t, params):
    np.savez(NAME+"_t=" + str(t) + "_params.npz", *params)

def load_params(NAME, t):
    arrays=np.load(NAME+"_t=" + str(t) + "_params.npz")
    return [arrays[key] for key in arrays]

def train(params_fixed,params,optimizer,opt_state,nsteps,cost_values):
    @jit
    def gradient_step(x_batch,y_batch,params, opt_state):
        value, grads = real_costF_grad_all_mean_and_value(x_batch, y_batch, params, params_fixed)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    j=-1
    iterations_per_batch=60000/batch_size
    accuracies=[]

    for step in range(nsteps):
        if step%iterations_per_batch==0 or step==nsteps-1:
            save_params(DIRNAME+"/"+FILENAME,step,params)

        j+=1

        if j % iterations_per_batch == 0:
            random_mix_2 = np.random.choice(len(x_train_vect), len(x_train_vect), replace = False)
            j = 0

        batch_subset = random_mix_2[j * batch_size: (j+1) * batch_size]
        x_batch = x_train_vect[batch_subset]
        y_batch = y_train_vect[batch_subset]

        params,opt_state,value=gradient_step(x_batch,y_batch,params,opt_state)

        cost_values.append(float(value))

        if step%300==0:
            print(f"{step:5d}:  cost={value:2.3f}             ")
        if step%iterations_per_batch==0 or step==nsteps-1:
            conf_matrix=get_confusion_matrix(params, fixed_params, samplesize=100)

            plot_confusion_matrix(name,conf_matrix)
            plt.savefig(DIRNAME+"/"+FILENAME+"_confusion.pdf")
            plt.show()

            accuracy=np.mean(np.diag(conf_matrix))
            print("Epoch ", int(step/iterations_per_batch), " Accuracy: ", accuracy)
            accuracies.append(accuracy)

    print("Accuracies:")
    print(accuracies)

    return params,opt_state

"""# Initializing the trainable parameters (once)"""

# create necessary convolutional matrices

# create template matrices
# in this new version, create them only for 1x1 channels, later
# use roll to on-the-fly produce them for arbitrary channel combinations
# see code inside network(..)
# (saving memory)
matrix_10=create_conv_stride_matrices_centered_fixed_channel(M_input=28//img_scaling,stride=2,channels_target=channels_1,channels_input=channels_0,
                                                kernel_max_shift=2)
matrix_10=jnp.array(matrix_10.reshape(-1, *matrix_10.shape[-2:])) # flatten all these channel/kernel dims

matrix_21=create_conv_stride_matrices_centered_fixed_channel(M_input=14//img_scaling,stride=2,channels_target=channels_2,channels_input=channels_1,
                                                kernel_max_shift=2)
matrix_21=jnp.array(matrix_21.reshape(-1, *matrix_21.shape[-2:])) # flatten all these channel/kernel dims

M_pixels=28//img_scaling

pixels0=M_pixels**2
pixels1=(M_pixels//2)**2
pixels2=(int(M_pixels//4))**2
N0=pixels0
N1=pixels1*channels_1
N2=pixels2*channels_2
# N3: see above, defined at beginning
N4=10

kernel_10_max=2 # max kernel shift (will be 4x4 kernel)
kernel_21_max=2

num_kernel_params_10=(kernel_10_max*2)**2
num_kernel_params_21=(kernel_21_max*2)**2

print("Total number of parameters: ", channels_1*(kernel_10_max)**2 + channels_1*channels_2*(kernel_21_max)**2 + N2*N3 + N3*N4)

# initialization of parameters

gamma_max = 1.0

keys = random.split(key,7)

gamma0 = gamma_max * jnp.float32(jnp.ones((N0,)))
gamma1 = gamma_max * jnp.float32(jnp.ones((N1,)))
gamma2 = gamma_max * jnp.float32(jnp.ones((N2,)))
gamma3 = gamma_max * jnp.float32(jnp.ones((N3,)))
gamma4 = gamma_max * jnp.float32(jnp.ones((N4,)))

fixed_params=(gamma0,gamma1,gamma2,gamma3,gamma4)

omega0 = random_layer0_detuning_scale * (jnp.float32(randomVector(N0)) - 1)
omega1 = random_detuning_scale * (jnp.float32(randomVector(N1)) - .5)
omega2 = random_detuning_scale * (jnp.float32(randomVector(N2)) - .5)
omega3 = random_detuning_scale * (jnp.float32(randomVector(N3)) - .5)
omega4 = random_detuning_scale * (jnp.float32(randomVector(N4)) - .5)

# JJ3 is a full weight matrix that connects layers 3 and 4 (output)
JJ3 = random_scale * jnp.sqrt(6/(N3+N4)) * jnp.float32(randomJJ(N3, N4) - .5)

# JJ2 is a full weight matrix that connects layers 2 and 3
JJ2 = random_scale * jnp.sqrt(6/(N2+N3)) * jnp.float32(randomJJ(N2, N3) - .5)

# kernel matrix connecting layer 0 to layer 1
parameters10 = jnp.array(random_kernel_scale *np.random.randn(channels_1,channels_0,4,4))
parameters10 = parameters10.flatten()
num_parameters10 = len(parameters10)

# kernel matrix connecting layer 1 to layer 2
parameters21 = jnp.array(random_kernel_scale *np.random.randn(channels_2,channels_1,4,4))
parameters21 = parameters21.flatten()
num_parameters21 = len(parameters21)

params=(parameters10,parameters21,JJ2,JJ3,omega0,omega1,omega2,omega3,omega4)



# running the training

name="results"
optimizer_func=optimizersched

print(name)

params_fixed=(gamma0, gamma1, gamma2,gamma3,gamma4)
optimizer=optimizer_func
opt_state=optimizer.init(params)

FILENAME=name

cost_values=[0.0]
params,opt_state=train(params_fixed,params,optimizer,opt_state,train_batches,cost_values)

# output cost values:

plt.plot(cost_values)
plt.title("cost "+name)
plt.savefig(DIRNAME+"/"+FILENAME+"_cost.pdf")
plt.show()

np.savetxt(DIRNAME+"/"+FILENAME+"_cost.csv", cost_values, delimiter=",")

# output confusion matrix:

conf_matrix=get_confusion_matrix(params,fixed_params,samplesize=100)

plot_confusion_matrix(name,conf_matrix)
plt.savefig(DIRNAME+"/"+FILENAME+"_confusion.pdf")
plt.show()

np.savetxt(DIRNAME+"/"+FILENAME+"_confusion.csv", conf_matrix, delimiter=",")

# output total accuracy

accuracy=np.array([np.mean(np.diag(conf_matrix))])
np.savetxt(DIRNAME+"/"+FILENAME+"_accuracy.csv", accuracy, delimiter=",")
