# -*- coding: utf-8 -*-
"""CONVOLUTIONAL ANN version"""

scale_down_images=True # really use 14x14 instead of 28x28
input_scaling=1

# Import necessary libraries

import numpy as np
from numpy import loadtxt

import jax.numpy as jnp
import jax
from jax import jit, vmap, grad, value_and_grad
from jax.scipy.linalg import expm
import jax.lax as lax

import optax

import tensorflow as tf
import keras

import matplotlib.pyplot as plt
import time

if scale_down_images:
    scale_image_factor=4
else:
    scale_image_factor=1


def sigmoid(z):
    return 1./(jnp.exp(-z)+1)

@jit
def network(x, weights, biases):
    # two conv+stride-2 layers, then two fully connected
    # weights[0] and weights[1] are the conv. kernels of shape
    # (out_channels,in_channels,height,width)
    # weights[2] and weights[3] are the fully connected weights
    sx = x.shape
    x = jnp.reshape( x, (sx[0],1,M_pixels,M_pixels) ) # insert channel index
    y = sigmoid( lax.conv( x, weights[0], (2,2), 'SAME') + biases[0][None,:,None,None])
    # trim shape for consistency in terms of parameters with neuromorphic setup!
    y = sigmoid( lax.conv( y, weights[1], (2,2), 'SAME') + biases[1][None,:,None,None])[:,:,:3,:3]
    s=y.shape
    y = jnp.reshape(y, (s[0], s[1]*s[2]*s[3] ) ) # flatten
    y = sigmoid( jnp.dot( y, weights[2] ) + biases[2] ) # weights here are shape (in-dim,out-dim)
    y = jnp.dot( y, weights[3]) + biases[3] 
    return y

@jit
def costF(x, y, weights, biases):
    prob = jnp.log(jax.nn.softmax(network(x, weights, biases)))
    return - jnp.mean( jnp.sum(prob * y,axis=1) )

@jit
def real_costF_grad_all_mean_and_value(x, y, weights, biases):
    return value_and_grad(costF, argnums=[2,3])(x, y, weights, biases)

# loading fashion mnist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)
x_train_vect = jnp.float32(jnp.array(input_scaling * (1 - jnp.array(x_train)/256)))
x_test_vect = jnp.float32(jnp.array(input_scaling * (1 - jnp.array(x_test)/256)))


if scale_down_images:
    # scaling down pictures
    n_train=np.shape(x_train)[0]
    x_train_new = np.empty((n_train,int(784/4)))
    n_test=np.shape(x_test)[0]
    x_test_new = np.empty((n_test,int(784/4)))

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

    x_train_vect = jnp.float32(jnp.array(input_scaling * (1 - jnp.array(x_train_new)/256)))

    x_test_vect = jnp.float32(jnp.array(input_scaling * (1 - jnp.array(x_test_new)/256)))


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

conf_samples_per_fig=1000
pixels_per_fig=int(784/scale_image_factor)
x_test_batch_large=np.empty((conf_samples_per_fig*10,pixels_per_fig),dtype='float32')

for j in range(10):
    indices=np.where(y_test==j)[0]
    x_test_batch_large[j*conf_samples_per_fig:(j+1)*conf_samples_per_fig,:]=x_test_vect[indices[:conf_samples_per_fig],:]

x_test_batch_large=x_test_batch_large

def get_confusion_matrix(params,samplesize=None):

    if samplesize is None:
        samplesize=conf_samples_per_fig
    weights=params[0]
    biases=params[1]

    acc_test_digit=np.zeros((10,10))
    for j in range(10):
        n_sub_samples=int(conf_samples_per_fig/samplesize)
        for sample_idx in range(n_sub_samples):
            the_set = x_test_batch_large[j*conf_samples_per_fig:j*conf_samples_per_fig+samplesize]
            out_network_test = jnp.ravel(network(the_set, weights, biases))
            out_network_test = out_network_test.reshape(len(the_set), 10)
            for k in range(10):
                acc_test_digit[j][k]+= sum(jnp.argmax(out_network_test, axis = 1) == k)/len(the_set)
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

def train(params,optimizer,opt_state,nsteps,cost_values):
    @jax.jit
    def gradient_step(x_batch,y_batch,params, opt_state):
        value, grads = real_costF_grad_all_mean_and_value(x_batch,y_batch,params[0], params[1])
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    j=-1
    iterations_per_batch=60000/batch_size
    accuracies=[]

    for step in range(nsteps):
        j+=1

        if j % iterations_per_batch == 0:
            random_mix_2 = np.random.choice(len(x_train_vect), len(x_train_vect), replace = False)
            j = 0

        batch_subset = random_mix_2[j * batch_size: (j+1) * batch_size]
        x_batch = x_train_vect[batch_subset]
        y_batch = y_train_vect[batch_subset]
        params,opt_state,value=gradient_step(x_batch,y_batch,params,opt_state)
        cost_values.append(float(value))
        if step%300==0 or step==nsteps-1:
            #clear_output(wait=True)
            #visualize_test_batch_params(*params)
            #plt.show()
            print(f"{step:5d}:  cost={value:2.3f}             ")

        if step%iterations_per_batch==0 or step==nsteps-1:
            conf_matrix=get_confusion_matrix(params, samplesize=100)

            plot_confusion_matrix(name,conf_matrix)
            plt.savefig(DIRNAME+"/"+FILENAME+"_confusion.pdf")
            plt.show()

            accuracy=np.mean(np.diag(conf_matrix))
            print("Epoch ", int(step/iterations_per_batch), " Accuracy: ", accuracy)
            accuracies.append(accuracy)

    print("Accuracies:")
    print(accuracies)

    return params,opt_state

# initialization with 'real' parameter types

channels_0=1
channels_1=10
channels_2=16

if scale_down_images:
    linear_scaling=2
else:
    linear_scaling=1
M_pixels=28//linear_scaling
pixels0=M_pixels**2
pixels1=(M_pixels//2)**2
pixels2=(M_pixels//4)**2
N0=pixels0
N1=pixels1*channels_1
N2=pixels2*channels_2

print(pixels1,pixels2)

N3=100
N4=10

weights=( np.random.randn(channels_1,channels_0,4,4),
          np.random.randn(channels_2,channels_1,4,4),
          np.random.randn(N2,N3)*np.sqrt(6/(N3+N2)),
          np.random.randn(N3,N4)*np.sqrt(6/(N4+N3)) )
biases= ( 0.1*np.random.randn(channels_1),
          0.1*np.random.randn(channels_2),
          0.1*np.random.randn(N3),
          0.1*np.random.randn(N4) )

print(sum([w.size for w in weights]) + sum([b.size for b in biases]), " parameters")

batch_size=20

DIRNAME="outputs"

#schedule = optax.warmup_cosine_decay_schedule(
#  init_value=0.0,
#  peak_value=1e-3,
#  warmup_steps=30000,
#  decay_steps=270000,
#  end_value=0.0,
#)

max_learning_rate=0.01

optimizersched = optax.chain(
    optax.amsgrad(learning_rate=max_learning_rate),
    optax.clip(0.001)
)

training_configurations=[
    ["amsgrad_big_sched_300000_batch20_inputscale5",optimizersched,2e-3,10*300_000]
]

for (name,optimizer_func,learningrate,nsteps) in training_configurations:
    print()
    print(name)

    params=(weights,biases)

    optimizer=optimizer_func
    opt_state=optimizer.init(params)

    FILENAME=name

    cost_values=[0.0]
    params,opt_state=train(params,optimizer,opt_state,nsteps,cost_values)

    # output cost values:

    plt.plot(cost_values)
    plt.title("cost "+name)
    plt.savefig(DIRNAME+"/"+FILENAME+"_cost.pdf")
    plt.show()

    np.savetxt(DIRNAME+"/"+FILENAME+"_cost.csv", cost_values, delimiter=",")

    # output confusion matrix:

    conf_matrix=get_confusion_matrix(params, samplesize=100)

    plot_confusion_matrix(name,conf_matrix)
    plt.savefig(DIRNAME+"/"+FILENAME+"_confusion.pdf")
    plt.show()

    np.savetxt(DIRNAME+"/"+FILENAME+"_confusion.csv", conf_matrix, delimiter=",")

    # output total accuracy

    accuracy=np.array([np.mean(np.diag(conf_matrix))])
    np.savetxt(DIRNAME+"/"+FILENAME+"_accuracy.csv", accuracy, delimiter=",")

conf_matrix=get_confusion_matrix(params,samplesize=100)

plot_confusion_matrix(params,conf_matrix)

np.mean(np.diag(conf_matrix)*100)

conf_matrix

(base) fmar@zp11:~/neuromorphic/high_lr_conv_ANN_small> 
(base) fmar@zp11:~/neuromorphic/high_lr_conv_ANN_small> 
(base) fmar@zp11:~/neuromorphic/high_lr_conv_ANN_small> cd ..
(base) fmar@zp11:~/neuromorphic> cd high_lr_conv_ANN_large/
(base) fmar@zp11:~/neuromorphic/high_lr_conv_ANN_large> cat fashion_mnist.py
# -*- coding: utf-8 -*-
"""CONVOLUTIONAL ANN version"""

scale_down_images=False # really use 14x14 instead of 28x28
input_scaling=1

# Import necessary libraries

import numpy as np
from numpy import loadtxt

import jax.numpy as jnp
import jax
from jax import jit, vmap, grad, value_and_grad
from jax.scipy.linalg import expm
import jax.lax as lax

import optax

import tensorflow as tf
import keras

import matplotlib.pyplot as plt
import time

if scale_down_images:
    scale_image_factor=4
else:
    scale_image_factor=1


def sigmoid(z):
    return 1./(jnp.exp(-z)+1)

@jit
def network(x, weights, biases):
    # two conv+stride-2 layers, then two fully connected
    # weights[0] and weights[1] are the conv. kernels of shape
    # (out_channels,in_channels,height,width)
    # weights[2] and weights[3] are the fully connected weights
    sx = x.shape
    x = jnp.reshape( x, (sx[0],1,M_pixels,M_pixels) ) # insert channel index
    y = sigmoid( lax.conv( x, weights[0], (2,2), 'SAME') + biases[0][None,:,None,None])
    y = sigmoid( lax.conv( y, weights[1], (2,2), 'SAME') + biases[1][None,:,None,None])
    s=y.shape
    y = jnp.reshape(y, (s[0], s[1]*s[2]*s[3] ) ) # flatten
    y = sigmoid( jnp.dot( y, weights[2] ) + biases[2] ) # weights here are shape (in-dim,out-dim)
    y = jnp.dot( y, weights[3]) + biases[3] 
    return y

@jit
def costF(x, y, weights, biases):
    prob = jnp.log(jax.nn.softmax(network(x, weights, biases)))
    return - jnp.mean( jnp.sum(prob * y,axis=1) )

@jit
def real_costF_grad_all_mean_and_value(x, y, weights, biases):
    return value_and_grad(costF, argnums=[2,3])(x, y, weights, biases)

# loading fashion mnist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)
x_train_vect = jnp.float32(jnp.array(input_scaling * (1 - jnp.array(x_train)/256)))
x_test_vect = jnp.float32(jnp.array(input_scaling * (1 - jnp.array(x_test)/256)))


if scale_down_images:
    # scaling down pictures
    n_train=np.shape(x_train)[0]
    x_train_new = np.empty((n_train,int(784/4)))
    n_test=np.shape(x_test)[0]
    x_test_new = np.empty((n_test,int(784/4)))

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

    x_train_vect = jnp.float32(jnp.array(input_scaling * (1 - jnp.array(x_train_new)/256)))

    x_test_vect = jnp.float32(jnp.array(input_scaling * (1 - jnp.array(x_test_new)/256)))


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

# visualize_test_batch
# to be used during training

# create a test batch that is nicely ordered
# first samples_per_fig will be '0' etc
samples_per_fig=5
pixels_per_fig=int(784/scale_image_factor)
x_test_batch=np.empty((samples_per_fig*10,pixels_per_fig),dtype='float32')

for j in range(10):
    indices=np.where(y_test==j)[0]
    x_test_batch[j*samples_per_fig:(j+1)*samples_per_fig,:]=x_test_vect[indices[:samples_per_fig],:]

x_test_batch=x_test_batch

conf_samples_per_fig=1000
pixels_per_fig=int(784/scale_image_factor)
x_test_batch_large=np.empty((conf_samples_per_fig*10,pixels_per_fig),dtype='float32')

for j in range(10):
    indices=np.where(y_test==j)[0]
    x_test_batch_large[j*conf_samples_per_fig:(j+1)*conf_samples_per_fig,:]=x_test_vect[indices[:conf_samples_per_fig],:]

x_test_batch_large=x_test_batch_large

def get_confusion_matrix(params,samplesize=None):

    if samplesize is None:
        samplesize=conf_samples_per_fig
    weights=params[0]
    biases=params[1]

    acc_test_digit=np.zeros((10,10))
    for j in range(10):
        n_sub_samples=int(conf_samples_per_fig/samplesize)
        for sample_idx in range(n_sub_samples):
            the_set = x_test_batch_large[j*conf_samples_per_fig:j*conf_samples_per_fig+samplesize]
            out_network_test = jnp.ravel(network(the_set, weights, biases))
            out_network_test = out_network_test.reshape(len(the_set), 10)
            for k in range(10):
                acc_test_digit[j][k]+= sum(jnp.argmax(out_network_test, axis = 1) == k)/len(the_set)
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

def train(params,optimizer,opt_state,nsteps,cost_values):
    @jax.jit
    def gradient_step(x_batch,y_batch,params, opt_state):
        value, grads = real_costF_grad_all_mean_and_value(x_batch,y_batch,params[0], params[1])
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    j=-1
    iterations_per_batch=60000/batch_size
    accuracies=[]

    for step in range(nsteps):
        j+=1

        if j % iterations_per_batch == 0:
            random_mix_2 = np.random.choice(len(x_train_vect), len(x_train_vect), replace = False)
            j = 0

        batch_subset = random_mix_2[j * batch_size: (j+1) * batch_size]
        x_batch = x_train_vect[batch_subset]
        y_batch = y_train_vect[batch_subset]
        params,opt_state,value=gradient_step(x_batch,y_batch,params,opt_state)
        cost_values.append(float(value))
        if step%300==0 or step==nsteps-1:
            #clear_output(wait=True)
            #visualize_test_batch_params(*params)
            #plt.show()
            print(f"{step:5d}:  cost={value:2.3f}             ")

        if step%iterations_per_batch==0 or step==nsteps-1:
            conf_matrix=get_confusion_matrix(params, samplesize=100)

            plot_confusion_matrix(name,conf_matrix)
            plt.savefig(DIRNAME+"/"+FILENAME+"_confusion.pdf")
            plt.show()

            accuracy=np.mean(np.diag(conf_matrix))
            print("Epoch ", int(step/iterations_per_batch), " Accuracy: ", accuracy)
            accuracies.append(accuracy)

    print("Accuracies:")
    print(accuracies)

    return params,opt_state

# initialization with 'real' parameter types

channels_0=1
channels_1=10
channels_2=16

M_pixels=28
pixels0=28*28
pixels1=14*14
pixels2=7*7
N0=pixels0
N1=pixels1*channels_1
N2=pixels2*channels_2

N3=100
N4=10

weights=( np.random.randn(channels_1,channels_0,4,4),
          np.random.randn(channels_2,channels_1,4,4),
          np.random.randn(N2,N3)*np.sqrt(6/(N3+N2)),
          np.random.randn(N3,N4)*np.sqrt(6/(N4+N3)) )
biases= ( 0.1*np.random.randn(channels_1),
          0.1*np.random.randn(channels_2),
          0.1*np.random.randn(N3),
          0.1*np.random.randn(N4) )

batch_size=20

DIRNAME="outputs"

#schedule = optax.warmup_cosine_decay_schedule(
#  init_value=0.0,
#  peak_value=1e-3,
#  warmup_steps=30000,
#  decay_steps=270000,
#  end_value=0.0,
#)
max_learning_rate=0.01

optimizersched = optax.chain(
    optax.amsgrad(learning_rate=max_learning_rate),
    optax.clip(0.001)
)

training_configurations=[
    ["amsgrad_big_sched_300000_batch20_inputscale5",optimizersched,2e-3,10*300_000]
]

for (name,optimizer_func,learningrate,nsteps) in training_configurations:
    print()
    print(name)

    params=(weights,biases)

    optimizer=optimizer_func
    opt_state=optimizer.init(params)

    FILENAME=name

    cost_values=[0.0]
    params,opt_state=train(params,optimizer,opt_state,nsteps,cost_values)

    # output cost values:

    plt.plot(cost_values)
    plt.title("cost "+name)
    plt.savefig(DIRNAME+"/"+FILENAME+"_cost.pdf")
    plt.show()

    np.savetxt(DIRNAME+"/"+FILENAME+"_cost.csv", cost_values, delimiter=",")

    # output confusion matrix:

    conf_matrix=get_confusion_matrix(params, samplesize=100)

    plot_confusion_matrix(name,conf_matrix)
    plt.savefig(DIRNAME+"/"+FILENAME+"_confusion.pdf")
    plt.show()

    np.savetxt(DIRNAME+"/"+FILENAME+"_confusion.csv", conf_matrix, delimiter=",")

    # output total accuracy

    accuracy=np.array([np.mean(np.diag(conf_matrix))])
    np.savetxt(DIRNAME+"/"+FILENAME+"_accuracy.csv", accuracy, delimiter=",")

conf_matrix=get_confusion_matrix(params,samplesize=100)

plot_confusion_matrix(params,conf_matrix)

np.mean(np.diag(conf_matrix)*100)

conf_matrix
