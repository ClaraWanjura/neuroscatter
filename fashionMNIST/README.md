Code for training on the fashion MNIST data set. 
Run each of the python files in a separate folder to train. Accuracies will be printed to standard output,  but also saved separately inside the subfolder outputs/ (which has to exist).

We distinguish: 

- fully connected (fc) vs. convolutional (conv)
- neuromorphic scattering setup (scatt) vs. standard digital artificial neural network (ANN)
- large images ('large', 28x28) vs small, scaled-down images ('small', 14x14)

Network architectures:

- fully connected, large: 784 - 784 - 10
- fully connected, small: 196 - 196 - 10
- convolutional, large: (28x28,1) - (14x14,10) - (7x7,16) - 100 - 10
- convolutional, small: (14x14,1) - (7x7,10) - (3x3,16) - 100 - 10
  
