import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    HH = filter_size
    WW = filter_size
    F = num_filters
    
    self.params['W1'] = weight_scale * np.random.randn(F, C, HH, WW)
    self.params['b1'] = np.zeros(F)
    
    # stride and padding so that the output has the same dimension
    stride = 1
    pad = (filter_size - 1) / 2
    Hc = (H - HH + 2*pad) / stride + 1
    Wc = (W - WW + 2*pad) / stride + 1
    
    # output from conv layer has size (F, Hc, Wc)
    # relu doesn't change size
    s_pool = 2
    h_pool = 2
    w_pool = 2
    Hp = (Hc - h_pool) / s_pool + 1
    Wp = (Wc - w_pool) / s_pool + 1
    
    # output from pooling layer has size (F, Hp, Wp)
    H2 = hidden_dim
    self.params['W2'] = weight_scale * np.random.randn(F*Hp*Wp, H2)
    self.params['b2'] = np.zeros(H2)
    
    # output from first hidden layer has size H2
    Hscore = num_classes
    self.params['W3'] = weight_scale * np.random.randn(H2, Hscore)
    self.params['b3'] = np.zeros(Hscore)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # first layers conv-relu-pool
    X_conv, cache_conv = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    N, F, Hp, Wp = X_conv.shape
    
    # reshape for next layer
    X_conv = X_conv.reshape((N, F*Hp*Wp))
    
    # second layers affine-relu
    X2, cache2 = affine_relu_forward(X_conv, W2, b2)
    
    # third layers affine
    scores, cache_scores = affine_forward(X2, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg * np.sum(W1 * W1)
    loss += 0.5 * self.reg * np.sum(W2 * W2)
    loss += 0.5 * self.reg * np.sum(W3 * W3)
    
    # first backward pass, affine backward
    dx2, dw3, db3 = affine_backward(dscores, cache_scores)
    grads['W3'] = dw3 + self.reg * W3
    grads['b3'] = db3
    
    # second backward pass, affine-relu
    dx_conv, dw2, db2 = affine_relu_backward(dx2, cache2)
    grads['W2'] = dw2 + self.reg * W2
    grads['b2'] = db2
    
    # third backward pass, conv-relu-pool backward
    dx_conv = dx_conv.reshape((N, F, Hp, Wp))
    dx, dw1, db1 = conv_relu_pool_backward(dx_conv, cache_conv)
    grads['W1'] = dw1 + self.reg * W1
    grads['b1'] = db1
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
