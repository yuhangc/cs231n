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


class GeneralConvNet(object):
  """
  A general convolutional network with the following architecture:
  
  (conv - relu - conv - [batchnorm] - relu - 2x2 max pool) * L - (affine - relu) * M - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim, num_filters, filter_size,
               hidden_dims, num_classes=10, use_batchnorm=False, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: List (L, ) of number of filters to use in the convolutional layers
    - filter_size: List (L, ) of sizes of filters to use in the convolutional layers
    - hidden_dim: List of (M, ) number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.L = len(num_filters)
    self.M = len(hidden_dims)
    self.num_filters = num_filters
    self.filter_size = filter_size
    self.hidden_dims = hidden_dims

    # initialize weights for convolutional layers
    C, H, W = input_dim
    L = self.L
    M = self.M
    F = self.num_filters
    
    F = [C] + F
    Hc = np.zeros(L + 1, dtype=int)
    Wc = np.zeros(W + 1, dtype=int)
    Hc[0] = H
    Wc[0] = W
    
    s_pool = 2
    h_pool = 2
    w_pool = 2

    for i in xrange(L):
        # use padding and stride that preserves the shape
        HH = self.filter_size[i]
        WW = self.filter_size[i]
        self.params['W1conv' + str(i+1)] = weight_scale * np.random.randn(F[i+1], F[i], HH, WW)
        self.params['b1conv' + str(i+1)] = np.zeros(F[i+1])
        self.params['W2conv' + str(i+1)] = weight_scale * np.random.randn(F[i+1], F[i+1], HH, WW)
        self.params['b2conv' + str(i+1)] = np.zeros(F[i+1])
        
        # output size from one conv-conv-pool layer
        Hc[i+1] = (Hc[i] - h_pool) / s_pool + 1
        Wc[i+1] = (Wc[i] - w_pool) / s_pool + 1
        
    # initialize batchnorm parameters
    self.bn_params = {}
    if self.use_batchnorm:
        self.bn_params = {'bn_param' + str(i+1): {'mode': 'train'} for i in xrange(self.L)}
        gammas = {'gamma' + str(i+1): np.ones(num_filters[i]) for i in xrange(self.L)}
        betas = {'beta' + str(i+1): np.zeros(num_filters[i]) for i in xrange(self.L)}
      
        self.params.update(gammas)
        self.params.update(betas)
        
    # initialize weights for the affine layers
    Ha = [F[L] * Hc[L] * Wc[L]] + hidden_dims
    
    for i in xrange(M):
        self.params['Waffine' + str(i+1)] = weight_scale * np.random.randn(Ha[i], Ha[i+1])
        self.params['baffine' + str(i+1)] = np.zeros(Ha[i+1])
    
    # final affine layer for score
    self.params['Wscore'] = weight_scale * np.random.randn(Ha[M], num_classes)
    self.params['bscore'] = np.zeros(num_classes)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'
    
    # Set train/test mode for batchnorm params
    if self.use_batchnorm:
        for key, bn_param in self.bn_params.iteritems():
            bn_param['mode'] = mode

    scores = None
    # forward pass of the convolutional layers
    x_conv = {}
    cache_conv = {}
    
    x_conv['x0'] = X
    for i in xrange(self.L):
        w1 = self.params['W1conv' + str(i+1)]
        b1 = self.params['b1conv' + str(i+1)]
        w2 = self.params['W2conv' + str(i+1)]
        b2 = self.params['b2conv' + str(i+1)]
        x_in = x_conv['x' + str(i)]
        
        conv_param = {'stride': 1, 'pad': (self.filter_size[i] - 1) / 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        if self.use_batchnorm:
            gamma = self.params['gamma' + str(i+1)]
            beta = self.params['beta' + str(i+1)]
            bn_param = self.bn_params['bn_param' + str(i+1)]
            x_out, cache = general_conv_layer_forward(x_in, w1, b1, w2, b2, 
                                                      conv_param, gamma, beta, bn_param, pool_param)
            
            x_conv['x' + str(i+1)] = x_out
            cache_conv['x' + str(i+1)] = cache
        else:
            pass
    
    # forward pass of the linear layers
    x_affine = {}
    cache_affine = {}
    
    x_in = x_conv['x' + str(self.L)]
    N, F, H, W = x_in.shape
    x_affine['x0'] = x_in.reshape(N, F * H * W)
    for i in xrange(self.M):
        w = self.params['Waffine' + str(i+1)]
        b = self.params['baffine' + str(i+1)]
        x_in = x_affine['x' + str(i)]
        
        x_out, cache = affine_relu_forward(x_in, w, b)
        
        x_affine['x' + str(i+1)] = x_out
        cache_affine['x' + str(i+1)] = cache
    
    # final forward pass to get the scores
    w = self.params['Wscore']
    b = self.params['bscore']
    
    scores, cache_scores = affine_forward(x_affine['x' + str(self.M)], w, b)
        
    ############################################################################
    #                            END OF FORWARD PASS                           #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    
    # caluclate the loss based on softmax
    loss, dscores = softmax_loss(scores, y)
    
    # add in L2 regulation loss
    for i in xrange(self.L):
        w1 = self.params['W1conv' + str(i+1)]
        b1 = self.params['b1conv' + str(i+1)]
        w2 = self.params['W2conv' + str(i+1)]
        b2 = self.params['b2conv' + str(i+1)]
        loss += 0.5 * self.reg * np.sum(w1 * w1)
        loss += 0.5 * self.reg * np.sum(b1 * b1)
        loss += 0.5 * self.reg * np.sum(w2 * w2)
        loss += 0.5 * self.reg * np.sum(b2 * b2)
        
    for i in xrange(self.M):
        w = self.params['Waffine' + str(i+1)]
        b = self.params['baffine' + str(i+1)]
        loss += 0.5 * self.reg * (np.sum(w * w) + np.sum(b * b))
    
    w = self.params['Wscore']
    b = self.params['bscore']
    loss += 0.5 * self.reg * (np.sum(w * w) + np.sum(b * b))
    
    # affine score layer backward pass
    dout, dw, db = affine_backward(dscores, cache_scores)
    grads['Wscore'] = dw + self.reg * w
    grads['bscore'] = db
    
    # backward pass of affine layers
    x_affine['dx' + str(self.M)] = dout
    for i in range(self.M)[::-1]:
        dout = x_affine['dx' + str(i+1)]
        cache = cache_affine['x' + str(i+1)]
        w = self.params['Waffine' + str(i+1)]
        
        dx, dw, db = affine_relu_backward(dout, cache)
        
        x_affine['dx' + str(i)] = dx
        grads['Waffine' + str(i+1)] = dw + self.reg * w
        grads['baffine' + str(i+1)] = db
        
    # backward pass for convolutional layers
    x_conv['dx' + str(self.L)] = x_affine['dx0'].reshape(N, F, H, W)
    for i in range(self.L)[::-1]:
        dout = x_conv['dx' + str(i+1)]
        cache = cache_conv['x' + str(i+1)]
        w1 = self.params['W1conv' + str(i+1)]
        b1 = self.params['b1conv' + str(i+1)]
        w2 = self.params['W2conv' + str(i+1)]
        b2 = self.params['b2conv' + str(i+1)]
        
        if self.use_batchnorm:
            dx, dw1, db1, dw2, db2, dgamma, dbeta = general_conv_layer_backward(dout, cache)
            
            x_conv['dx' + str(i)] = dx
            grads['W1conv' + str(i+1)] = dw1 + self.reg * w1
            grads['b1conv' + str(i+1)] = db1
            grads['W2conv' + str(i+1)] = dw2 + self.reg * w2
            grads['b2conv' + str(i+1)] = db2
            grads['gamma' + str(i+1)] = dgamma
            grads['beta' + str(i+1)] = dbeta
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads

# forward and backward functions for sandwich layers
def conv_batch_relu_forward(x, w, b, conv_param, gamma, beta, bn_param):
    xc, cache_conv = conv_forward_fast(x, w, b, conv_param)
    xb, cache_batchnorm = spatial_batchnorm_forward(xc, gamma, beta, bn_param)
    out, cache_relu = relu_forward(xb)
    cache = (cache_conv, cache_batchnorm, cache_relu)
    
    return out, cache

def conv_batch_relu_backward(dout, cache):
    cache_conv, cache_batchnorm, cache_relu = cache
    
    dxb = relu_backward(dout, cache_relu)
    dxc, dgamma, dbeta = spatial_batchnorm_backward(dxb, cache_batchnorm)
    dx, dw, db = conv_backward_fast(dxc, cache_conv)
    
    return dx, dw, db, dgamma, dbeta

def general_conv_layer_forward(x, w1, b1, w2, b2, conv_param, gamma, beta, bn_param, pool_param):
    xc1, cache_conv1 = conv_relu_forward(x, w1, b1, conv_param)
    xc2, cache_conv2 = conv_batch_relu_forward(xc1, w2, b2, conv_param, gamma, beta, bn_param)
    out, cache_pool = max_pool_forward_fast(xc2, pool_param)
    
    cache = (cache_conv1, cache_conv2, cache_pool)
    
    return out, cache

def general_conv_layer_backward(dout, cache):
    cache_conv1, cache_conv2, cache_pool = cache
    
    dxc2 = max_pool_backward_fast(dout, cache_pool)
    dxc1, dw2, db2, dgamma, dbeta = conv_batch_relu_backward(dxc2, cache_conv2)
    dx, dw1, db1 = conv_relu_backward(dxc1, cache_conv1)
    
    return dx, dw1, db1, dw2, db2, dgamma, dbeta
  
pass


