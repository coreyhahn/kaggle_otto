# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:10:40 2015

@author: cah
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 12:21:45 2015

@author: cah
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 22:04:35 2015

@author: cah
"""

#import re
import kaggle_csv as kcsv
import numpy as np
import theano, sys, math
import hdf5data
from theano import tensor as T
from theano import shared
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict
import cPickle
from skimage.transform import rotate as rotate
import datetime

BATCH_SIZE=1024
 
def relu_f(vec):
    """ Wrapper to quickly change the rectified linear unit function """
    return (vec + abs(vec)) / 2.
         
def dropout(rng, x, p=0.5):
    """ Zero-out random values in x with probability p using rng """
    if p > 0. and p < 1.:
        seed = rng.randint(2 ** 30)
        srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
        mask = srng.binomial(n=1, p=1.-p, size=x.shape,
                dtype='float32') #change
        return T.cast(x * mask,'float32')
    return  T.cast(x,'float32')
         
def fast_dropout(rng, x):
    """ Multiply activations by N(1,1) """
    seed = rng.randint(2 ** 30)
    srng = RandomStreams(seed)
    mask = srng.normal(size=x.shape, avg=1., dtype='float32') #change
    return T.cast(x * mask,'float32')
         
def build_shared_zeros(shape, name):
    """ Builds a theano shared variable filled with a zeros numpy array """
    return shared(value=np.zeros(shape, dtype='float32'), #change
            name=name, borrow=True)
         
 
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """
 
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
            filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                                image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
        print (image_shape[1],filter_shape[1])
        assert image_shape[1] == filter_shape[1]
        self.input = input
 
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype='float32'),
            borrow=True)
 
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype='float32')
        self.b = theano.shared(value=b_values, borrow=True)
 
        # convolve input feature maps with filters
        conv_out = T.cast(conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape),'float32')
 
        # downsample each feature map individually, using maxpooling
        pooled_out = T.cast(downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True),'float32')
 
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + T.cast(self.b.dimshuffle('x', 0, 'x', 'x'),'float32'))
        # TODO relu output of convolutions
        #self.output = relu_f(pooled_out + T.cast(self.b.dimshuffle('x', 0, 'x', 'x'),'float32')) 
        
        # store parameters of this layer
        self.params = [self.W, self.b]
    def __repr__(self):
        return "ConvPoolLayer" #might have to change this
 
 
 
class ReLU(object):
    """ Basic rectified-linear transformation layer (W.X + b) 
        Multipurpose"""
    def __init__(self, rng, input, n_in, n_out, drop_out=0.0, W=None, b=None, fdrop=False):
        if W is None:
            W_values = np.asarray(rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype='float32')
            W_values *= 4  
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b = build_shared_zeros((n_out,), 'b')
        self.input = input
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.output = T.dot(self.input, self.W) + self.b
        self.pre_activation = self.output
        if drop_out:
            if fdrop:
                self.pre_activation = fast_dropout(rng, self.pre_activation)
                self.output = relu_f(self.pre_activation) 
            else:
                self.W=W * 1. / (1. - dr)   
                self.b=b * 1. / (1. - dr)
                self.output = dropout(numpy_rng, self.output, dr)
                self.output = relu_f(self.pre_activation) 
        else:
            self.output = relu_f(self.pre_activation) 
    def __repr__(self):
        return "ReLU"
 
class DatasetMiniBatchIterator(object):
    def __init__(self, x, y, batch_size=BATCH_SIZE, randomize=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.randomize = randomize
        from sklearn.utils import check_random_state
        self.rng = check_random_state(42)
 
    def __iter__(self):
        n_samples = self.x.shape[0]
        if self.randomize:
            for _ in xrange(n_samples / BATCH_SIZE):
                if BATCH_SIZE > 1:
                    i = int(self.rng.rand(1) * ((n_samples+BATCH_SIZE-1) / BATCH_SIZE))
                else:
                    i = int(math.floor(self.rng.rand(1) * n_samples))
                yield (i, self.x[i*self.batch_size:(i+1)*self.batch_size],self.y[i*self.batch_size:(i+1)*self.batch_size])
        else:
#            for i in xrange((n_samples + self.batch_size - 1) / self.batch_size):
            for i in xrange((n_samples) / self.batch_size):
                tempx = self.x[i*self.batch_size:(i+1)*self.batch_size]
                yield (tempx,self.y[i*self.batch_size:(i+1)*self.batch_size])
            
            tempx = np.vstack((self.x[(i+1)*self.batch_size:],self.x[:self.batch_size-(self.x.shape[0]-(i+1)*self.batch_size)]))         
            tempy = np.hstack((self.y[(i+1)*self.batch_size:],self.y[:self.batch_size-(self.y.shape[0]-(i+1)*self.batch_size)]))   
            yield (tempx,tempy)

class DatasetMiniBatchIteratorEven(object):
    def __init__(self, x, y, batch_size=BATCH_SIZE, randomize=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.randomize = randomize
        from sklearn.utils import check_random_state
        self.rng = check_random_state(42)
 
    def __iter__(self):
        n_samples = self.x.shape[0]
        (h1, h2) = np.histogram(self.y,len(set(self.y)))
        norm1 = np.int32(h1*0.0)
        for n in np.arange(norm1.size):
            norm1[n] = np.int32(np.floor(1.0*np.max(h1)/h1[n]))
        script = []
        for n in np.arange(n_samples):
            script.append((n,0,0)) #normal add
            for m in np.arange(norm1[self.y[n]]-1):
                script.append((n,1,0))
        if self.randomize == True:
            self.rng.shuffle(script)
        rem = len(script) % self.batch_size
        script.extend(script[0:self.batch_size-rem])
        
        batchx = np.float32(np.zeros((self.batch_size,self.x.shape[1]))) 
        batchy = np.int32(np.zeros((self.batch_size)))  
        batchcount = 0
        for s1 in script:
            if s1[1] == 1:
                t1 = self.randshift(self.x[s1[0]])
            else:
                t1 = self.x[s1[0]]
            batchx[batchcount,:] = t1
            batchy[batchcount] = self.y[s1[0]]
            batchcount += 1
            if batchcount == self.batch_size:
                yield (batchx,batchy)
                batchcount = 0
        

    def randshift(self, x):
        squsize = np.int32(np.sqrt(x.size/3))
        x = np.reshape(x,(3,squsize,squsize))
        if self.rng.rand() <.5:
            x[0,:,:] = np.fliplr(x[0,:,:])
            x[1,:,:] = np.fliplr(x[1,:,:])
            x[2,:,:] = np.fliplr(x[2,:,:])
    
        rowshift = self.rng.randint(0,10);
        colshift = self.rng.randint(0,10);
        x1 = np.zeros((3,x.shape[1]+10,x.shape[2]+10),dtype=np.float32)
        x1[:,rowshift:rowshift+x.shape[1], colshift:colshift+x.shape[2]] = x
        x = x1[:,5:-5,5:-5]
        x = np.reshape(x,(1,-1))
        
        return x
    
class DatasetMiniBatchIteratorRnd(object):
    def __init__(self, x, y, batch_size=BATCH_SIZE, multiply_size=512, randomize=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.batch_multiply = multiply_size
        self.multrndstate = 42
        self.randomize = randomize
        from sklearn.utils import check_random_state
        self.rng = check_random_state(42)
        self.rng2 = check_random_state(self.multrndstate)
        
    def __iter__(self):
        n_samples = self.x.shape[0]
        if self.randomize:
            for _ in xrange(n_samples / BATCH_SIZE):
                if BATCH_SIZE > 1:
                    i = int(self.rng.rand(1) * ((n_samples+BATCH_SIZE-1) / BATCH_SIZE))
                else:
                    i = int(math.floor(self.rng.rand(1) * n_samples))
                yield (i, self.x[i*self.batch_size:(i+1)*self.batch_size],self.y[i*self.batch_size:(i+1)*self.batch_size])
        else:
#            for i in xrange((n_samples + self.batch_size - 1) / self.batch_size):
            from sklearn.utils import check_random_state
            self.rng2 = check_random_state(self.multrndstate)
#            print self.rng2.randint(0,1e6)
            for i in xrange((n_samples) / self.batch_size):
                tempx = self.x[i*self.batch_size:(i+1)*self.batch_size]
                yield (tempx,self.y[i*self.batch_size:(i+1)*self.batch_size])
            tempx = np.vstack((self.x[(i+1)*self.batch_size:],self.x[:self.batch_size-(self.x.shape[0]-(i+1)*self.batch_size)]))         
            tempy = np.hstack((self.y[(i+1)*self.batch_size:],self.y[:self.batch_size-(self.y.shape[0]-(i+1)*self.batch_size)]))   
            yield (tempx,tempy) 
            
            for i in xrange((n_samples) / self.batch_size *(self.batch_multiply-1)):
                indx = np.arange(i*self.batch_size,(i+1)*self.batch_size) % self.x.shape[0]
                tempx = self.x[indx]
                yield (self.rnd_shift((tempx)),self.y[indx])
            
#            tempx = np.vstack((self.x[(i+1)*self.batch_size:],self.x[:self.batch_size-(self.x.shape[0]-(i+1)*self.batch_size)]))         
#            tempy = np.hstack((self.y[(i+1)*self.batch_size:],self.y[:self.batch_size-(self.y.shape[0]-(i+1)*self.batch_size)]))   
#            yield (self.rnd_shift(tempx),tempy)
 
    def rnd_shift(self, x_a):
        for n in np.arange(x_a.shape[0]):
            x = x_a[n,:]
            squsize = np.int32(np.sqrt(x.size))
            x = np.reshape(x,(squsize,squsize))
            rowshift = self.rng2.randint(0,20);
            colshift = self.rng2.randint(0,20);
            x1 = np.zeros((x.shape[0]+20,x.shape[1]+20),dtype=np.float32)+1.0
            x1[rowshift:rowshift+x.shape[0], colshift:colshift+x.shape[1]] = x
            x = x1[10:-10,10:-10]
            x_a[n,:] = np.reshape(x,(1,x.size))
        return x_a
         
    def rnd_rot(self, x_a):
        for n in np.arange(x_a.shape[0]):
            x = x_a[n,:]
            squsize = np.int32(np.sqrt(x.size))
            x = np.reshape(x,(squsize,squsize))
            a1 = self.rng2.rand()*90
            x = rotate(x,a1)
            x_a[n,:] = np.reshape(np.float32(x),(1,x.size))
        return x_a
                
        
class LogisticRegression:
    """Multi-class Logistic Regression (no dropout in this layer)
    """
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        if W != None:
            self.W = W
        else:
            self.W = build_shared_zeros((n_in, n_out), 'W')
        if b != None:
            self.b = b
        else:
            self.b = build_shared_zeros((n_out,), 'b')
 
        # P(Y|X) = softmax(W.X + b)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        #this is the prediction. pred
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.output = self.y_pred
        self.params = [self.W, self.b]
 
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
 
    def negative_log_likelihood_sum(self, y):
        return -T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
 
    def training_cost(self, y):
        """ Wrapper for standard name """
        return self.negative_log_likelihood_sum(y)
 
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                ("y", y.type, "y_pred", self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            print("!!! y should be of int type")
            return T.mean(T.neq(self.y_pred, np.asarray(y, dtype='int')))
 
    def prediction(self, input):
        return self.y_pred
        
    def pred_soft(self, input):
        return self.p_y_given_x

class ConvDropNet02(object):
    """ Convolutional Neural network class 
        Given the parameters for each problem. This class is left to be more customizeable
        and almost acts as function


    """
    def __init__(self, numpy_rng, theano_rng=None, 
                 n_ins=48*48,
                 conv_reshaper=(BATCH_SIZE, 1, 48, 48),
                 batch_size=BATCH_SIZE,
                 Conv ={'image_shape1':(BATCH_SIZE, 1, 48, 48),'image_shape2':(BATCH_SIZE, 20, 22, 22),'image_shape3':(BATCH_SIZE, 40, 10, 10),'image_shape4':(BATCH_SIZE, 60, 4, 4)},
                 filters={'filter_shape1':(20, 1, 5, 5),'filter_shape2':(40, 20, 3, 3),'filter_shape3':(60, 40, 3, 3)},
                 poolsize=(2,2),
                 layers_types=[LeNetConvPoolLayer,LeNetConvPoolLayer, LeNetConvPoolLayer, ReLU, ReLU, ReLU, LogisticRegression],
                 layers_sizes=['NA', 'NA', 'NA', 512, 256, 256], 
                 n_outs=121, 
                 rho=0.98,
                 eps=1.E-6,
                 max_norm=0.,
                 debugprint=True,
                 fast_drop=True,
                 dropout_rates=[0., 0., 0., 0.5, 0.5, 0.5, 0.] #match this up with actual layers
                 ):
 
 
        self.layers = []
        self.params = []
        self.n_layers = len(layers_types)
        self.layers_types = layers_types
        assert self.n_layers > 0
        self.max_norm = max_norm
        self._rho = rho  # ``momentum'' for adadelta
        self._eps = eps  # epsilon for adadelta
        self._accugrads = []  # for adadelta
        self._accudeltas = []  # for adadelta
        if theano_rng == None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
 
        self.x = T.fmatrix('x')
        self.y = T.ivector('y')
        
        self.layers_ins = [n_ins] + layers_sizes
        self.layers_outs = layers_sizes + [n_outs]
        
        layer_input = self.x
 
 
        self.batch_size = BATCH_SIZE
 
        # Reshape matrix of rasterized images of shape (batch_size,28*28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        conv_layer_input=T.cast(layer_input.reshape(conv_reshaper),'float32') #change later params
        
        #change these for each conv layer, and specify params
        self.poolsize=poolsize
        
 
        self.dropout_rates = dropout_rates
        if fast_drop:
            if dropout_rates[0]:
                dropout_layer_input = fast_dropout(numpy_rng, self.x)
            else:
                dropout_layer_input = self.x
        else:
            dropout_layer_input = dropout(numpy_rng, self.x, p=dropout_rates[0])
        self.dropout_layers = []
 
 
        layer0=LeNetConvPoolLayer(rng=numpy_rng, input=conv_layer_input, 
          filter_shape=filters['filter_shape1'], image_shape=Conv['image_shape1'], 
          poolsize=self.poolsize)
        self.params.extend(layer0.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
            'accugrad') for t in layer0.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
            'accudelta') for t in layer0.params])
        assert hasattr(layer0, 'output')   
        self.dropout_layers.append(layer0)
        dropout_layer_input = T.cast(layer0.output,'float32')
        #print dropout_layer_input
 
        layer1=LeNetConvPoolLayer(rng=numpy_rng,input=dropout_layer_input, filter_shape=filters['filter_shape2'], 
          image_shape=Conv['image_shape2'], poolsize=self.poolsize)
        self.params.extend(layer1.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
           'accugrad') for t in layer1.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
           'accudelta') for t in layer1.params])
        assert hasattr(layer1, 'output')
        self.dropout_layers.append(layer1)
        dropout_layer_input = T.cast(layer1.output,'float32')
        #print dropout_layer_input

        layer1b=LeNetConvPoolLayer(rng=numpy_rng,input=dropout_layer_input, filter_shape=filters['filter_shape3'], 
          image_shape=Conv['image_shape3'], poolsize=self.poolsize)
        self.params.extend(layer1b.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
           'accugrad') for t in layer1b.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
           'accudelta') for t in layer1b.params])
        assert hasattr(layer1b, 'output')
        self.dropout_layers.append(layer1b)
        dropout_layer_input = T.cast(layer1b.output.flatten(2),'float32')
        #print dropout_layer_input
        
        # construct fully-connected ReLU layers
        n_in_array = image_shape=Conv['image_shape4']
        
        layer2= ReLU(rng=numpy_rng, input=dropout_layer_input, drop_out=dropout_rates[3] ,fdrop=True, n_in=np.prod(n_in_array[1:]), n_out=layers_sizes[3])
        self.params.extend(layer2.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
            'accugrad') for t in layer2.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
            'accudelta') for t in layer2.params])
        assert hasattr(layer2, 'output')
        self.dropout_layers.append(layer2)
        dropout_layer_input = layer2.output
        #print dropout_layer_input   
 
        layer3= ReLU(rng=numpy_rng, input=dropout_layer_input, drop_out=dropout_rates[4] , fdrop=True, n_in=layers_sizes[3], n_out=layers_sizes[4])    
        self.params.extend(layer3.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
            'accugrad') for t in layer3.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
            'accudelta') for t in layer3.params])
        assert hasattr(layer3, 'output')
        self.dropout_layers.append(layer3)
        dropout_layer_input = T.cast(layer3.output,'float32')
        #print dropout_layer_input
 
        layer4= ReLU(rng=numpy_rng, input=dropout_layer_input, drop_out=dropout_rates[5] , fdrop=True, n_in=layers_sizes[4], n_out=layers_sizes[5])
        self.params.extend(layer4.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
            'accugrad') for t in layer4.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
            'accudelta') for t in layer4.params])
        assert hasattr(layer4, 'output')
        self.dropout_layers.append(layer4)
        dropout_layer_input = T.cast(layer4.output,'float32')
        #print dropout_layer_input
 
        # classify the values
        layer5= LogisticRegression(rng=numpy_rng, input=dropout_layer_input, n_in=layers_sizes[5], n_out=n_outs)
        self.params.extend(layer5.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
            'accugrad') for t in layer5.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
            'accudelta') for t in layer5.params])
        assert hasattr(layer5, 'output')
        self.dropout_layers.append(layer5)
        dropout_layer_input = T.cast(layer5.output,'float32')
        print dropout_layer_input
 
 
        assert hasattr(self.dropout_layers[-1], 'training_cost')
        assert hasattr(self.dropout_layers[-1], 'errors')
        
        self.mean_cost = self.dropout_layers[-1].negative_log_likelihood(self.y)
        #print self.mean_cost.dtype
        self.cost = self.dropout_layers[-1].training_cost(self.y)
        self.prediction_1 = self.dropout_layers[-1].prediction(self.x)
        self.prediction_soft = self.dropout_layers[-1].pred_soft(self.x)
        if debugprint:
            theano.printing.debugprint(self.cost) 
        # the non-dropout errors
        self.errors = self.dropout_layers[-1].errors(self.y)
    
    def __repr__(self):
        dimensions_layers_str = map(lambda x: "x".join(map(str, x)),
                                    zip(self.layers_ins, self.layers_outs))
        return "_".join(map(lambda x: "_".join((x[0].__name__, x[1])),
                            zip(self.layers_types, dimensions_layers_str))) + "\n"\
        + "dropout rates: " + str(self.dropout_rates)
 
 
    def get_SGD_trainer(self):
        """ Returns a plain SGD minibatch trainer with learning rate as param.
        """
        batch_x = T.fmatrix('batch_x')
        #batch_x= T.matrix('batch_x', dtype=theano.config.floatX)
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        # using mean_cost so that the learning rate is not too dependent
        # on the batch size
        gparams = T.grad(self.mean_cost, self.params)
 
        # compute list of weights updates
        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            if self.max_norm:
                W = param - gparam * learning_rate
                col_norms = W.norm(2, axis=0)
                desired_norms = T.clip(col_norms, 0, self.max_norm)
                updates[param] = W * (desired_norms / (1e-6 + col_norms))
            else:
                updates[param] = param - gparam * learning_rate
 
        train_fn = theano.function(inputs=[theano.Param(batch_x),
                                           theano.Param(batch_y),
                                           theano.Param(learning_rate)],
                                   outputs=self.mean_cost,
                                   updates=updates,
                                   givens={self.x: batch_x, self.y: batch_y})
 
        return train_fn
 
    def get_adagrad_trainer(self):
        """ Returns an Adagrad (Duchi et al. 2010) trainer using a learning rate.
        """
        batch_x = T.fmatrix('batch_x')
        #batch_x= T.matrix('batch_x', dtype=theano.config.floatX)
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.mean_cost, self.params)
 
        # compute list of weights updates
        updates = OrderedDict()
        for accugrad, param, gparam in zip(self._accugrads, self.params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = accugrad + gparam * gparam
            dx = - (learning_rate / T.sqrt(agrad + self._eps)) * gparam
            if self.max_norm:
                W = param + dx
                col_norms = W.norm(2, axis=0)
                desired_norms = T.clip(col_norms, 0, self.max_norm)
                updates[param] = W * (desired_norms / (1e-6 + col_norms))
            else:
                updates[param] = param + dx
            updates[accugrad] = agrad
 
        train_fn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y),
            theano.Param(learning_rate)],
            outputs=self.mean_cost,
            updates=updates,
            givens={self.x: batch_x, self.y: batch_y})
 
        return train_fn
 
    def get_adadelta_trainer(self):
        """ Returns an Adadelta (Zeiler 2012) trainer using self._rho and
        self._eps params.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.mean_cost, self.params)
        # compute list of weights updates
        updates = OrderedDict()
        for accugrad, accudelta, param, gparam in zip(self._accugrads,
                self._accudeltas, self.params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
            agrad=T.cast(agrad, 'float32')
            dx = - T.sqrt((accudelta + self._eps)
                          / (agrad + self._eps)) * gparam
            dx=T.cast(dx, 'float32')
            updates[accudelta] = T.cast((self._rho * accudelta
                                  + (1 - self._rho) * dx * dx),'float32')
            if self.max_norm:
                W = T.cast(param + dx,'float32')
                col_norms = T.cast(W.norm(2, axis=0),'float32')
                desired_norms = T.cast(T.clip(col_norms, 0, self.max_norm),'float32')
                updates[param] = T.cast(W * (desired_norms / (1e-6 + col_norms)),'float32')
            else:
                updates[param] = T.cast(param + dx,'float32')
            updates[accugrad] = T.cast(agrad,'float32')
 
        train_fn = theano.function(inputs=[theano.Param(batch_x),
                                           theano.Param(batch_y)],
                                   outputs=self.mean_cost,
                                   updates=updates,
                                   givens={self.x: batch_x, self.y: batch_y},
                                   allow_input_downcast=True)
 
        return train_fn
 
 
    def score_classif(self, given_set):
        """ Returns functions to get current classification errors. """
        batch_x = T.fmatrix('batch_x')
        #batch_x= T.matrix('batch_x', dtype=theano.config.floatX)
        batch_y = T.ivector('batch_y')
        score = theano.function(inputs=[theano.Param(batch_x),
                                        theano.Param(batch_y)],
                                outputs=self.errors,
                                givens={self.x: batch_x, self.y: batch_y},
                                allow_input_downcast=True)
 
        def scoref():
            """ returned function that scans the entire set given as input """
            return [score(batch_x, batch_y) for batch_x, batch_y in given_set]
 
        return scoref
 
    def predict(self,X):
        """ Returns functions to get current classification errors. """
        batch_x = T.fmatrix('batch_x')
        #batch_x= T.matrix('batch_x', dtype=theano.config.floatX)
        predictor = theano.function(inputs=[theano.Param(batch_x)],
                                outputs=self.prediction_1,
                                givens={self.x: batch_x},
                                allow_input_downcast=True)
        return predictor(X)

    def predict_soft(self,X):
        """ Returns functions to get current classification errors. """
        batch_x = T.fmatrix('batch_x')
        #batch_x= T.matrix('batch_x', dtype=theano.config.floatX)
        predictor1 = theano.function(inputs=[theano.Param(batch_x)],
                                outputs=self.prediction_soft,
                                givens={self.x: batch_x},
                                allow_input_downcast=True)
        return predictor1(X) 
 
 
class ConvDropNet02(object):
    """ Convolutional Neural network class 
        Given the parameters for each problem. This class is left to be more customizeable
        and almost acts as function


    """
    def __init__(self, numpy_rng, theano_rng=None, 
                 n_ins=48*48,
                 conv_reshaper=(BATCH_SIZE, 1, 48, 48),
                 batch_size=BATCH_SIZE,
                 Conv ={'image_shape1':(BATCH_SIZE, 1, 48, 48),'image_shape2':(BATCH_SIZE, 20, 22, 22),'image_shape3':(BATCH_SIZE, 40, 10, 10),'image_shape4':(BATCH_SIZE, 60, 4, 4)},
                 filters={'filter_shape1':(20, 1, 5, 5),'filter_shape2':(40, 20, 3, 3),'filter_shape3':(60, 40, 3, 3)},
                 poolsize=(2,2),
                 layers_types=[LeNetConvPoolLayer,LeNetConvPoolLayer, LeNetConvPoolLayer, ReLU, ReLU, ReLU, LogisticRegression],
                 layers_sizes=['NA', 'NA', 'NA', 512, 256, 256], 
                 n_outs=121, 
                 rho=0.98,
                 eps=1.E-6,
                 max_norm=0.,
                 debugprint=True,
                 fast_drop=True,
                 dropout_rates=[0., 0., 0., 0.5, 0.5, 0.5, 0.] #match this up with actual layers
                 ):
 
 
        self.layers = []
        self.params = []
        self.n_layers = len(layers_types)
        self.layers_types = layers_types
        assert self.n_layers > 0
        self.max_norm = max_norm
        self._rho = rho  # ``momentum'' for adadelta
        self._eps = eps  # epsilon for adadelta
        self._accugrads = []  # for adadelta
        self._accudeltas = []  # for adadelta
        if theano_rng == None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
 
        self.x = T.fmatrix('x')
        self.y = T.ivector('y')
        
        self.layers_ins = [n_ins] + layers_sizes
        self.layers_outs = layers_sizes + [n_outs]
        
        layer_input = self.x
 
 
        self.batch_size = BATCH_SIZE
 
        # Reshape matrix of rasterized images of shape (batch_size,28*28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        conv_layer_input=T.cast(layer_input.reshape(conv_reshaper),'float32') #change later params
        
        #change these for each conv layer, and specify params
        self.poolsize=poolsize
        
 
        self.dropout_rates = dropout_rates
        if fast_drop:
            if dropout_rates[0]:
                dropout_layer_input = fast_dropout(numpy_rng, self.x)
            else:
                dropout_layer_input = self.x
        else:
            dropout_layer_input = dropout(numpy_rng, self.x, p=dropout_rates[0])
        self.dropout_layers = []
 
 
#        layer0=LeNetConvPoolLayer(rng=numpy_rng, input=conv_layer_input, 
#          filter_shape=filters['filter_shape1'], image_shape=Conv['image_shape1'], 
#          poolsize=self.poolsize)
#        self.params.extend(layer0.params)
#        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
#            'accugrad') for t in layer0.params])
#        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
#            'accudelta') for t in layer0.params])
#        assert hasattr(layer0, 'output')   
#        self.dropout_layers.append(layer0)
#        dropout_layer_input = T.cast(layer0.output,'float32')
#        #print dropout_layer_input
# 
#        layer1=LeNetConvPoolLayer(rng=numpy_rng,input=dropout_layer_input, filter_shape=filters['filter_shape2'], 
#          image_shape=Conv['image_shape2'], poolsize=self.poolsize)
#        self.params.extend(layer1.params)
#        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
#           'accugrad') for t in layer1.params])
#        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
#           'accudelta') for t in layer1.params])
#        assert hasattr(layer1, 'output')
#        self.dropout_layers.append(layer1)
#        dropout_layer_input = T.cast(layer1.output,'float32')
#        #print dropout_layer_input
#        
#        layer1b=LeNetConvPoolLayer(rng=numpy_rng,input=dropout_layer_input, filter_shape=filters['filter_shape3'], 
#          image_shape=Conv['image_shape3'], poolsize=self.poolsize)
#        self.params.extend(layer1b.params)
#        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
#           'accugrad') for t in layer1b.params])
#        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
#           'accudelta') for t in layer1b.params])
#        assert hasattr(layer1b, 'output')
#        self.dropout_layers.append(layer1b)
#        dropout_layer_input = T.cast(layer1b.output,'float32')
#        #print dropout_layer_input
#
#        layer1c=LeNetConvPoolLayer(rng=numpy_rng,input=dropout_layer_input, filter_shape=filters['filter_shape4'], 
#          image_shape=Conv['image_shape4'], poolsize=self.poolsize)
#        self.params.extend(layer1c.params)
#        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
#           'accugrad') for t in layer1c.params])
#        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
#           'accudelta') for t in layer1c.params])
#        assert hasattr(layer1c, 'output')
#        self.dropout_layers.append(layer1c)
#        dropout_layer_input = T.cast(layer1c.output.flatten(2),'float32')
#        #print dropout_layer_input        
#        
#        # construct fully-connected ReLU layers
#        n_in_array = image_shape=Conv['image_shape5']
        
#        layer2= ReLU(rng=numpy_rng, input=dropout_layer_input, drop_out=dropout_rates[3] ,fdrop=True, n_in=numpy.prod(n_in_array[1:]), n_out=layers_sizes[3])
#        self.params.extend(layer2.params)
#        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
#            'accugrad') for t in layer2.params])
#        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
#            'accudelta') for t in layer2.params])
#        assert hasattr(layer2, 'output')
#        self.dropout_layers.append(layer2)
#        dropout_layer_input = layer2.output
#        #print dropout_layer_input   
 
        layer3= ReLU(rng=numpy_rng, input=conv_layer_input, drop_out=dropout_rates[0] , fdrop=True, n_in=conv_reshaper[1], n_out=layers_sizes[0])    
        self.params.extend(layer3.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
            'accugrad') for t in layer3.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
            'accudelta') for t in layer3.params])
        assert hasattr(layer3, 'output')
        self.dropout_layers.append(layer3)
        dropout_layer_input = T.cast(layer3.output,'float32')
        #print dropout_layer_input
 
        layer4= ReLU(rng=numpy_rng, input=dropout_layer_input, drop_out=dropout_rates[1] , fdrop=True, n_in=layers_sizes[0], n_out=layers_sizes[1])
        self.params.extend(layer4.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
            'accugrad') for t in layer4.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
            'accudelta') for t in layer4.params])
        assert hasattr(layer4, 'output')
        self.dropout_layers.append(layer4)
        dropout_layer_input = T.cast(layer4.output,'float32')
        #print dropout_layer_input
 
        # classify the values
        layer5= LogisticRegression(rng=numpy_rng, input=dropout_layer_input, n_in=layers_sizes[1], n_out=n_outs)
        self.params.extend(layer5.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
            'accugrad') for t in layer5.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
            'accudelta') for t in layer5.params])
        assert hasattr(layer5, 'output')
        self.dropout_layers.append(layer5)
        dropout_layer_input = T.cast(layer5.output,'float32')
        print dropout_layer_input
 
 
        assert hasattr(self.dropout_layers[-1], 'training_cost')
        assert hasattr(self.dropout_layers[-1], 'errors')
        
        self.mean_cost = self.dropout_layers[-1].negative_log_likelihood(self.y)
        #print self.mean_cost.dtype
        self.cost = self.dropout_layers[-1].training_cost(self.y)
        self.prediction_1 = self.dropout_layers[-1].prediction(self.x)
        self.prediction_soft = self.dropout_layers[-1].pred_soft(self.x)
        if debugprint:
            theano.printing.debugprint(self.cost) 
        # the non-dropout errors
        self.errors = self.dropout_layers[-1].errors(self.y)
    
    def __repr__(self):
        dimensions_layers_str = map(lambda x: "x".join(map(str, x)),
                                    zip(self.layers_ins, self.layers_outs))
        return "_".join(map(lambda x: "_".join((x[0].__name__, x[1])),
                            zip(self.layers_types, dimensions_layers_str))) + "\n"\
        + "dropout rates: " + str(self.dropout_rates)
 
 
    def get_SGD_trainer(self):
        """ Returns a plain SGD minibatch trainer with learning rate as param.
        """
        batch_x = T.fmatrix('batch_x')
        #batch_x= T.matrix('batch_x', dtype=theano.config.floatX)
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        # using mean_cost so that the learning rate is not too dependent
        # on the batch size
        gparams = T.grad(self.mean_cost, self.params)
 
        # compute list of weights updates
        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            if self.max_norm:
                W = param - gparam * learning_rate
                col_norms = W.norm(2, axis=0)
                desired_norms = T.clip(col_norms, 0, self.max_norm)
                updates[param] = W * (desired_norms / (1e-6 + col_norms))
            else:
                updates[param] = param - gparam * learning_rate
 
        train_fn = theano.function(inputs=[theano.Param(batch_x),
                                           theano.Param(batch_y),
                                           theano.Param(learning_rate)],
                                   outputs=self.mean_cost,
                                   updates=updates,
                                   givens={self.x: batch_x, self.y: batch_y})
 
        return train_fn
 
    def get_adagrad_trainer(self):
        """ Returns an Adagrad (Duchi et al. 2010) trainer using a learning rate.
        """
        batch_x = T.fmatrix('batch_x')
        #batch_x= T.matrix('batch_x', dtype=theano.config.floatX)
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.mean_cost, self.params)
 
        # compute list of weights updates
        updates = OrderedDict()
        for accugrad, param, gparam in zip(self._accugrads, self.params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = accugrad + gparam * gparam
            dx = - (learning_rate / T.sqrt(agrad + self._eps)) * gparam
            if self.max_norm:
                W = param + dx
                col_norms = W.norm(2, axis=0)
                desired_norms = T.clip(col_norms, 0, self.max_norm)
                updates[param] = W * (desired_norms / (1e-6 + col_norms))
            else:
                updates[param] = param + dx
            updates[accugrad] = agrad
 
        train_fn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y),
            theano.Param(learning_rate)],
            outputs=self.mean_cost,
            updates=updates,
            givens={self.x: batch_x, self.y: batch_y})
 
        return train_fn
 
    def get_adadelta_trainer(self):
        """ Returns an Adadelta (Zeiler 2012) trainer using self._rho and
        self._eps params.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.mean_cost, self.params)
        # compute list of weights updates
        updates = OrderedDict()
        for accugrad, accudelta, param, gparam in zip(self._accugrads,
                self._accudeltas, self.params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
            agrad=T.cast(agrad, 'float32')
            dx = - T.sqrt((accudelta + self._eps)
                          / (agrad + self._eps)) * gparam
            dx=T.cast(dx, 'float32')
            updates[accudelta] = T.cast((self._rho * accudelta
                                  + (1 - self._rho) * dx * dx),'float32')
            if self.max_norm:
                W = T.cast(param + dx,'float32')
                col_norms = T.cast(W.norm(2, axis=0),'float32')
                desired_norms = T.cast(T.clip(col_norms, 0, self.max_norm),'float32')
                updates[param] = T.cast(W * (desired_norms / (1e-6 + col_norms)),'float32')
            else:
                updates[param] = T.cast(param + dx,'float32')
            updates[accugrad] = T.cast(agrad,'float32')
 
        train_fn = theano.function(inputs=[theano.Param(batch_x),
                                           theano.Param(batch_y)],
                                   outputs=self.mean_cost,
                                   updates=updates,
                                   givens={self.x: batch_x, self.y: batch_y},
                                   allow_input_downcast=True)
 
        return train_fn
 
 
    def score_classif(self, given_set):
        """ Returns functions to get current classification errors. """
        batch_x = T.fmatrix('batch_x')
        #batch_x= T.matrix('batch_x', dtype=theano.config.floatX)
        batch_y = T.ivector('batch_y')
        score = theano.function(inputs=[theano.Param(batch_x),
                                        theano.Param(batch_y)],
                                outputs=self.errors,
                                givens={self.x: batch_x, self.y: batch_y},
                                allow_input_downcast=True)
 
        def scoref():
            """ returned function that scans the entire set given as input """
            return [score(batch_x, batch_y) for batch_x, batch_y in given_set]
 
        return scoref
 
    def predict(self,X):
        """ Returns functions to get current classification errors. """
        batch_x = T.fmatrix('batch_x')
        #batch_x= T.matrix('batch_x', dtype=theano.config.floatX)
        predictor = theano.function(inputs=[theano.Param(batch_x)],
                                outputs=self.prediction_1,
                                givens={self.x: batch_x},
                                allow_input_downcast=True)
        return predictor(X)

    def predict_soft(self,X):
        """ Returns functions to get current classification errors. """
        batch_x = T.fmatrix('batch_x')
        #batch_x= T.matrix('batch_x', dtype=theano.config.floatX)
        predictor1 = theano.function(inputs=[theano.Param(batch_x)],
                                outputs=self.prediction_soft,
                                givens={self.x: batch_x},
                                allow_input_downcast=True)
        return predictor1(X) 
 
 
def add_fit_and_score_early_stop(class_to_chg):
    """ Mutates a class to add the fit() and score() functions to a NeuralNet.
    """
    from types import MethodType
    def fit(self, x_train, y_train, x_dev=None, y_dev=None,
            max_epochs=300, early_stopping=True, split_ratio=0.1, data_multiply=1, 
            method='adadelta', verbose=False, plot=False):
 
        """
        Fits the neural network to `x_train` and `y_train`. 
        If x_dev nor y_dev are not given, it will do a `split_ratio` cross-
        validation split on `x_train` and `y_train` (for early stopping).
        """
        import time, copy
        if x_dev == None or y_dev == None:
            from sklearn.cross_validation import train_test_split
            x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train,
                    test_size=split_ratio, random_state=42)
        if method == 'sgd':
            train_fn = self.get_SGD_trainer()
        elif method == 'adagrad':
            train_fn = self.get_adagrad_trainer()
        elif method == 'adadelta':
            train_fn = self.get_adadelta_trainer()
        train_set_iterator = DatasetMiniBatchIterator(x_train, y_train, randomize=False)
#        train_set_iterator = DatasetMiniBatchIteratorEven(x_train, y_train)
        dev_set_iterator = DatasetMiniBatchIterator(x_dev, y_dev)
        train_scoref = self.score_classif(train_set_iterator)
        dev_scoref = self.score_classif(dev_set_iterator)
        best_dev_loss = np.inf
        epoch = 0
 
        patience = 10000  # look as this many examples regardless 
        patience_increase = 2.  # wait this much longer when a new best is
                                    # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                           # considered significant
 
        done_looping = False
        print '... training the model'
        # early-stopping parameters
        test_score = 0.
        start_time = time.clock()
 
        done_looping = False
        epoch = 0
        timer = None
 
        if plot:
            verbose = True
            self._costs = []
            self._train_errors = []
            self._dev_errors = []
            self._updates = []
     
        while (epoch < max_epochs) and (not done_looping):
            epoch += 1
            if not verbose:
                sys.stdout.write("\r%0.2f%%" % (epoch * 100./ max_epochs))
                sys.stdout.flush()
            avg_costs = []
            timer = time.time()
            for iteration, (x, y) in enumerate(train_set_iterator):
#                if y.shape[0] != BATCH_SIZE:
#                    print y.shape[0]
#                    g1 = np.arange(y_train.shape[0])
#                    np.random.shuffle(g1)
#                    for n in g1[:(BATCH_SIZE-y.shape[0])]:
#                        x = np.vstack((x,x_train[n,:]))
#                        y = np.hstack((y,y_train[n]))
                if method == 'sgd' or method == 'adagrad':
                    avg_cost = train_fn(x, y, lr=.04)  # LR is very dataset dependent
                elif method == 'adadelta':
                    avg_cost = train_fn(x, y)
                if type(avg_cost) == list:
                    avg_costs.append(avg_cost[0])
                else:
                    avg_costs.append(avg_cost)
                if patience <= iteration:  #i think i fixed this part
                    done_looping = True
                    break
            if verbose:
                mean_costs = np.mean(avg_costs)
                mean_train_errors = np.mean(train_scoref())
                dt = datetime.datetime.now()
                print(dt.strftime("%A, %d. %B %Y %I:%M%p"))
                print('  epoch %i took %f seconds' %
                      (epoch, time.time() - timer))
                print('  epoch %i, avg costs %f' %
                      (epoch, mean_costs))
                print('  epoch %i, training error %f' %
                      (epoch, mean_train_errors))
                
                sys.stdout.flush()
                if plot:
                    self._costs.append(mean_costs)
                    self._train_errors.append(mean_train_errors)
            dev_errors = np.mean(dev_scoref())
            if plot:
                self._dev_errors.append(dev_errors)
            if dev_errors < best_dev_loss:
                best_dev_loss = dev_errors
                best_params = copy.deepcopy(self.params)
                if verbose:
                    print('!!!  epoch %i, validation error of best model %f' %
                          (epoch, dev_errors))
                    sys.stdout.flush()      
                if (dev_errors < best_dev_loss *
                improvement_threshold):
                    patience = max(patience, iteration * patience_increase)
            if epoch % 10 == 0:
                tfile = open('dnntemp.pkl','wb')
                cPickle.dump((self, best_params),tfile)
                tfile.close()
        if not verbose:
            print("")
        for i, param in enumerate(best_params):
            self.params[i] = param
     
    def score(self, x, y):
        """ error rates """
        iterator = DatasetMiniBatchIterator(x, y)
        scoref = self.score_classif(iterator)
        return np.mean(scoref())
 
     
    class_to_chg.fit = MethodType(fit, None, class_to_chg)
    class_to_chg.score = MethodType(score, None, class_to_chg)
 
 
if __name__ == "__main__":
    from sklearn import cross_validation, preprocessing
    if 0:
        savefile = open('traindata.pkl', 'rb')
        (x_train, y_train, t1) = cPickle.load(savefile)
        savefile.close()
        
        x_train = np.asarray(x_train,dtype=np.float32)
        y_train = np.asarray(y_train, dtype='int32')-1
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(
                          x_train, y_train, test_size=0.1, random_state=42)

        add_fit_and_score_early_stop(ConvDropNet02)

        
        dnn = ConvDropNet02(numpy_rng=np.random.RandomState(123),
            theano_rng=None,   
            n_ins=x_train.shape[1],
            conv_reshaper=(BATCH_SIZE, x_train.shape[1]),
            batch_size=BATCH_SIZE,
    #        Conv = {'image_shape1':(BATCH_SIZE, 1, squsize, squsize),'image_shape2':(BATCH_SIZE, 64, 14, 14),'image_shape3':(BATCH_SIZE, 128, 6, 6)},        
#            Conv = {'image_shape1':(BATCH_SIZE, 3, squsize, squsize),'image_shape2':(BATCH_SIZE, firststage, conv1size, conv1size),'image_shape3':(BATCH_SIZE, secondstage, conv2size, conv2size),'image_shape4':(BATCH_SIZE, thirdstage, conv3size, conv3size),'image_shape5':(BATCH_SIZE, forthstage, conv4size, conv4size)},
#            filters={'filter_shape1':(firststage, 3, conv1, conv1),'filter_shape2':(secondstage, firststage, conv2, conv2),'filter_shape3':(thirdstage, secondstage, conv3, conv3),'filter_shape4':(forthstage, thirdstage, conv4, conv4)},
            poolsize=(2,2),
            layers_types=[ReLU, ReLU, LogisticRegression],
            layers_sizes=[2048*2, 2048],
            n_outs=len(set(y_train)),
            rho=0.98,
            eps=1.E-6,
            max_norm=4,
            fast_drop=True, 
            dropout_rates=[0.5, 0.5, 0.], #match this up with actual layers, last 
            debugprint=False)
            
        #train the model here, plot=False ,adadelta=fast and no learning rate need be provided
        dnn.fit(x_train, y_train, max_epochs=60, method='adadelta', verbose=True, plot=False) 
        test_error = dnn.score(x_test, y_test)
        print("score: %f" % (1. - test_error))
        print dnn.__repr__
        
        tfile = open('dnntemp.pkl','wb')
        cPickle.dump((dnn, dnn.params),tfile)
        tfile.close()
    elif 0:
        print 'confustion matrix'
        savefile = open('traindata.pkl', 'rb')
        (x_train, y_train, t1) = cPickle.load(savefile)
        savefile.close()
        x_train = np.asarray(x_train,dtype=np.float32)
        y_train = np.asarray(y_train, dtype='int32')-1
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(
                          x_train, y_train, test_size=0.1, random_state=42)
#        add_fit_and_score_early_stop(ConvDropNet02)  
        
        savefile = open('dnn_002.pkl','rb')
        (dnn, d1) = cPickle.load(savefile)
        savefile.close()

#        savefile = open('traindata_48_dataset_04.pkl', 'rb')
#        (t1, y1, class1 ,t2) = cPickle.load(savefile)
#        savefile.close()
#        classdict = dict(zip(y1, class1))
        namesClasses = [str(x) for x in np.arange(9)]
        
        bsize = BATCH_SIZE
        y_pred = np.zeros((x_test.shape[0]),dtype=np.float32)
        for n in np.arange(x_test.shape[0]/bsize):
            y_pred[n*bsize:(n+1)*bsize] = dnn.predict(x_test[n*bsize:(n+1)*bsize,:])        
        
        n=n+1
        lastx = x_test[n*bsize:(n+1)*bsize,:]
        remrows = bsize-lastx.shape[0]
        lastx = np.vstack((lastx,np.zeros((remrows,lastx.shape[1]), dtype=np.float32)))
        lasty = dnn.predict(lastx) 
        y_pred[n*bsize:(n+1)*bsize] = lasty[:(bsize-remrows)]
        
        
#        y_pred = dnn.predict(x_test)
        
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        class_report = classification_report(y_test, y_pred, target_names=namesClasses)
        print class_report
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
    else:
        print 'write submission'
        savefile = open('testdata.pkl', 'rb')
        (x_test, t1, name1) = cPickle.load(savefile)
        savefile.close()
 
        savefile = open('dnn_002.pkl','rb')
        (dnn, d1) = cPickle.load(savefile)
        savefile.close()
        
#        y1 = np.arange(9)
#        class1 = []
#        
#        classdict = dict(zip(y1, class1))
#        y_str = [classdict[x] for x in np.arange(121)]
        
        bsize = BATCH_SIZE
        ypred = np.zeros((x_test.shape[0],9),dtype=np.float32)
        for n in np.arange(x_test.shape[0]/bsize):
            ypred[n*bsize:(n+1)*bsize,:] = dnn.predict_soft(x_test[n*bsize:(n+1)*bsize,:])        
        
        n=n+1
        lastx = x_test[n*bsize:(n+1)*bsize,:]
        remrows = bsize-lastx.shape[0]
        lastx = np.vstack((lastx,np.zeros((remrows,lastx.shape[1]), dtype=np.float32)))
        lasty = dnn.predict_soft(lastx) 
        ypred[n*bsize:(n+1)*bsize,:] = lasty[:(bsize-remrows),:]
#       print ypred[0:10]
        print ypred.shape
        y_str = ['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']
        kcsv.print_csv(ypred, name1, y_str,indexname='id')
        print 'done'