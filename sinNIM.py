"""sinNIM"""

from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from FFnetwork.ffnetwork import FFNetwork
from FFnetwork.network import Network
from FFnetwork.layer import Layer
from FFnetwork.regularization import Regularization
from .networkNIM import NetworkNIM

class SInetNIM( NetworkNIM ):
    """Implementation of SInetNIM

    Attributes:
        network (siFFNetwork object): feedforward network
        input_size (int): dimensions of stimulus (assuming already an Xmatrix)
        noise_dist (str): noise distribution used for cost function

        activations (list of tf ops): evaluates the layer-wise activations of
            the model
        cost (tf op): evaluates the cost function of the model
        cost_reg (tf op): evaluates the regularization penalty of the model
        cost_penalized (tf op): evaluates sum of `cost` and `cost_reg`
        train_step (tf op): evaluates one training step using the specified
            cost function and learning algorithm

        data_in_ph (tf.placeholder): placeholder for input data
        data_out_ph (tf.placeholder): placeholder for output data
        data_in_var (tf.Variable): Variable for input data fed by data_in_ph
        data_out_var (tf.Variable): Variable for output data fed by data_out_ph
        data_in_batch (tf.Variable): batched version of data_in_var
        data_out_batch (tf.Variable): batched version of data_out_var
        indices (tf.placeholder): placeholder for indices into data_in/out_var
            to create data_in/out_batch

        learning_alg (str): algorithm used for learning parameters
        learning_rate (float): learning rate used by the gradient
            descent-based optimizers
        num_examples (int): total number of examples in training data
        use_batches (bool): all training data is placed on the GPU by
            piping data into i/o variables through a feed_dict. If use_batches
            is False, all data from these variables are used during training.
            If use_batches is True, the a separate index feed_dict is used to
            specify which chunks of data from these variables acts as input to
            the model.

        graph (tf.Graph): dataflow graph for the network
        use_gpu (bool): input into sess_config
        sess_config (tf.ConfigProto): specifies device for model training
        saver (tf.train.Saver): for saving and restoring variables
        merge_summaries (tf op): op that merges all summary ops
        init (tf op): op that initializes global variables in graph

    """

    def __init__(
            self,
            stim_dims,
            num_examples,
            num_neurons=1,
            num_subunits=None,  # default is LN model with no hidden layers
            first_filter_width=None,
            shift_spacing=1,
            binocular=False,
            act_funcs='relu',
            ei_layers=None,
            noise_dist='poisson',
            reg_list=None,
            init_type='trunc_normal',
            learning_alg='lbfgs',
            learning_rate=1e-3,
            reg_initializer=None,
            use_batches=False,
            tf_seed=0,
            use_gpu=None):
        """Constructor for SInetNIM class: identical to NetworkNIM class other than
            building a slightly different graph (otherwise transparent_

        Args:
            stim_dims: int representing number of stimulus dimensions
                (must match X-matrix)
            num_examples (int): total number of examples in training data
            num_neurons: number of neurons to fit simultaneously
            num_subunits: (list of ints, or empty list), listing number of
                subunits at each stage for LN models, leave blank [], and NIM
                have number of subunits as such: [3]
            [Note: 'layer_sizes' for FFNetwork calculated from the above
                quantities]
            ei_layers: intermediate layers can be constrained to be composed of
                E and I neurons. This should be a list of the length of the
                num_subunits, with the number of Inh neurons defined for each
                layer. This will make that subunit layer have a certain number
                of inhibitory outputs, and the next layer to be constrained to
                have positive weights. Use '-1's for layers where there should
                be no constraints
            act_funcs (str or list of strs, optional): activation function for
                network layers; replicated if a single element.
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' |
                'elu' | 'quad'
            noise_dist (str, optional): noise distribution used by network
                ['gaussian'] | 'poisson' | 'bernoulli'
            init_type (str or list of strs, optional): initialization
                used for network weights; replicated if a single element.
                ['trunc_normal'] | 'normal' | 'zeros' | 'xavier'
            learning_alg (str, optional): algorithm used for learning
                parameters.
                ['lbfgs'] | 'adam'
            learning_rate (float, optional): learning rate used by the
                gradient descent-based optimizers ('adam'). Default is 1e-3.
            reg_initializer (dict):
            use_batches (boolean, optional): determines how data is fed to
                model; if False, all data is pinned to variables used
                throughout fitting; if True, a data slicer and batcher are
                constructed to feed the model shuffled batches throughout
                training
            tf_seed (scalar): rng seed for tensorflow to facilitate building
                reproducible models
            use_gpu (bool): use gpu for training model

        Raises:
            TypeError: If layer_sizes argument is not specified
            TypeError: If num_examples argument is not specified
            ValueError: If noise_dist argument is not valid string
            ValueError: If learning_alg is not a valid string

        Thinking about Ninhibitory units and positivity constraints:
        -- The last (output layer) is never negative, so should only correspond
           to subunits
        -- Positivity constraints never applied to weights on input

        """

        # call __init__() method of super-super class
        super(NetworkNIM, self).__init__()

        # input checking
        if num_subunits is None:
            # This will be LN models (default)
            layer_sizes = [stim_dims] + [num_neurons]
            ei_layers = []
        else:
            if not isinstance(num_subunits,list):
                num_subunits = [num_subunits]
            layer_sizes = [stim_dims] + num_subunits + [num_neurons]
            if ei_layers is None:
                ei_layers = [-1]*len(num_subunits)
            assert len(num_subunits) == len(ei_layers), \
                'ei_layers must have the same length as num_subunits.'

        # Establish positivity constraints
        num_layers = len(layer_sizes)-1
        pos_constraint = [False]*(num_layers)
        num_inh_layers = [0]*(num_layers)
        for nn in range(len(ei_layers)):
            if ei_layers[nn] >= 0:
                pos_constraint[nn+1] = True
                num_inh_layers[nn] = ei_layers[nn]

        # input checking
        if num_examples is None:
            raise ValueError('Must specify number of training examples')

        # set model attributes from input
        self.input_size = stim_dims
        self.output_size = num_neurons
        self.shift_spacing = shift_spacing  # particular to SInet
        self.binocular = binocular # particular to SInet
        self.activation_functions = act_funcs
        self.noise_dist = noise_dist
        self.learning_alg = learning_alg
        self.learning_rate = learning_rate
        self.num_examples = num_examples
        self.use_batches = use_batches

        # for saving and restoring models
        self.graph = tf.Graph()  # must be initialized before graph creation

        # for specifying device
        if use_gpu is not None:
            self.use_gpu = use_gpu
            if use_gpu:
                self.sess_config = tf.ConfigProto(device_count={'GPU': 1})
            else:
                self.sess_config = tf.ConfigProto(device_count={'GPU': 0})

        # build model graph
        with self.graph.as_default():

            np.random.seed(tf_seed)
            tf.set_random_seed(tf_seed)

            # define pipeline for feeding data into model
            with tf.variable_scope('data'):
                self._initialize_data_pipeline()

            # initialize weights and create model
            self.network = siFFNetwork(
                scope='model',
                inputs=self.data_in_batch,
                first_filter_width=first_filter_width,
                shift_spacing=shift_spacing,
                binocular=binocular,
                layer_sizes=layer_sizes,
                num_inh=num_inh_layers,
                activation_funcs=act_funcs,
                pos_constraint=pos_constraint,
                weights_initializer=init_type,
                biases_initializer='zeros',
                reg_initializer=reg_initializer,
                log_activations=True)

            # Initialize regularization, if there is any
            if reg_list is not None:
                for reg_type, reg_val_list in reg_list.iteritems():
                    if not isinstance(reg_val_list,list):
                        reg_val_list = [reg_val_list]*num_layers
                    assert len(reg_val_list) == num_layers, \
                        'Need to match number of layers with regularization values.'
                    for nn in range(num_layers):
                        if reg_val_list[nn] is not None:
                            self.network.layers[nn].reg.vals[reg_type] = reg_val_list[nn]

            # define loss function
            with tf.variable_scope('loss'):
                self._define_loss()

            # define optimization routine
            with tf.variable_scope('optimizer'):
                self._define_optimizer()

            # add additional ops
            # for saving and restoring models (initialized after var creation)
            self.saver = tf.train.Saver()
            # collect all summaries into a single op
            self.merge_summaries = tf.summary.merge_all()
            # add variable initialization op to graph
            self.init = tf.global_variables_initializer()
    # END SInetNIM.__init__


    def copy_model( self, target=None,
                   layers_to_transfer=None,
                   target_layers=None,
                   init_type='trunc_normal',tf_seed=0):

        num_layers = len(self.network.layers)
        if target is None:
            # Re-derive subunit layers and ei-masks
            num_subunits = [0]*(num_layers-1)
            ei_layers = [-1]*(num_layers-1)
            for nn in range(len(num_subunits)):
                num_subunits[nn] = self.network.layers[nn].num_outputs
                if self.network.layers[nn+1].pos_constraint:
                    ei_layers[nn] = 0  # will copy ei_mask later
            # Modify convolutional layer subunits
            num_subunits[0] = int(num_subunits[0] / self.input_size * self.shift_spacing)

            # Accumulate regularization list from layers
            reg_list = {}
            for reg_type, reg_vals in self.network.layers[0].reg.vals.iteritems():
                reg_list[reg_type] = [None]*num_layers
            for nn in range(num_layers):
                for reg_type, reg_vals in self.network.layers[nn].reg.vals.iteritems():
                    reg_list[reg_type][nn] = self.network.layers[nn].reg.vals[reg_type]

            # Make new target
            target = SInetNIM( self.input_size, self.num_examples,
                               num_neurons=self.output_size,
                               num_subunits = num_subunits,
                               first_filter_width=self.network.layers[0].weights.shape[0],
                               binocular=self.binocular,
                               shift_spacing=self.shift_spacing,
                               act_funcs = self.activation_functions,
                               ei_layers=ei_layers,
                               noise_dist=self.noise_dist,
                               reg_list=reg_list,
                               init_type=init_type,
                               learning_alg=self.learning_alg,
                               learning_rate=self.learning_rate,
                               use_batches = self.use_batches,
                               tf_seed=tf_seed,
                               use_gpu=self.use_gpu)

            # the rest of the properties will be copied directly, which includes ei_layer stuff, act_funcs

        # Figure out mapping from self.layers to target.layers
        num_layers_target = len(target.network.layers)
        if layers_to_transfer is not None:
            assert max(layers_to_transfer) <= num_layers, 'Too many layers to transfer.'
            if target_layers is None:
                assert len(layers_to_transfer) <= num_layers_target, 'Too many layers to transfer.'
                target_layers = range(length(layers_to_transfer))
        if target_layers is not None:
            assert max(target_layers) <= num_layers_target, 'Too many layers for target.'
            if layers_to_transfer is None:
                assert len(target_layers) <= num_layers, 'Too many target layers.'
                layers_to_transfer = range(len(target_layers))
        if num_layers >= num_layers_target:
            target_copy = range(num_layers_target)
            if target_layers is None:
                self_copy = target_copy  # then default is copy the top_most layers
            else:
                self_copy = target_layers
        else:
            self_copy = range(num_layers)
            if target_layers is None:
                target_copy = self_copy # then default is copy the top_most layers
            else:
                target_copy = target_layers
        assert len(target_copy) == len(self_copy), 'Number of targets and transfers must match.'

        # Copy information from self to new target NIM
        for nn in range(len(self_copy)):
            self_layer = self.network.layers[self_copy[nn]]
            tar = target_copy[nn]

            # Copy remaining layer properties
            target.network.layers[tar].ei_mask = self_layer.ei_mask

            if self_layer.num_outputs <= target.network.layers[tar].num_outputs:
                target.network.layers[tar].weights[:,0:self_layer.num_outputs] \
                    = self_layer.weights
                target.network.layers[tar].biases[0:self_layer.num_outputs] \
                    = self_layer.biases
            else:
                target.network.layers[tar].weights = \
                    self_layer.weights[:,0:target.network.layers[tar].num_outputs]
                target.network.layers[tar].biases = \
                    self_layer.biases[0:target.network.layers[tar].num_outputs]

        return target
    # END SInetNIM.make_copy

class siFFNetwork( FFNetwork ):
    """Implementation of siFFNetwork"""

    def __init__(
            self,
            scope=None,
            inputs=None,
            first_filter_width=None,
            shift_spacing=1,
            binocular=False,
            layer_sizes=None,
            activation_funcs='relu',
            weights_initializer='trunc_normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=False,
            log_activations=False):
        """Constructor for Network class

        Args:
            scope (str): name scope for network
            inputs (tf Tensor or placeholder): input to network
            layer_sizes (list of ints): list of layer sizes, including input
                and output
            activation_funcs (str or list of strs, optional): pointwise
                function for each layer; replicated if a single element.
                See Layer class for options.
            weights_initializer (str or list of strs, optional): initializer
                for the weights in each layer; replicated if a single element.
                See Layer class for options.
            biases_initializer (str or list of strs, optional): initializer for
                the biases in each layer; replicated if a single element.
                See Layer class for options.
            reg_initializer (dict): reg_type/vals as key-value pairs to
                uniformly initialize layer regularization
            num_inh (None, int or list of ints, optional)
            pos_constraint (bool or list of bools, optional):
            log_activations (bool, optional): True to use tf.summary on layer
                activations

        Raises:
            TypeError: If `scope` is not specified
            TypeError: If `inputs` is not specified
            TypeError: If `layer_sizes` is not specified
            ValueError: If `activation_funcs` is not a properly-sized list
            ValueError: If `weights_initializer` is not a properly-sized list
            ValueError: If `biases_initializer` is not a properly-sized list

        """

        # check for required inputs
        if scope is None:
            raise TypeError('Must specify network scope')
        if inputs is None:
            raise TypeError('Must specify network input')
        if layer_sizes is None:
            raise TypeError('Must specify layer sizes')

        self.scope = scope
        self.num_layers = len(layer_sizes) - 1

        # expand layer options
        if type(activation_funcs) is not list:
            activation_funcs = [activation_funcs] * self.num_layers
        elif len(activation_funcs) != self.num_layers:
            raise ValueError('Invalid number of activation_funcs')

        if type(weights_initializer) is not list:
            weights_initializer = [weights_initializer] * self.num_layers
        elif len(weights_initializer) != self.num_layers:
            raise ValueError('Invalid number of weights_initializer')

        if type(biases_initializer) is not list:
            biases_initializer = [biases_initializer] * self.num_layers
        elif len(biases_initializer) != self.num_layers:
            raise ValueError('Invalid number of biases_initializer')

        if type(num_inh) is not list:
            num_inh = [num_inh] * self.num_layers
        elif len(num_inh) != self.num_layers:
            raise ValueError('Invalid number of num_inh')

        if type(pos_constraint) is not list:
            pos_constraint = [pos_constraint] * self.num_layers
        elif len(pos_constraint) != self.num_layers:
            raise ValueError('Invalid number of pos_con')

        self.layers = []
        with tf.name_scope(self.scope):
            self.layers.append(siLayer(
                scope='conv_layer',
                inputs=inputs,
                filter_width=first_filter_width,
                shift_spacing=shift_spacing,
                binocular=binocular,
                num_inputs=layer_sizes[0],
                num_filters=layer_sizes[1],
                activation_func=activation_funcs[0],
                weights_initializer=weights_initializer[0],
                biases_initializer=biases_initializer[0],
                reg_initializer=reg_initializer,
                num_inh=num_inh[0],
                pos_constraint=pos_constraint[0],
                log_activations=log_activations))
            inputs = self.layers[0].outputs

            # num_inputs to next layer is adjusted by number of shifts, so recalculate
            layer_sizes[1]=self.layers[0].num_outputs
            # Attach rest of layers
            for layer in range(1,self.num_layers):
                self.layers.append(Layer(
                    scope='layer_%i' % layer,
                    inputs=inputs,
                    num_inputs=layer_sizes[layer],
                    num_outputs=layer_sizes[layer + 1],
                    activation_func=activation_funcs[layer],
                    weights_initializer=weights_initializer[layer],
                    biases_initializer=biases_initializer[layer],
                    reg_initializer=reg_initializer,
                    num_inh=num_inh[layer],
                    pos_constraint=pos_constraint[layer],
                    log_activations=log_activations))
                inputs = self.layers[-1].outputs

        if log_activations:
            self.log = True
        else:
            self.log = False
    # END siFFnetwork.__init__

class siLayer(Layer):
    """Implementation of fully connected neural network layer

    Attributes:
        scope (str): name scope for variables and operations in layer
        num_inputs (int): number of inputs to layer
        num_outputs (int): number of outputs of layer
        outputs (tf Tensor): output of layer
        num_inh (int): number of inhibitory units in layer
        weights_ph (tf placeholder): placeholder for weights in layer
        biases_ph (tf placeholder): placeholder for biases in layer
        weights_var (tf Tensor): weights in layer
        biases_var (tf Tensor): biases in layer
        weights (numpy array): shadow variable of `weights_var` that allows for 
            easier manipulation outside of tf sessions
        biases (numpy array): shadow variable of `biases_var` that allows for 
            easier manipulation outside of tf sessions
        activation_func (tf activation function): activation function in layer
        reg (Regularization object): holds regularizations values and matrices
            (as tf constants) for layer
        ei_mask_var (tf constant): mask of +/-1s to multiply output of layer
        ei_mask (list): mask of +/-1s to multiply output of layer; shadows 
            `ei_mask_tf` for easier manipulation outside of tf sessions
        pos_constraint (bool): positivity constraint on weights in layer
        log (bool): use tf summary writers on layer output

    """

    def __init__(
            self,
            scope=None,
            inputs=None,
            num_inputs=None,
            num_filters=None,
            filter_width=None,
            shift_spacing=1,
            binocular=False,
            activation_func='relu',
            weights_initializer='trunc_normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=False,
            log_activations=False):
        """Constructor for Layer class

        Args:
            scope (str): name scope for variables and operations in layer
            inputs (tf Tensor or placeholder): input to layer
            num_inputs (int): dimension of input data
            num_outputs (int): dimension of output data
            activation_func (str, optional): pointwise function applied to  
                output of affine transformation
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' | 'elu' | 'quad'
            weights_initializer (str, optional): initializer for the weights
                ['trunc_normal'] | 'normal' | 'zeros'
            biases_initializer (str, optional): initializer for the biases
                'trunc_normal' | 'normal' | ['zeros']
            reg_initializer (dict, optional): see Regularizer docs for info
            num_inh (int, optional): number of inhibitory units in layer
            pos_constraint (bool, optional): True to constrain layer weights to be 
                positive
            log_activations (bool, optional): True to use tf.summary on layer 
                activations

        Raises:
            TypeError: If `variable_scope` is not specified
            TypeError: If `inputs` is not specified
            TypeError: If `num_inputs` or `num_outputs` is not specified
            ValueError: If `num_inh` is greater than total number of units
            ValueError: If `activation_func` is not a valid string
            ValueError: If `weights_initializer` is not a valid string
            ValueError: If `biases_initializer` is not a valid string

        """

        # check for required inputs
        if scope is None:
            raise TypeError('Must specify layer scope')
        if inputs is None:
            raise TypeError('Must specify layer input')
        if num_inputs is None or num_filters is None:
            raise TypeError('Must specify both input dimensions and number of filters.')

        self.scope = scope

        self.num_inputs = num_inputs
        self.num_outputs = num_filters
        if filter_width is None:
            if binocular is False:
                filter_width = num_inputs
            else:
                filter_width = num_inputs/2

        # resolve activation function string
        if activation_func == 'relu':
            self.activation_func = tf.nn.relu
        elif activation_func == 'sigmoid':
            self.activation_func = tf.sigmoid
        elif activation_func == 'tanh':
            self.activation_func = tf.tanh
        elif activation_func == 'linear':
            self.activation_func = tf.identity
        elif activation_func == 'softplus':
            self.activation_func = tf.nn.softplus
        elif activation_func == 'quad':
            self.activation_func = tf.square
        elif activation_func == 'elu':
            self.activation_func = tf.nn.elu
        else:
            raise ValueError('Invalid activation function ''%s''' %
                             activation_func)

        # create excitatory/inhibitory mask
        if num_inh > num_filters:
            raise ValueError('Too many inhibitory units designated')
        self.ei_mask = [1] * (num_filters - num_inh) + [-1] * num_inh
        if num_inh > 0:
            self.ei_mask_var = tf.constant(
                self.ei_mask, dtype=tf.float32, name='ei_mask')
        else:
            self.ei_mask_var = None

        # save positivity constraint on weights
        self.pos_constraint = pos_constraint
        assert pos_constraint is False, 'No positive constraint should be applied to this layer.'

        # use tf's summary writer to save layer activation histograms
        if log_activations:
            self.log = True
        else:
            self.log = False

        # build layer
        with tf.name_scope(self.scope):

            # weights initialzed so num_filters is in nchannels argument of conv1d
            if binocular is False:
                weight_dims = (filter_width, 1, num_filters)
            else:
                weight_dims = (filter_width, 1, num_filters)

            if weights_initializer == 'trunc_normal':
                init_weights = np.random.normal(size=weight_dims, scale=0.1)
            elif weights_initializer == 'normal':
                init_weights = np.random.normal(size=weight_dims, scale=0.1)
            elif weights_initializer == 'zeros':
                init_weights = np.zeros(shape=weight_dims, dtype='float32')
            else:
                raise ValueError('Invalid weights_initializer ''%s''' %
                                 weights_initializer)
            # initialize numpy array that will feed placeholder
            if pos_constraint is True:
                init_weights = np.maximum(init_weights,0)

            self.weights = init_weights.astype('float32')
            # initialize weights placeholder/variable
            with tf.name_scope('weights_init'):
                self.weights_ph = tf.placeholder_with_default(
                    self.weights,
                    shape=[num_inputs, 1, num_filters],
                    name='weights_ph')
            self.weights_var = tf.Variable(
                self.weights_ph,
                dtype=tf.float32,
                name='weights_var')

            self.num_outputs = num_filters * int(np.floor(num_inputs/shift_spacing))

            # resolve biases initializer string
            bias_dims = (1, num_filters)
            if biases_initializer == 'trunc_normal':
                init_biases = np.random.normal(size=bias_dims, scale=0.1)
            elif biases_initializer == 'normal':
                init_biases = np.random.normal(size=bias_dims, scale=0.1)
            elif biases_initializer == 'zeros':
                init_biases = np.zeros(shape=bias_dims, dtype='float32')
            else:
                raise ValueError('Invalid biases_initializer ''%s''' %
                                 biases_initializer)
            # initialize numpy array that will feed placeholder
            self.biases = init_biases.astype('float32')
            # initialize biases placeholder/variable
            with tf.name_scope('biases_init'):
                self.biases_ph = tf.placeholder_with_default(
                    self.biases,
                    shape=[1, num_filters],
                    name='biases_ph')
            self.biases_var = tf.Variable(
                self.biases_ph,
                dtype=tf.float32,
                name='biases_var')

            # save layer regularization info
            self.reg = Regularization(num_inputs=num_inputs,
                                      num_outputs=num_filters,vals=reg_initializer)

            # push data through layer
            #pre = tf.add(tf.matmul(inputs, self.weights_var), self.biases_var)
            if binocular is False:
                pre = tf.nn.conv1d( tf.expand_dims(inputs,-1), self.weights_var, shift_spacing, padding='SAME' )
            else:
                stimL = tf.expand_dims( tf.slice( inputs2d, [0,0], size[-1,num_inputs/2] ), -1 )
                stimR = tf.expand_dims( tf.slice( inputs2d, [0,num_inputs/2], size[-1,num_inputs/2] ), -1 )
                pre = tf.concat(
                    tf.nn.conv1d(stimL, self.weights_var, shift_spacing, padding='SAME' ),
                    tf.nn.conv1d(stimR, self.weights_var, shift_spacing, padding='SAME' ) )

            #self.num_outputs = pre.get_shape()[0].value
            #print('Conv outputs = ''%d''' % self.num_outputs)
            #print(self.num_outputs)

            if self.ei_mask_var is not None:
                post = tf.multiply( tf.add(self.activation_func(pre),self.biases_var),
                                   self.ei_mask_var)
            else:
                post = tf.add( self.activation_func(pre), self.biases_var )

            self.outputs = tf.reshape( post, [-1,self.num_outputs] )

            if self.log:
                tf.summary.histogram('act_pre', pre)
                tf.summary.histogram('act_post', post)
    # END __init__

    def assign_layer_params(self, sess):
        """Read weights/biases in numpy arrays into tf Variables"""
        sess.run(
            [self.weights_var.initializer, self.biases_var.initializer],
            feed_dict={self.weights_ph: self.weights,
                       self.biases_ph: self.biases})

    def write_layer_params(self, sess):
        """Write weights/biases in tf Variables to numpy arrays"""
        self.weights = sess.run(self.weights_var)
        if self.pos_constraint is True:
            self.weights = np.maximum(self.weights, 0)
        self.biases = sess.run(self.biases_var)

    def define_regularization_loss(self):
        """Wrapper function for building regularization portion of graph"""
        with tf.name_scope(self.scope):
            return self.reg.define_reg_loss(self.weights_var)

    def set_regularization(self, reg_type, reg_val):
        """Wrapper function for setting regularization"""
        return self.reg.set_reg_val(reg_type, reg_val)

    def assign_reg_vals(self, sess):
        """Wrapper function for assigning regularization values"""
        self.reg.assign_reg_vals(sess)

    def get_reg_pen(self, sess):
        """Wrapper function for returning regularization penalty struct"""
        return self.reg.get_reg_penalty(sess)
