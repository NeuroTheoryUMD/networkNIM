"""sinNIM"""

from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from FFnetwork.ffnetwork import FFNetwork
#from FFnetwork.network import Network
from FFnetwork.layer import Layer
#from FFnetwork.regularization import Regularization
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

    def __init__( self,
                  network_params,
                  num_examples,
                  noise_dist='poisson',
                  learning_alg='lbfgs',
                  learning_rate=1e-3,
                  use_batches=False,
                  tf_seed=0,
                  use_gpu=None ):

        """Constructor for SInetNIM class: identical to NetworkNIM class other than
            building a slightly different graph (otherwise transparent_

        Args:
            network_params: created using 'createNIMparams', which has at least the following
                arguments that get passed to FFnetwork (and possibly more):
                -> layer_sizes (list of integers, or 3-d lists of integers)
                -> activation_funcs (list of strings)
                -> pos_constraints (list of Booleans)
                -> reg_list (dictionary for regularization-class)
                -> num_conv_layers
                -> max_layer
                -> shift_spacing
                -> binocular
            num_examples (int): total number of examples in training data
            noise_dist (str, optional): noise distribution used by network
                ['gaussian'] | 'poisson' | 'bernoulli'
            init_type (str or list of strs, optional): initialization
                used for network weights; replicated if a single element.
                ['trunc_normal'] | 'normal' | 'zeros' | 'xavier'
            learning_alg (str, optional): algorithm used for learning parameters.
                ['lbfgs'] | 'adam'
            learning_rate (float, optional): learning rate used by the
                gradient descent-based optimizers ('adam'). Default is 1e-3.
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

        assert len(network_params['layer_sizes']) > 2, 'Must have at least one hidden layer for a sinNIM.'

        # Make stim_dimensions with default of single dimension correspond to space
        stim_dims = network_params['layer_sizes'][0]
        if isinstance(stim_dims, list):
            while len(stim_dims)<3:
                stim_dims.append(1)
        else:
            stim_dims = [1,stim_dims,1]

        # Work out filter size (from filter_widths and num lags)
        if network_params['first_filter_width'] is None:
            first_filter_size = stim_dims
        else:
            first_filter_size = [stim_dims[0],network_params['first_filter_width'],1]  # assumes 1-d filter here
            if stim_dims[2] > 1:
                first_filter_size[2] = network_params['first_filter_width']  # then 2-d filter
        network_params['first_filter_size'] = first_filter_size

        # call __init__() method of super-super class
        super(SInetNIM, self).__init__(
            network_params, num_examples,
            noise_dist = noise_dist,
            learning_alg= learning_alg,
            learning_rate = learning_rate,
            use_batches = use_batches,
            tf_seed = tf_seed,
            use_gpu = use_gpu )

        # Add additional SIN-specific parameters
        self.num_conv_layers = network_params['num_conv_layers']
        self.max_layer = network_params['max_layer']
        self.binocular = network_params['binocular']

    # END SInetNIM.__init__

    def _define_network(self, network_params ):

        self.network = siFFNetwork( scope = 'siFFnetwork',
                                    inputs = self.data_in_batch,
                                    params_dict = network_params )
    # END SInetNIM._define_network

    def copy_model(self, other_network_params=None, target=None,
                   layers_to_transfer=None,
                   target_layers=None,
                   init_type='trunc_normal', tf_seed=0):

        # Figure out first-filter size
        filter_width = self.network.layers[0].weights.shape[0]/self.stim_dims[0]

        # Figure out shift_spacing
        num_shifts = self.network.layers[0].num_outputs / self.network.layers[0].weights.shape[1]
        Nspace = self.stim_dims[1]*self.stim_dims[2]
        shift_spacing = Nspace/num_shifts

        # Adjust if doing 2-dimensions in space
        if (self.stim_dims[1]>1) and ((self.stim_dims[2]>1)):
            shift_spacing = np.sqrt(Nspace/num_shifts)
            if filter_width == Nspace:
                filter_width = [self.stim_dims[2],self.stim_dims[1]]  # if default to stim dimensions
            else:
                filter_width = np.sqrt(filter_width) # if entered lower filter width, it will be a square

        # Also might need to modify outputs of first layer

        additional_params = {'first_filter_size': int(filter_width),
                             'binocular': self.binocular,
                             'shift_spacing': int(shift_spacing) }

        target = super(SInetNIM, self).copy_model(
            target = target,
            layers_to_transfer = layers_to_transfer,
            target_layers = target_layers,
            init_type=init_type, tf_seed=tf_seed )
        # the rest of the properties will be copied directly, which includes ei_layer stuff, act_funcs
        return target
    # END SInetNIM.make_copy

    def create_NIM_copy( self, init_type=None, tf_seed=None, other_network_params = None ):

        if other_network_params is not None:
            FFparams = other_network_params
        else:
            FFparams = self.network_params

        target = SInetNIM( FFparams, self.num_examples,
                           noise_dist=self.noise_dist,
                           learning_alg=self.learning_alg,
                           learning_rate=self.learning_rate,
                           use_batches=self.use_batches,
                           tf_seed=tf_seed,
                           use_gpu=self.use_gpu )
        # copy weights and stuff
        return target
    # END SInetNIM.create_new_NIM


class siFFNetwork( FFNetwork ):
    """Implementation of siFFNetwork"""

    def __init__( self, scope=None, inputs=None, params_dict=None ):
        """Constructor for Network class

        Args:
            scope (str): name scope for network
            inputs (tf Tensor or placeholder): input to network
            params_dict (dictionary with the following fields):
                SI-NETWORK SPECIFIC:
                -> first_filter_size: size of filters in first layer, if different than input size
                    DEFAULT = input size
                -> shift_spacing (int): convolutional "strides" to be passed back into conv2d
                    DEFAULT = 1
                -> binocular (boolean): currently doesn't work
                    DEFAULT = FALSE
                    
                INHERITED FFNetwork PARAMS (see FFNetwork documentation): 
                -> layer_sizes (list of ints)
                -> activation_funcs (str or list of strs, optional)
                -> weights_initializer (str or list of strs, optional)
                -> biases_initializer (str or list of strs, optional)
                -> reg_initializers (list of dicts)
                -> num_inh (None, int or list of ints, optional)
                -> pos_constraint (bool or list of bools, optional):
                -> log_activations (bool, optional)

        Raises:
            TypeError: If `scope` is not specified
            TypeError: If `inputs` is not specified
            TypeError: If `layer_sizes` is not specified
        """

        super(siFFNetwork, self).__init__(
            scope = scope,
            inputs = inputs,
            params_dict = params_dict )
    # END siFFnetwork.__init__

    def _define_network( self, inputs, network_params ):

        # Pull out useful params from network_params
        layer_sizes = network_params['layer_sizes']
        # Additional si-specific params
        first_filter_size = network_params['first_filter_size']
        shift_spacing = network_params['shift_spacing']
        binocular = network_params['binocular']

        self.layers = []
        num_conv_layers = network_params['num_conv_layers']
        for nn in range(num_conv_layers):
            self.layers.append(
                siLayer( scope = 'conv_layer_%i' % nn,
                         inputs = inputs,
                         filter_size = first_filter_size,
                         shift_spacing = shift_spacing,
                         binocular = binocular,
                         input_dims = layer_sizes[nn],
                         num_filters = layer_sizes[nn+1],
                         activation_func = network_params['activation_funcs'][nn],
                         weights_initializer = network_params['weights_initializers'][nn],
                         biases_initializer = network_params['biases_initializers'][nn],
                         reg_initializer = network_params['reg_initializers'][nn],
                         num_inh = network_params['num_inh'][nn],
                         pos_constraint = network_params['pos_constraints'][nn],
                         log_activations = network_params['log_activations'] ) )

            # num_inputs to next layer is adjusted by number of shifts, so recalculate
            inputs = self.layers[-1].outputs
            layer_sizes[nn+1] = self.layers[nn].output_dims

        # Insert max layer if desired
        if network_params['max_layer']:
            self.layers.append(
                max_layer( scope='max_layer',
                           inputs=inputs,
                           input_dims=layer_sizes[num_conv_layers],
                           output_dims=layer_sizes[num_conv_layers+1],
                           activation_func=network_params['activation_funcs'][num_conv_layers],
                           weights_initializer=network_params['weights_initializers'][num_conv_layers],
                           biases_initializer=network_params['biases_initializers'][num_conv_layers],
                           reg_initializer=network_params['reg_initializers'][num_conv_layers],
                           num_inh=network_params['num_inh'][num_conv_layers],
                           pos_constraint=network_params['pos_constraints'][num_conv_layers],
                           log_activations=network_params['log_activations']))
            num_conv_layers = num_conv_layers + 1
            inputs = self.layers[-1].outputs

        # Attach rest of layers -- just like FFNetwork constructor
        for layer in range(num_conv_layers, self.num_layers):
            self.layers.append(
                Layer( scope = 'layer_%i' % layer,
                       inputs = inputs,
                       input_dims = layer_sizes[layer],
                       output_dims = layer_sizes[layer + 1],
                       activation_func=network_params['activation_funcs'][layer],
                       weights_initializer=network_params['weights_initializers'][layer],
                       biases_initializer=network_params['biases_initializers'][layer],
                       reg_initializer=network_params['reg_initializers'][layer],
                       num_inh=network_params['num_inh'][layer],
                       pos_constraint=network_params['pos_constraints'][layer],
                       log_activations=network_params['log_activations']))
            inputs = self.layers[-1].outputs

    # END siFFNetwork._define_network



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
            input_dims=None, # this can be a list up to 3-dimensions
            num_filters=None,
            filter_size=None, # this can be a list up to 3-dimensions
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

        # Error checking
        assert pos_constraint is False, 'No positive constraint should be applied to this layer.'

        # Process stim and filter dimensions (potentially both passed in as num_inputs list)
        if isinstance( input_dims, list ):
            while len(input_dims)<3:
                input_dims.append(1)
        else:
            input_dims = [1,input_dims,1]  # assume 1-dimensional (space)

        if filter_size is None:
            filter_size = input_dims
            if binocular is True:
                filter_size[1] = filter_size[1] / 2
        else:
            if isinstance(filter_size,list):
                while len(filter_size) < 3:
                    filter_size.extend(1)
            else:
                filter_size = [filter_size,1,1]

        # If output dimensions already established, just strip out num_filters
        if isinstance( num_filters, list):
            num_filters = num_filters[0]

        # Calculate number of shifts (for output)
        num_shifts = [1,1]
        if input_dims[1] > 1:
            num_shifts[0] = int(np.floor(input_dims[1]/shift_spacing))
        if input_dims[2] > 1:
            num_shifts[1] = int(np.floor(input_dims[2]/shift_spacing))

        # Set up additional params_dict for siLayer
        siLayer_params = {'binocular': binocular, 'shift_spacing': shift_spacing,
                          'num_shifts': num_shifts, 'filter_size': filter_size}

        super(siLayer, self).__init__(
                scope=scope,
                inputs=inputs,
                input_dims = input_dims,
                filter_dims = filter_size,
                output_dims = num_filters,   # Note difference from layer
                activation_func=activation_func,
                weights_initializer=weights_initializer,
                biases_initializer=biases_initializer,
                reg_initializer=reg_initializer,
                num_inh=num_inh,
                pos_constraint=False,  # Note difference from layer
                log_activations=log_activations,
                additional_params_dict=siLayer_params )

        self.input_dims = input_dims
        self.num_filters = num_filters
        self.output_dims = [num_filters] + num_shifts  # note this is implicitly multi-dimensional
        #print('conv LAYER output: ',self.num_outputs)

    # END siLayer.__init__

    def _define_network( self, inputs, params_dict=None ):
        # push data through layer

        # Unfold siLayer-specific parameters for building graph
        binocular = params_dict['binocular']
        shift_spacing = params_dict['shift_spacing']
        filter_size = params_dict['filter_size']
        num_shifts = params_dict['num_shifts']

        # Reshape of inputs (4-D):
        input_dims = [-1, self.input_dims[2], self.input_dims[1], self.input_dims[0]]
        # this is reverse-order from Matlab: [space-2, space-1, lags, and num_examples]
        shaped_input = tf.reshape(inputs, input_dims)

        # Reshape weights (4:D:
        conv_filter_dims = [filter_size[2], filter_size[1], filter_size[0], self.num_filters ]
        ws_conv = tf.reshape(self.weights_var, conv_filter_dims)
        # this is reverse-order from Matlab: [space-2, space-1, lags] and num_filters is explicitly last dim

        # Make strides list
        strides = [1,1,1,1]
        if conv_filter_dims[1]>1:
            strides[1]=shift_spacing
        if conv_filter_dims[2] > 1:
            strides[2] = shift_spacing

        if binocular is False:
            pre = tf.nn.conv2d( shaped_input, ws_conv, strides, padding='SAME' ) #, data_format="NCHW")
        else:
            NX = params_dict['input_dims'][1]/2
            stimL = tf.slice( shaped_input, [0,0,0,0], [-1,NX,-1,-1] )
            stimR = tf.slice( shaped_input, [0,NX,0,0], [-1,NX,-1,-1] )
            pre = tf.concat(
                tf.nn.conv2d(stimL, ws_conv, strides, padding='SAME'),
                tf.nn.conv2d(stimR, ws_conv, strides, padding='SAME'))

        if self.ei_mask_var is not None:
            post = tf.multiply(tf.add(self.activation_func(pre), self.biases_var), self.ei_mask_var)
        else:
            post = tf.add(self.activation_func(pre), self.biases_var)

        self.outputs = tf.reshape(post, [-1, self.num_filters * num_shifts[0] * num_shifts[1]])

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
    # END siLayer._build_layer

class max_layer(Layer):
    """Implementation of layer which limits single weights along specified input dimension.
    It will always follow the last convolutional layer (siLayer) in siNIM

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
            input_dims=None,  # this can be a list up to 3-dimensions
            output_dims=None,
            activation_func='relu',
            weights_initializer='trunc_normal',
            biases_initializer='zeros',
            reg_initializer=None,
            num_inh=0,
            pos_constraint=False,
            log_activations=False ):
        """Constructor for Layer class

        Args:
            scope (str): name scope for variables and operations in layer
            inputs (tf Tensor or placeholder): input to layer
            input_dims (int): dimensions of input data
            output_dims (int): dimensions of output data
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

        super(max_layer, self).__init__(
                scope=scope,
                inputs=inputs,
                input_dims = input_dims,
                output_dims = output_dims,
                activation_func = activation_func,
                weights_initializer = weights_initializer,
                biases_initializer = biases_initializer,
                reg_initializer = reg_initializer,
                num_inh = num_inh,
                pos_constraint=pos_constraint,
                log_activations=log_activations )

        #print('max LAYER output: ',self.num_outputs)

    # END maxLayer.__init__

    def _define_network( self, inputs, params_dict=None ):
        # push data through max_layer

        # Reshape input and output to handle weights and shifts separately
        num_shifts = self.input_dims[1]*self.input_dims[2]
        num_input_filts = self.input_dims[0]
        shaped_input = tf.transpose( tf.reshape( inputs, [-1,num_shifts,num_input_filts,1] ), perm=[1,2,0,3] )

        if self.pos_constraint:
            shaped_weights = tf.reshape(
                tf.maximum(0.0, self.weights_var), [num_shifts, num_input_filts, 1, self.num_filters])
        else:
            shaped_weights = tf.reshape( self.weights_var, [num_shifts, num_input_filts, 1, self.num_filters] )

        #print(shaped_weights)
        #print(shaped_input)
        #test = tf.matmul( shaped_input, shaped_weights )
        #print('test', test)

        pre_max_over_shifts = tf.reduce_max( tf.matmul(shaped_input,shaped_weights), axis=0 )
        pre_sum_over_filters = tf.reduce_sum( pre_max_over_shifts, axis=0 )
        pre = tf.add( pre_sum_over_filters, self.biases_var)

        if self.ei_mask_var is not None:
            post = tf.multiply(self.activation_func(pre), self.ei_mask_var)
        else:
            post = self.activation_func(pre)

        self.outputs = post

        if self.log:
            tf.summary.histogram('act_pre', pre)
            tf.summary.histogram('act_post', post)
        # END maxLayer._define_network
