"""Network-NIM

@author: Dan Butts (modified from code by Matt Whiteway), July 2017

"""

from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from FFnetwork.ffnetwork import FFNetwork
from FFnetwork.network import Network


class NetworkNIM(Network):
    """Tensorflow (tf) implementation of network-NIM class

    Attributes:
        network (FFNetwork object): feedforward network
        input_dims (int): dimensions of stimulus (up to 3-D)
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

    _allowed_noise_dists = ['gaussian', 'poisson', 'bernoulli']
    _allowed_learning_algs = ['adam', 'lbfgs']

    def __init__(
            self,
            stim_dims,
            num_examples,
            num_neurons=1,
            num_subunits=None,  # default is LN model with no hidden layers
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
            use_gpu=None,
            additional_params=None):
        """Constructor for network-NIM class

        Args:
            stim_dims: int or list representing number of stimulus dimensions
                (must ultimately match X-matrix)
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

        # call __init__() method of super class
        super(NetworkNIM, self).__init__()

        # process stim-dims into 3-d dimensions
        if not isinstance(stim_dims,list):
            # then just 1-dimension (place in time)
            stim_dims = [stim_dims,1,1]
        else:
            while len(stim_dims) < 3:
                stim_dims.append(1)
        self.input_size = stim_dims[0]*stim_dims[1]*stim_dims[2]

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
        self.num_layers = len(layer_sizes)-1
        pos_constraint = [False]*self.num_layers
        num_inh_layers = [0]*self.num_layers
        for nn in range(len(ei_layers)):
            if ei_layers[nn] >= 0:
                pos_constraint[nn+1] = True
                num_inh_layers[nn] = ei_layers[nn]

        # input checking
        if num_examples is None:
            raise ValueError('Must specify number of training examples')

        # set model attributes from input
        self.stim_dims = stim_dims
        self.output_size = num_neurons
        self.activation_functions = act_funcs
        self.noise_dist = noise_dist
        self.learning_alg = learning_alg
        self.learning_rate = learning_rate
        self.num_examples = num_examples
        self.use_batches = use_batches

        # params for FFNetwork
        network_params = {
            'scope': 'model',
            'layer_sizes': layer_sizes,
            'num_inh': num_inh_layers,
            'activation_funcs': act_funcs,
            'pos_constraint': pos_constraint,
            'weights_initializer': init_type,
            'biases_initializer': 'zeros',
            'reg_initializer': reg_initializer,
            'log_activations': False
        }

        self._build_graph(use_gpu, tf_seed, reg_list, network_params, additional_params )
    # END networkNIM.__init__

    def _build_graph(self, use_gpu, tf_seed, reg_list, network_params, additional_params=None):

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
            self._define_network( network_params, additional_params )

            # Initialize regularization, if there is any
            if reg_list is not None:
                for reg_type, reg_val_list in reg_list.iteritems():
                    if not isinstance(reg_val_list, list):
                        reg_val_list = [reg_val_list]*self.num_layers
                    assert len(reg_val_list) == self.num_layers, \
                        'Need to match number of layers with regularization values.'
                    for nn in range(self.num_layers):
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

    def _define_network(self, network_params, additional_params=None):

        self.network = FFNetwork(inputs=self.data_in_batch,
                                 additional_params=additional_params, **network_params )

    def _define_loss(self):
        """Loss function that will be used to optimize model parameters"""

        data_out = self.data_out_batch
        pred = self.network.layers[-1].outputs

        # define cost function
        if self.noise_dist == 'gaussian':
            with tf.name_scope('gaussian_loss'):
                # should variable 'cost' be defined here too?
                cost = tf.nn.l2_loss(data_out - pred)
                self.unit_cost = tf.reduce_mean(tf.square(data_out-pred),axis=0)
                #cost = tf.reduce_sum(self.unit_cost) # make two separate calculations
                self.cost = cost
        elif self.noise_dist == 'poisson':
            with tf.name_scope('poisson_loss'):
                cost = -tf.reduce_sum(
                    tf.multiply(data_out,tf.log(self._log_min + pred)) - pred )
                self.unit_cost = -tf.reduce_sum(
                    tf.multiply(data_out,tf.log(self._log_min + pred)) - pred, axis=0)
                #cost = tf.reduce_sum(self.unit_cost)
                # normalize by number of spikes
                self.cost = tf.divide(cost, tf.reduce_sum(data_out))
        elif self.noise_dist == 'bernoulli':
            with tf.name_scope('bernoulli_loss'):
                cost = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=data_out,logits=pred) )
                self.unit_cost = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=data_out,logits=pred), axis=0 )
                #cost = tf.reduce_sum(self.unit_cost)
                self.cost = cost

        # add regularization penalties
        with tf.name_scope('regularization'):
            self.cost_reg = self.network.define_regularization_loss()

        self.cost_penalized = tf.add(self.cost, self.cost_reg)

        # save summary of cost
        with tf.variable_scope('summaries'):
            tf.summary.scalar('cost', cost)

    def _assign_model_params(self, sess):
        """Functions assigns parameter values to randomly initialized model"""
        with self.graph.as_default():
            self.network.assign_model_params(sess)

    def _assign_reg_vals(self, sess):
        """Loops through all current regularization penalties and updates
        parameter values"""
        with self.graph.as_default():
            self.network.assign_reg_vals(sess)

    def set_regularization(self, reg_type, reg_val, layer_target=None):
        """Add or reassign regularization values
        
        Args:
            reg_type (str): see allowed_reg_types in regularization.py
            reg_val (int): corresponding regularization value
            layer_target (int or list of ints): specifies which layers the 
                current reg_type/reg_val pair is applied to
                
        """

        if layer_target is None:
            # set all layers
            layer_target = range(self.network.num_layers)

        # set regularization at the layer level
        rebuild_graph = False
        for layer in layer_target:
            new_reg_type = self.network.layers[layer].set_regularization(
                reg_type, reg_val)
            rebuild_graph = rebuild_graph or new_reg_type

        if rebuild_graph:
            with self.graph.as_default():

                # redefine loss function
                with tf.name_scope('loss'):
                    self._define_loss()

                # redefine optimization routine
                with tf.variable_scope('optimizer'):
                    self._define_optimizer()
    # END set_regularization

    def get_LL(self, input_data, output_data, data_indxs=None):
        """Get cost from loss function and regularization terms

        Args:
            input_data (time x input_dim numpy array): input to model
            output_data (time x output_dim numpy array): desired output of 
                model
            data_indxs (numpy array, optional): indexes of data to use in
                calculating forward pass; if not supplied, all data is used

        Returns:
            cost (float): value of model's cost function evaluated on previous
                model data or that used as input
            reg_pen (float): value of model's regularization penalty

        Raises:
            ValueError: If data_in/out time dims don't match

        """

        # check input
        if input_data.shape[0] != output_data.shape[0]:
            raise ValueError('Input and output data must have matching ' +
                             'number of examples')
        if input_data.shape[0] != self.num_examples:
            raise ValueError('Input/output data dims must match model values')

        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            self._restore_params(sess, input_data, output_data)

            cost = sess.run(self.cost,
                            feed_dict={self.indices: data_indxs})

            return cost
    # END get_LL

    def eval_models(self, input_data, output_data, data_indxs=None):
        """Get cost for each output neuron without regularization terms

        Args:
            input_data (time x input_dim numpy array): input to model
            output_data (time x output_dim numpy array): desired output of
                model
            data_indxs (numpy array, optional): indexes of data to use in
                calculating forward pass; if not supplied, all data is used

        Returns:
            cost (float): value of model's cost function evaluated on previous
                model data or that used as input
            reg_pen (float): value of model's regularization penalty

        Raises:
            ValueError: If data_in/out time dims don't match

        """

        # check input
        if input_data.shape[0] != output_data.shape[0]:
            raise ValueError('Input and output data must have matching ' +
                             'number of examples')
        if input_data.shape[0] != self.num_examples:
            raise ValueError('Input/output data dims must match model values')

        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            self._restore_params(sess, input_data, output_data)
            LL_neuron = sess.run(self.unit_cost, feed_dict={self.indices: data_indxs})

            return LL_neuron
    # END get_LL_neuron

    def generate_prediction(self,input_data, data_indxs=None):

        # check input
        if input_data.shape[0] != self.num_examples:
            raise ValueError('Input/output data dims must match model values')

        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)

        # Generate fake_output data
        output_data = np.zeros( [len(data_indxs),self.output_size], dtype='float32' )

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            self._restore_params(sess, input_data, output_data)
            pred = sess.run(self.network.layers[-1].outputs, feed_dict={self.indices: data_indxs})

            return pred
    # END NetworkNIM.generate_prediction

    def get_reg_pen(self):
        """Return reg penalties in a dictionary"""

        reg_dict = {}
        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            # initialize all parameters randomly
            sess.run(self.init)

            # overwrite randomly initialized values of model with stored values
            self._assign_model_params(sess)

            # update regularization parameter values
            self._assign_reg_vals(sess)

            with tf.name_scope('get_reg_pen'):  # to keep the graph clean-ish
                for layer in range(self.network.num_layers):
                    reg_dict['layer%i' % layer] = \
                            self.network.layers[layer].get_reg_pen(sess)

        return reg_dict
    # END get_reg_pen


    def copy_model( self, target = None,
                    layers_to_transfer = None,
                    target_layers = None,
                    additional_params = None,
                    init_type='trunc_normal', tf_seed=0 ):

        num_layers = len(self.network.layers)
        if target is None:
            # Re-derive subunit layers and ei-masks
            num_subunits = [0]*(num_layers-1)
            ei_layers = [-1]*(num_layers-1)
            for nn in range(len(num_subunits)):
                num_subunits[nn] = self.network.layers[nn].num_outputs
                if self.network.layers[nn+1].pos_constraint:
                    ei_layers[nn] = 0  # will copy ei_mask later

            # Accumulate regularization list from layers
            reg_list = {}
            for reg_type, reg_vals in self.network.layers[0].reg.vals.iteritems():
                reg_list[reg_type] = [None]*num_layers
            for nn in range(num_layers):
                for reg_type, reg_vals in self.network.layers[nn].reg.vals.iteritems():
                    reg_list[reg_type][nn] = self.network.layers[nn].reg.vals[reg_type]

            # Make new target
            target = self.create_NIM_copy( num_subunits = num_subunits,
                                           ei_layers=ei_layers,
                                           reg_list=reg_list,
                                           init_type=init_type,
                                           tf_seed=tf_seed,
                                           additional_params = additional_params )
            # the rest of the properties will be copied directly, which includes ei_layer stuff, act_funcs


        # Figure out mapping from self.layers to target.layers
        num_layers_target = len(target.network.layers)
        if layers_to_transfer is not None:
            assert max(layers_to_transfer) <= num_layers, 'Too many layers to transfer.'
            if target_layers is None:
                assert len(layers_to_transfer) <= num_layers_target, 'Too many layers to transfer.'
                target_layers = range(len(layers_to_transfer))
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
    # END make_copy

    def create_NIM_copy( self, num_subunits, ei_layers, reg_list, init_type,
                        tf_seed, additional_params=None ):

        target = NetworkNIM(self.stim_dims, self.num_examples,
                            num_neurons=self.output_size,
                            num_subunits=num_subunits,
                            act_funcs=self.activation_functions,
                            ei_layers=ei_layers,
                            noise_dist=self.noise_dist,
                            reg_list=reg_list,
                            init_type=init_type,
                            learning_alg=self.learning_alg,
                            learning_rate=self.learning_rate,
                            use_batches=self.use_batches,
                            tf_seed=tf_seed,
                            use_gpu=self.use_gpu )

        return target
    # END NetworkNIM.create_new_NIM
