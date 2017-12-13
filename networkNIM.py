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
            network_params,
            noise_dist='poisson',
            learning_alg='lbfgs',
            learning_rate=1e-3,
            use_batches=False,
            tf_seed=0,
            use_gpu=None ):
        """Constructor for network-NIM class

        Args:
            network_params: created using 'createNIMparams', which has at least the following
                arguments that get passed to FFnetwork (and possibly more):
                -> layer_sizes (list of integers, or 3-d lists of integers)
                -> activation_funcs (list of strings)
                -> pos_constraints (list of Booleans)
            num_examples (int): total number of examples in training data
            noise_dist (str, optional): noise distribution used by network
                ['gaussian'] | 'poisson' | 'bernoulli'
            learning_alg (str, optional): algorithm used for learning
                parameters. 
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

        # input checking
        #if num_examples is None:
        #    raise ValueError('Must specify number of training examples')

        # call __init__() method of super class
        super(NetworkNIM, self).__init__()

        # process stim-dims into 3-d dimensions
        stim_dims = network_params['layer_sizes'][0]
        if not isinstance(stim_dims, list):
            # then just 1-dimension (place in time)
            stim_dims = [stim_dims,1,1]
        else:
            while len(stim_dims) < 3:
                stim_dims.append(1)
        # make a list out of this
        self.input_size = [stim_dims[0]*stim_dims[1]*stim_dims[2]]


        # set model attributes from input
        self.network_params = network_params  # kindof redundant, but currently not a better way...
        self.stim_dims = stim_dims
        self.output_size = [network_params['layer_sizes'][-1]] # make a list
        self.num_layers = len(network_params['layer_sizes'])-1
        self.activation_functions = network_params['activation_funcs']
        self.noise_dist = noise_dist
        self.learning_alg = learning_alg
        self.learning_rate = learning_rate
        self.use_batches = use_batches
        self.use_gpu = use_gpu
        self.tf_seed = tf_seed

        self._define_network( network_params )

        # set parameters for graph (constructed for each train)
        self.graph = None
        self.saver = None
        self.merge_summaries = None
        self.init = None

    # END networkNIM.__init__

    def _define_network( self, network_params ):

        self.network = FFNetwork( scope = 'FFNetwork',
                                  params_dict = network_params )

    def build_graph(self):

        # for saving and restoring models
        self.graph = tf.Graph()  # must be initialized before graph creation

        # for specifying device
        if self.use_gpu is not None:
            if self.use_gpu:
                self.sess_config = tf.ConfigProto(device_count={'GPU': 1})
            else:
                self.sess_config = tf.ConfigProto(device_count={'GPU': 0})

        # build model graph
        with self.graph.as_default():

            np.random.seed(self.tf_seed)
            tf.set_random_seed(self.tf_seed)

            # define pipeline for feeding data into model
            with tf.variable_scope('data'):
                self._initialize_data_pipeline()

            # Build network graph
            self.network.build_graph( self.data_in_batch[0], self.network_params )

            # Define loss function
            with tf.variable_scope('loss'):
                self._define_loss()

            # Define optimization routine
            with tf.variable_scope('optimizer'):
                self._define_optimizer()

            # add additional ops
            # for saving and restoring models (initialized after var creation)
            self.saver = tf.train.Saver()
            # collect all summaries into a single op
            self.merge_summaries = tf.summary.merge_all()
            # add variable initialization op to graph
            self.init = tf.global_variables_initializer()

    def _define_loss(self):
        """Loss function that will be used to optimize model parameters"""

        data_out = self.data_out_batch[0]
        pred = self.network.layers[-1].outputs

        # define cost function
        cost = 0.0
        if self.noise_dist == 'gaussian':
            with tf.name_scope('gaussian_loss'):
                # should variable 'cost' be defined here too?
                cost = tf.nn.l2_loss(data_out - pred) / pred.shape[0]
                self.unit_cost = tf.reduce_mean(tf.square(data_out-pred), axis=0)

        elif self.noise_dist == 'poisson':
            with tf.name_scope('poisson_loss'):
                cost_norm = tf.maximum( tf.reduce_sum(data_out, axis=0), 1)
                cost = -tf.reduce_sum( tf.divide(
                    tf.multiply(data_out,tf.log(self._log_min + pred)) - pred,
                    cost_norm ) )
                self.unit_cost = tf.divide( -tf.reduce_sum(
                    tf.multiply(data_out,tf.log(self._log_min + pred)) - pred, axis=0), cost_norm )

        elif self.noise_dist == 'bernoulli':
            with tf.name_scope('bernoulli_loss'):
                # Check per-cell normalization with cross-entropy
                cost_norm = tf.maximum( tf.reduce_sum(data_out, axis=0), 1)
                cost = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=data_out,logits=pred) )
                self.unit_cost = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=data_out,logits=pred), axis=0 )
                #cost = tf.reduce_sum(self.unit_cost)
        else:
            print('Cost function not supported.')

        self.cost = cost

        # add regularization penalties
        with tf.name_scope('regularization'):
            self.cost_reg = self.network.define_regularization_loss()

        self.cost_penalized = tf.add(self.cost, self.cost_reg)

        # save summary of cost
        with tf.variable_scope('summaries'):
            tf.summary.scalar('cost', cost)
    # END NetworkNIM._define_loss

    def _assign_model_params(self, sess):
        """Functions assigns parameter values to randomly initialized model"""
        with self.graph.as_default():
            self.network.assign_model_params(sess)

    def _write_model_params(self, sess):
        """Pass write_model_params down to network"""
        self.network.write_model_params(sess)

    def _assign_reg_vals(self, sess):
        """Loops through all current regularization penalties and updates
        parameter values"""
        with self.graph.as_default():
            self.network.assign_reg_vals(sess)

    def _build_fit_variable_list( self, fit_parameter_list ):
        """Generates variable list to fit if argument is not none. 'fit_parameter_list'
        is generated by a """
        var_list = None
        if fit_parameter_list is not None:
            var_list = self.network._build_fit_variable_list( fit_parameter_list[0] )
        return var_list
    # END NetworkNIM._generate_variable_list

    def variables_to_fit(self, layers_to_skip=None, fit_biases=False):
        """Generates a list-of-lists-of-lists of correct format to specify all the
        variables to fit, as an argument for network.train

        Args:
            layers_to_skip: [default=None] This is a list of layers to not fit variables
            fit_biases: [default=False] Whether to default not fit biases"""

        if layers_to_skip is None:
            layers_to_skip = []
        else:
            if not isinstance(layers_to_skip,list):
                layers_to_skip = [layers_to_skip]

        fit_list = [{}]*self.network.num_layers
        for layer in range(self.network.num_layers):
            fit_list[layer]['weights']=True
            fit_list[layer]['biases']=fit_biases
            if layer in layers_to_skip:
                fit_list[layer]['weights'] = False
                fit_list[layer]['biases'] = False

        return [fit_list]
        # END NetworkNIM.set_fit_variables

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
        for layer in layer_target:
            new_reg_type = self.network.layers[layer].set_regularization(reg_type, reg_val)
        # No longer need to know this since not building graph until train
    # END NetworkNIM.set_regularization

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

        assert self.graph is not None, 'Must fit model first.'
        # check input
        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            self._restore_params(sess, input_data, output_data)
            LL_neuron = sess.run(self.unit_cost, feed_dict={self.indices: data_indxs})

            return LL_neuron
    # END get_LL_neuron

    def generate_prediction(self, input_data, data_indxs=None, layer=-1 ):

        assert self.graph is not None, 'Must fit model first.'
        # check input
        if layer >= 0:
            assert layer < len(self.network.layers), 'This layer does not exist.'

        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)

        # Generate fake_output data
        output_data = np.zeros( [self.num_examples,self.output_size[0]], dtype='float32' )

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            self._restore_params(sess, input_data, output_data)
            pred = sess.run(self.network.layers[layer].outputs, feed_dict={self.indices: data_indxs})

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


    def copy_model( self, alternate_network_params = None, target = None,
                    layers_to_transfer = None,
                    target_layers = None,
                    init_type='trunc_normal', tf_seed=0 ):

        num_layers = len(self.network.layers)
        if target is None:
            # Make new target
            target = self.create_NIM_copy( init_type=init_type, tf_seed=tf_seed,
                                           alternate_network_params=alternate_network_params )

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
            self_num_outputs = self_layer.output_dims[0]*self_layer.output_dims[1]*self_layer.output_dims[2]
            tar_num_outputs = target.network.layers[tar].output_dims[0]\
                              *target.network.layers[tar].output_dims[1]*target.network.layers[tar].output_dims[2]

            # Copy remaining layer properties
            target.network.layers[tar].ei_mask = self_layer.ei_mask

            if self_num_outputs <= tar_num_outputs:
                target.network.layers[tar].weights[:,0:self_num_outputs] \
                    = self_layer.weights
                target.network.layers[tar].biases[0:self_num_outputs] \
                    = self_layer.biases
            else:
                target.network.layers[tar].weights = \
                    self_layer.weights[:,0:tar_num_outputs]
                target.network.layers[tar].biases = \
                    self_layer.biases[0:tar_num_outputs]

        return target
    # END make_copy

    def create_NIM_copy( self, init_type=None, tf_seed=None, alternate_network_params=None ):

        if alternate_network_params is not None:
            network_params = alternate_network_params
        else:
            network_params = self.network_params

        target = NetworkNIM( network_params,
                             noise_dist = self.noise_dist,
                             learning_alg = self.learning_alg,
                             learning_rate = self.learning_rate,
                             use_batches = self.use_batches,
                             tf_seed = tf_seed,
                             use_gpu = self.use_gpu )
        return target
    # END NetworkNIM.create_new_NIM
