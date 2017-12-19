"""metaNIM"""

from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from FFnetwork.ffnetwork import FFNetwork
from FFnetwork.network import Network


class metaNIM(Network):
    """Tensorflow (tf) implementation of network-NIM class

    Attributes:
        network (FFNetwork object): feedforward network
    """

    _allowed_noise_dists = ['gaussian', 'poisson', 'bernoulli']

    def __init__(
            self,
            network_list,
            ffnet_out=-1,
            noise_dist='poisson',
            tf_seed=0 ):

        """Constructor for network-NIM class

        Args:
            network_list: created using 'FFNetwork_params', which has at least the following
        """

        # input checking
        # if num_examples is None:
        #    raise ValueError('Must specify number of training examples')

        # call __init__() method of super class
        super(metaNIM, self).__init__()

        if not isinstance( network_list, list ):
            network_list = [network_list]

        # set model attributes from input
        self.network_list = network_list
        self.num_input_streams = 0
        self.num_networks = len(network_list)
        if not isinstance(ffnet_out,list):
            ffnet_out = [ffnet_out]
        for nn in range(len(ffnet_out)):
            assert ffnet_out[nn] <= self.num_networks
        self.ffnet_out = ffnet_out
        self.input_size = [0]  # list of input sizes (for stimulus placeholders)
        self.output_size = [0]  # list of output sizes (for Robs placeholders)
        self.noise_dist = noise_dist
        self.tf_seed = tf_seed

        self._define_network(network_list)

        # set parameters for graph (constructed for each train)
        self.graph = None
        self.saver = None
        self.merge_summaries = None
        self.init = None

    # END networkNIM.__init__

    def _define_network(self, network_list):
        # Create the FFnetworks

        self.networks = []
        for nn in range(self.num_networks):
            # Check validity of network inputs
            if network_list[nn]['ffnet_n'] is not None:
                ffnet_n = network_list[nn]['ffnet_n']
                for mm in ffnet_n:
                    assert ffnet_n[mm] <= self.num_networks, 'Too many ffnetworks referenced.'
            # Determine inputs
            if network_list[nn]['xstim_n'] is not None:
                xstim_n = network_list[nn]['xstim_n']
                for mm in xstim_n:
                    if mm > self.num_input_streams:
                        self.num_input_streams = mm
                    stim_dims = network_list[nn]['layer_sizes'][0]
                    self.input_size[mm] = np.prod(stim_dims)

            # Build networks
            self.networks.append(
                FFNetwork(scope='network_%i' % nn, params_dict=network_list[nn]))

        # Assemble outputs
        for nn in range(len(self.ffnet_out)):
            ffnet_n = self.ffnet_out[nn]
            self.output_size[ffnet_n] = self.networks[ffnet_n].layers[-1].weights.shape[1]

        #print('num_inputs = ', self.input_size)
        #print('num_outputs = ', self.output_size)

    # END metaNIM._define_network

    def build_graph(self, learning_alg='adam', learning_rate=1e-3 ):

        # Check number of input streams
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
            for nn in range(self.num_networks):
                # Assemble input streams -- implicitly along input axis 1 (0 is T)
                input_cat = None
                if self.network_list[nn]['xstim_n'] is not None:
                    for ii in self.network_list[nn]['xstim_n']:
                        if input_cat is None:
                            input_cat = self.data_in_batch[ii]
                        else:
                            input_cat = tf.concat( (input_cat, self.data_in_batch[ii]), axis=1 )
                if self.network_list[nn]['ffnet_n'] is not None:
                    for ii in self.network_list[nn]['ffnet_n']:
                        if input_cat is None:
                            input_cat = self.networks[ii].layers[-1].outputs
                        else:
                            input_cat = tf.concat( (input_cat, self.networks[ii].layers[-1].outputs), axis=1 )

                self.networks[nn].build_graph(input_cat, self.network_list[nn])

            # Define loss function
            with tf.variable_scope('loss'):
                self._define_loss()

            # Define optimization routine
            with tf.variable_scope('optimizer'):
                self._define_optimizer( learning_alg, learning_rate )

            # add additional ops
            # for saving and restoring models (initialized after var creation)
            self.saver = tf.train.Saver()
            # collect all summaries into a single op
            self.merge_summaries = tf.summary.merge_all()
            # add variable initialization op to graph
            self.init = tf.global_variables_initializer()

    def _define_loss(self):
        """Loss function that will be used to optimize model parameters"""

        cost = 0.0
        for nn in range(len(self.ffnet_out)):
            data_out = self.data_out_batch[0]
            pred = self.networks[nn].layers[-1].outputs

            # define cost function
            if self.noise_dist == 'gaussian':
                with tf.name_scope('gaussian_loss'):
                    # should variable 'cost' be defined here too?
                    cost += tf.nn.l2_loss(data_out - pred) / pred.shape[0]
                    self.unit_cost = tf.reduce_mean(tf.square(data_out-pred), axis=0)

            elif self.noise_dist == 'poisson':
                with tf.name_scope('poisson_loss'):
                    cost_norm = tf.maximum( tf.reduce_sum(data_out, axis=0), 1)
                    cost += -tf.reduce_sum( tf.divide(
                        tf.multiply(data_out,tf.log(self._log_min + pred)) - pred,
                        cost_norm ) )
                    self.unit_cost = tf.divide( -tf.reduce_sum(
                        tf.multiply(data_out,tf.log(self._log_min + pred)) - pred, axis=0), cost_norm )

            elif self.noise_dist == 'bernoulli':
                with tf.name_scope('bernoulli_loss'):
                    # Check per-cell normalization with cross-entropy
                    cost_norm = tf.maximum( tf.reduce_sum(data_out, axis=0), 1)
                    cost += tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=data_out,logits=pred) )
                    self.unit_cost = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=data_out,logits=pred), axis=0 )
                    #cost = tf.reduce_sum(self.unit_cost)
            else:
                print('Cost function not supported.')

        self.cost = cost

        # add regularization penalties
        self.cost_reg = 0
        with tf.name_scope('regularization'):
            for nn in range(self.num_networks):
                self.cost_reg += self.networks[nn].define_regularization_loss()

        self.cost_penalized = tf.add(self.cost, self.cost_reg)

        # save summary of cost
        with tf.variable_scope('summaries'):
            tf.summary.scalar('cost', cost)
    # END metaNIM._define_loss

    def _assign_model_params(self, sess):
        """Functions assigns parameter values to randomly initialized model"""
        with self.graph.as_default():
            for nn in range(self.num_networks):
                self.networks[nn].assign_model_params(sess)

    def _write_model_params(self, sess):
        """Pass write_model_params down to the multiple networks"""
        for nn in range(self.num_networks):
            self.networks[nn].write_model_params(sess)

    def _assign_reg_vals(self, sess):
        """Loops through all current regularization penalties and updates
        parameter values"""
        with self.graph.as_default():
            for nn in range(self.num_networks):
                self.networks[nn].assign_reg_vals(sess)

    def _build_fit_variable_list( self, fit_parameter_list ):
        """Generates variable list to fit if argument is not none. 'fit_parameter_list'
        is generated by a """
        var_list = None
        if fit_parameter_list is not None:
            var_list = []
            for nn in range(self.num_networks):
                var_list += self.networks[nn]._build_fit_variable_list( fit_parameter_list[nn] )
        return var_list
    # END metaNIM._generate_variable_list

    def variables_to_fit(self, layers_to_skip=None, fit_biases=False):
        """Generates a list-of-lists-of-lists of correct format to specify all the
        variables to fit, as an argument for network.train

        Inputs:
            layers_to_skip: [default=None] this should be a list-of-lists, specifying
                a list of layers to skip for each network. If just single list, will assume
                this is skipping layers in the first network
            fit_biases: [default=False] this can be a single boolean value or list of values
                if want networks to have different default-bias-fitting states
            """

        if layers_to_skip is None:
            layers_to_skip = []
        else:
            if not isinstance(layers_to_skip,list):
                layers_to_skip = [layers_to_skip]
        if isinstance(fit_biases,list):
            assert len(fit_biases) == self.num_networks, 'fit_biases list must match the number of networks.'
        else:
            fit_biases = [fit_biases]*self.num_networks

        fit_list = []*self.num_networks
        for nn in range(self.num_networks):
            fit_list[nn] = [{}]*self.networks[nn].num_layers
            for layer in range(self.networks[nn].num_layers):
                fit_list[nn][layer]['weights']=True
                fit_list[nn][layer]['biases']=fit_biases
                if nn <= len(layers_to_skip):
                    if layer in layers_to_skip[nn]:
                        fit_list[nn][layer]['weights'] = False
                        fit_list[nn][layer]['biases'] = False

        return fit_list
        # END metaNIM.set_fit_variables

    def set_regularization(self, reg_type, reg_val, ffnet_n=0, layer_target=None):
        """Add or reassign regularization values

        Args:
            reg_type (str): see allowed_reg_types in regularization.py
            reg_val (int): corresponding regularization value
            ffnet_n(int): which network to assign regularization to (default = 0)
            layer_target (int or list of ints): specifies which layers the
                current reg_type/reg_val pair is applied to (default all in ffnet_n)

        """

        if layer_target is None:
            # set all layers
            for nn in range(self.num_networks):
                layer_target = range(self.networks[ffnet_n].num_layers)
        elif not isinstance(layer_target,list):
                layer_target = [layer_target]

        # set regularization at the layer level
        rebuild_graph = False
        for layer in layer_target:
            new_reg_type = self.networks[ffnet_n].layers[layer].set_regularization(
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

    def generate_prediction(self, input_data, data_indxs=None, ffnet_n=-1, layer=-1):

        assert self.graph is not None, 'Must fit model first.'
        # check input
        if layer >= 0:
            assert layer < len(self.networks[ffnet_n].layers), 'This layer does not exist.'

        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)

        # Generate fake_output data
        output_data = np.zeros( [self.num_examples, self.networks[ffnet_n].layers[-1].weights.shape[1]],
                                dtype='float32')

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            self._restore_params(sess, input_data, output_data)
            pred = sess.run(self.networks[ffnet_n].layers[layer].outputs, feed_dict={self.indices: data_indxs})

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


    def copy_model(self, alternate_network_params=None, target=None,
                   layers_to_transfer=None,
                   target_layers=None,
                   init_type='trunc_normal', tf_seed=0):

        num_layers = len(self.network.layers)
        if target is None:
            # Make new target
            target = self.create_NIM_copy(init_type=init_type, tf_seed=tf_seed,
                                          alternate_network_params=alternate_network_params)

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
                target_copy = self_copy  # then default is copy the top_most layers
            else:
                target_copy = target_layers
        assert len(target_copy) == len(self_copy), 'Number of targets and transfers must match.'

        # Copy information from self to new target NIM
        for nn in range(len(self_copy)):
            self_layer = self.network.layers[self_copy[nn]]
            tar = target_copy[nn]
            self_num_outputs = self_layer.output_dims[0] * self_layer.output_dims[1] * self_layer.output_dims[2]
            tar_num_outputs = target.network.layers[tar].output_dims[0] \
                              * target.network.layers[tar].output_dims[1] * target.network.layers[tar].output_dims[2]

            # Copy remaining layer properties
            target.network.layers[tar].ei_mask = self_layer.ei_mask

            if self_num_outputs <= tar_num_outputs:
                target.network.layers[tar].weights[:, 0:self_num_outputs] \
                    = self_layer.weights
                target.network.layers[tar].biases[0:self_num_outputs] \
                    = self_layer.biases
            else:
                target.network.layers[tar].weights = \
                    self_layer.weights[:, 0:tar_num_outputs]
                target.network.layers[tar].biases = \
                    self_layer.biases[0:tar_num_outputs]

        return target

    # END make_copy

    def create_NIM_copy(self, init_type=None, tf_seed=None, alternate_network_params=None):

        if alternate_network_params is not None:
            network_params = alternate_network_params
        else:
            network_params = self.network_params

        target = metaNIM(network_params,
                            noise_dist=self.noise_dist,
                            learning_alg=self.learning_alg,
                            learning_rate=self.learning_rate,
                            use_batches=self.use_batches,
                            tf_seed=tf_seed,
                            use_gpu=self.use_gpu)
        return target
        # END metaNIM.create_new_NIM
