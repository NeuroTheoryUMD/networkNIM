from __future__ import division
import numpy as np
from scipy.linalg import toeplitz

def FFnetwork_params( stim_dims = None,
                      layer_sizes = None,
                      ei_layers = None,
                      act_funcs = 'relu',
                      reg_list = None,
                      xstim_n = 0,
                      ffnet_n = None,
                      verbose = True,
                      num_conv_layers = 0, # the below are for convolutional network (SIN-NIM)
                      max_layer = None,
                      conv_filter_widths = None,
                      shift_spacing = 1,
                      binocular = False ):
    """This generates the information for the network_params dictionary that is passed into
    the constructor for the NetworkNIM. It has the following input arguments:
      -> stim_dims
      -> layer_sizes: list of number of subunits in each layer of network. Last layer should match number of
            neurons (Robs). Each entry can be a 3-d list, if there is a spatio-filter/temporal arrangement.
      -> ei_layers: if this is not none, it should be a list of # of inhibitory units for each layer other than
            the output_layer: so list should be of length one-less than layer_sizes. All the non-inhibitory units
            are of course excitatory, and having 'None' for a layer means it will be unrestricted.
      -> act_funcs: (str or list of strs, optional): activation function for network layers; replicated if a
            single element.
            ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' | 'elu' | 'quad' | 'lin'
      -> xstim_n: data-structure to process (in the case that there are more than one in the model). It should
            be 'None' if the network will be directed internally (see ffnet_n)
      -> ffnet_n: internal network that received input from (has to be None if xstim_n is used)
      This function can also add parameters specific to the SinNIM if num_conv_layers > 0
      -> conv_layers: number of convolutional layers
      -> max_layer: whether the layer following the convolution is a max-layer. None [default] means it is not,
            and otherwise specify type:
             1 = max over all filters and shifts
             2 = max over shifts only (but sum over filters)
             3 = max over filters only (but sum over shifts)
      -> first_filter_width: spatial dimension of filter (if different than stim_dims)
      -> shift_spacing: how much shift in between each convolutional operation
    """

    if layer_sizes is None:
        raise TypeError('Must specify layer_sizes.')
    if ei_layers is not None:
        assert len(ei_layers) == (len(layer_sizes)-1), \
            'ei_layers must be a list one less than the number of layers (in layer_sizes)'
    if stim_dims is None:
        raise TypeError('Must specify stimulus dimensions')
    if not isinstance(stim_dims, list):
        # then just 1-dimension (place in time)
        stim_dims = [stim_dims, 1, 1]
    else:
        while len(stim_dims) < 3:
            stim_dims.append(1)

    if xstim_n is not None:
        if not isinstance(xstim_n, list):
            xstim_n = [xstim_n]
    else:
        assert ffnet_n is not None, 'Must assign some input source.'
    if ffnet_n is not None:
        if not isinstance(ffnet_n, list):
            ffnet_n = [ffnet_n]

    # Build layer_sizes and ei_layers
    layer_sizes = [stim_dims] + layer_sizes

    # Establish positivity constraints
    num_layers = len(layer_sizes) - 1
    pos_constraints = [False] * num_layers
    num_inh_layers = [0] * num_layers
    if ei_layers is not None:
        for nn in range(len(ei_layers)):
            if ei_layers[nn] >= 0:
                pos_constraints[nn + 1] = True
                num_inh_layers[nn] = ei_layers[nn]
    if not isinstance(act_funcs, list):
        act_funcs = [act_funcs] * num_layers

    # Reformat regularization information into regularization for each layer
    reg_initializers = []
    for nn in range(num_layers):
        reg_initializers.append({})
        if reg_list is not None:
            for reg_type, reg_val_list in reg_list.iteritems():
                if not isinstance(reg_val_list, list):
                    if reg_val_list is not None:
                        reg_initializers[nn][reg_type] = reg_val_list
                else:
                    assert len(reg_val_list) == num_layers, 'reg_list length must match number of layers.'
                    if reg_val_list[nn] is not None:
                        reg_initializers[nn][reg_type] = reg_val_list[nn]

    network_params = {
        'xstim_n': xstim_n,
        'ffnet_n': ffnet_n,
        'layer_sizes': layer_sizes,
        'activation_funcs': act_funcs,
        'pos_constraints': pos_constraints,
        'num_inh': num_inh_layers,
        'reg_initializers': reg_initializers }

    # if convolutional, add the following SinNIM-specific fields
    if num_conv_layers > 0:
        network_params['num_conv_layers'] = num_conv_layers

        if max_layer is not None:
            assert max_layer in [1,2,3], 'max_layer must specify types 1-3. 1 = max over all, 2 = max over shifts, ...'
        network_params['max_layer'] = max_layer

        if not isinstance(conv_filter_widths,list):
            conv_filter_widths = [conv_filter_widths]
        while len(conv_filter_widths) < num_conv_layers:
            conv_filter_widths.append(None)
        network_params['conv_filter_widths'] = conv_filter_widths

        network_params['shift_spacing'] = shift_spacing
        network_params['binocular'] = binocular

    if verbose:
        print( 'Stim dimensions: ' + str(layer_sizes[0]) )
        for nn in range(num_conv_layers):
            s = 'Conv Layer ' + str(nn) + ' (' + act_funcs[nn] + '): [E' + str(layer_sizes[nn+1]-num_inh_layers[nn])
            s += '/I' + str(num_inh_layers[nn]) + ']'
            if pos_constraints[nn]:
                s += ' +'
            if conv_filter_widths[nn] is not None:
                s += '  \tfilter width = ' + str(conv_filter_widths[nn])
            print(s)
        for nn in range(num_conv_layers,num_layers):
            s = 'Layer ' + str(nn) + ' (' + act_funcs[nn] + '): [E' + str(layer_sizes[nn+1]-num_inh_layers[nn])
            s += '/I' + str(num_inh_layers[nn]) + ']'
            if pos_constraints[nn]:
                s += ' +'
            print(s)
    return network_params
# END createNIMparams


def shift_mat_zpad( x, shift, dim=0 ):
    # Takes a vector or matrix and shifts it along dimension dim by amount shift using zero-padding.
    # Positive shifts move the matrix right or down

    assert x.ndim < 3, 'only works in 2 dims or less at the moment.'
    if x.ndim == 1:
        oneDarray = True
        xcopy = np.zeros([len(x), 1])
        xcopy[:, 0] = x
    else:
        xcopy = x.copy()
        oneDarray = False
    sz = list(np.shape(xcopy))

    if sz[0] == 1:
        dim = 1

    if dim == 0:
        if shift >= 0:
            a = np.zeros((shift, sz[1]))
            b = xcopy[0:sz[0]-shift, :]
            xshifted = np.concatenate((a, b), axis=dim)
        else:
            a = np.zeros((-shift, sz[1]))
            b = xcopy[-shift:, :]
            xshifted = np.concatenate((b, a), axis=dim)
    elif dim == 1:
        if shift >= 0:
            a = np.zeros((sz[0], shift))
            b = xcopy[:, 0:sz[1]-shift]
            xshifted = np.concatenate((a, b), axis=dim)
        else:
            a = np.zeros((sz[0], -shift))
            b = xcopy[:, -shift:]
            xshifted = np.concatenate((b, a), axis=dim)

    # If the shift in one direction is bigger than the size of the stimulus in that direction return a zero matrix
    if (dim == 0 and abs(shift) > sz[0]) or (dim == 1 and abs(shift) > sz[1]):
        xshifted = np.zeros(sz)

    # Make into single-dimension if it started that way
    if oneDarray:
        xshifted = xshifted[:,0]

    return xshifted
# END shit_mat_zpad


def create_time_embedding(stim, pdims, up_fac=1, tent_spacing=1):
    """
    # All the arguments starting with a p are part of params structure which I will fix later
    # Takes a Txd stimulus matrix and creates a time-embedded matrix of size Tx(d*L), where L is the desired
    # number of time lags.
    # If stim is a 3d array, the spatial dimensions are folded into the 2nd dimension.
    # Assumes zero-padding.
    # Optional up-sampling of stimulus and tent-basis representation for filter estimation.
    # Note that xmatrix is formatted so that adjacent time lags are adjacent within a time-slice of the xmatrix, thus
    # x(t, 1:nLags) gives all time lags of the first spatial pixel at time t.
    #
    # INPUTS:
    #           stim: simulus matrix (time must be in the first dim).
    #           params: struct of simulus params (see NIM.create_stim_params)
    # OUTPUTS:
    #           xmat: time-embedded stim matrix
    """

    # Note for myself: pdims[0] is nLags and the rest is spatial dimension

    sz = list(np.shape(stim))

    # If there are two spatial dims, fold them into one
    if len(sz) > 2:
        stim = np.reshape(stim, (sz[0], np.prod(sz[1:])))

    # No support for more than two spatial dimensions
    if len(sz) > 3:
        print 'More than two spatial dimensions not supported, but creating xmatrix anyways...'

    # Check that the size of stim matches with the specified stim_params structure
    if np.prod(pdims[1:]) != sz[1]:
        print 'Stimulus dimension mismatch'
        raise ValueError

    modstim = stim.copy()
    # Up-sample stimulus if required
    if up_fac > 1:
        modstim = np.repeat(modstim, up_fac, 0)  # Repeats the stimulus along the time dimension
        sz = list(np.shape(modstim))  # Since we have a new value for time dimension

    # If using tent-basis representation
    if tent_spacing > 1:
        # Create a tent-basis (triangle) filter
        tent_filter = np.append( np.arange(1,tent_spacing)/tent_spacing, 1-np.arange(tent_spacing)/tent_spacing) / tent_spacing
        # Apply to the stimulus
        filtered_stim = np.zeros(sz)
        for ii in range(len(tent_filter)):
            filtered_stim = filtered_stim + shift_mat_zpad(modstim, ii-tent_spacing+1, 0) * tent_filter[ii]
        modstim = filtered_stim

    sz = list(np.shape(modstim))
    lag_spacing = tent_spacing  # If ptent_spacing is not given in input then manually put lag_spacing = 1
    # For temporal-only stimuli (this method can be faster if you're not using tent-basis rep)
    # For myself, add: & tent_spacing is empty (= & isempty...).  Since isempty(tent_spa...) is equivalent to
    # its value being 1 I added this condition to the if below temporarily:
    if sz[1] == 1 and tent_spacing == 1:
        xmat = toeplitz(np.reshape(modstim, (1, sz[0])), np.concatenate((modstim[0], np.zeros(pdims[0] - 1)), axis=0))
    else:  # Otherwise loop over lags and manually shift the stim matrix
        xmat = np.zeros((sz[0], np.prod(pdims)))
        for lag in range(pdims[0]):
            for xx in range(0, sz[1]):
                xmat[:, xx*pdims[0]+lag] = shift_mat_zpad(modstim[:, xx], lag_spacing*lag, 0)

    return xmat
# END create_time_embedding

