"""
Author: Mingjian He <mh105@mit.edu>

ssm module contains a general state-space model class used in SOMATA
"""

from somata.exact_inference import kalman, djkalman, inverse
import numpy as np
import numbers
from collections.abc import Iterable
from copy import deepcopy
from joblib import Parallel, delayed, cpu_count
from sorcery import dict_of
from scipy.linalg import block_diag


class StateSpaceModel(object):
    """ StateSpaceModel is a general object class for state-space modeling in SOMATA """
    # Class attributes (should not be manually changed or be mutable)
    # only listed here as a reference list for all attributes under Ssm

    # Attributes are intentionally not implemented as properties with
    # decorators in order to allow for flexible input types. Type-hints
    # are omitted for the same reason. numpy.asanyarray() will handle
    # type conversions in the end.

    type = 'ssm'
    nstate = None
    ncomp = None
    nchannel = None
    ntime = None
    nmodel = None
    components = None
    comp_nstates = None
    F = None
    Q = None
    mu0 = None
    Q0 = None
    G = None
    R = None
    y = None
    Fs = None
    stackable = ('F', 'Q', 'mu0', 'Q0', 'G', 'R')
    default_G = None

    def __init__(self, components=None, F=None, Q=None, mu0=None, Q0=None, G=None, R=None, y=None, Fs=None):
        """
        Constructor method for StateSpaceModel class
        :param components: a list of independent components in the model
        :param F: transition matrix
        :param Q: state noise covariance matrix
        :param mu0: initial state mean vector
        :param Q0: initial state covariance matrix
        :param G: observation matrix (row major)
        :param R: observation noise covariance matrix
        :param y: observed data (row major, can be multivariate)
        :param Fs: sampling frequency in Hz
        """
        # Instance attributes
        self.F = self._process_constructor_input(F)
        self.Q = self._process_constructor_input(Q)
        self.mu0 = self._process_constructor_input(mu0)
        self.Q0 = self._process_constructor_input(Q0)
        self.R = self._process_constructor_input(R)
        self.y = self._must_be_row(self._process_constructor_input(y))
        self.Fs = np.float64(Fs) if Fs is not None else None

        # Initialize components
        if components is None:
            self.ncomp = 0
            self.components = None
            self.comp_nstates = [0]
        elif isinstance(components, str):
            self._auto_populate_components(components)
        else:
            self._initialize_from_components(components)

        # Initialize observation matrix
        self._initialize_observation_matrix(G)

        # Check all dimensions
        self._check_dimensions()
        self._check_model_stack()

    # Dunder methods - magic methods and arithmetic-like operations
    def __repr__(self):
        """ Unambiguous and concise representation when calling StateSpaceModel() """
        return 'Ssm(' + str(self.ncomp) + ')<' + hex(id(self))[-4:] + '>'

    def __str__(self):
        """ Helpful information when calling print(StateSpaceModel()) """
        print_str = "<Ssm object at " + hex(id(self)) + ">\n " + \
                    "{0:8} = {1: <5} ".format("nstate", str(self.nstate)) + \
                    "{0:8} = {1}\n ".format("ncomp", str(self.ncomp)) + \
                    "{0:8} = {1: <5} ".format("nchannel", str(self.nchannel)) + \
                    "{0:8} = {1}\n ".format("ntime", str(self.ntime)) + \
                    "{0:8} = {1}\n ".format("nmodel", str(self.nmodel)) + \
                    "components = {}\n ".format(str(self.components)) + \
                    "{0:3}.shape = {1: <10} ".format("F", [str(x.shape) if x is not None else 'None' for x
                                                           in [self.F]][0]) + \
                    "{0:3}.shape = {1}\n ".format("Q", [str(x.shape) if x is not None else 'None' for x
                                                        in [self.Q]][0]) + \
                    "{0:3}.shape = {1: <10} ".format("mu0", [str(x.shape) if x is not None else 'None' for x
                                                             in [self.mu0]][0]) + \
                    "{0:3}.shape = {1}\n ".format("Q0", [str(x.shape) if x is not None else 'None' for x
                                                         in [self.Q0]][0]) + \
                    "{0:3}.shape = {1: <10} ".format("G", [str(x.shape) if x is not None else 'None' for x
                                                           in [self.G]][0]) + \
                    "{0:3}.shape = {1}\n ".format("R", [str(x.shape) if x is not None else 'None' for x
                                                        in [self.R]][0]) + \
                    "{0:3}.shape = {1: <10} ".format("y", [str(x.shape) if x is not None else 'None' for x
                                                           in [self.y]][0]) + \
                    "Fs = {0}\n ".format([str(x) + ' Hz' if x is not None else 'None' for x in [self.Fs]][0])
        return print_str

    def __len__(self):
        return self.nmodel

    def __add__(self, other):
        return self._stack(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        return self._permute(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        return self.__mul__(other)

    def __rmatmul__(self, other):
        return self.__matmul__(other)

    def __pow__(self, other: int):
        if other == 0:
            return StateSpaceModel()
        elif other > 0:
            # simply a magic method to call _expand_attr() with deepcopy()
            out_copy = deepcopy(self)
            out_copy._expand_attr()
            return out_copy
        else:
            raise SyntaxError('StateSpaceModel objects cannot be taken to negative power.')

    def __rpow__(self, other: int):
        return self.__pow__(other)

    # Check methods - sanity checks to ensure model validity
    def _check_dimensions(self, check_scope=(0, 1, 2)):
        """
        Check dimensions of instance attributes, ignoring None attributes
        :param check_scope: (int) specifying which ones to check.
            0 = state dimensions,
            1 = observed data dimensions,
            2 = components and observation matrix
        """
        if 0 in check_scope:
            # check the dimension of states
            check_state_tuple = (self.F, self.Q, self.mu0, self.Q0)
            check_state_n = len(check_state_tuple)
            check_state_result = np.zeros(check_state_n, dtype=np.int_)
            for ii in range(check_state_n):
                if check_state_tuple[ii] is not None:
                    attr_shape = check_state_tuple[ii].shape
                    check_state_result[ii] = attr_shape[0]
                    if ii != 2:  # mu0 is a vector therefore no need to check
                        assert attr_shape[0] == attr_shape[1], 'Matrix attribute is not square!'

            nstates = np.unique(check_state_result[np.nonzero(check_state_result)])
            if len(nstates) == 0:
                self.nstate = 0
            else:
                assert len(nstates) == 1, 'Number of states is ambiguous and inconsistent across attributes.'
                self.nstate = int(nstates[0])

        if 1 in check_scope:
            # check the dimension of observation channels
            self.nchannel = 0
            self.ntime = 0
            if self.R is not None:
                assert self.R.shape[0] == self.R.shape[1], 'Matrix attribute is not square!'
                self.nchannel = self.R.shape[0]
            if self.y is not None:
                self.nchannel = self.y.shape[0]
                self.ntime = self.y.shape[1]
                if self.R is not None:
                    assert self.y.shape[0] == self.R.shape[0], \
                        'Observation dimensions are inconsistent across attributes.'

        if 2 in check_scope:
            # check the dimensions of components and observation matrix
            if self.components is not None:
                assert self.nstate == sum([x.default_G.shape[1] for x in self.components]), \
                    'Input components give a different state dimension from that of other attributes.'

            if self.G is not None:
                if self.nchannel > 0:
                    assert self.G.shape[0] == self.nchannel, 'Observation matrix dimension mismatches nchannel.'
                if self.nstate > 0:
                    assert self.G.shape[1] == self.nstate, 'Observation matrix dimension mismatches nstate.'

    def _check_model_stack(self):
        """
        Check the number of alternative models stacked in instance attributes.
        third axis is reserved for stacking different models
        """
        # Different components and y should reside in different instances, therefore not stackable
        assert self.components is None or type(self.components) is list, \
            'Component structure should not be stacked in a single StateSpaceModel instance.'
        assert self.y is None or len(self.y.shape) < 3, \
            'Observed data y should not be stacked in a single StateSpaceModel instance.'
        assert self.Fs is None or isinstance(self.Fs, numbers.Number), \
            'Sampling frequency is not a valid Number type.'

        # The rest of instance attributes can be stacked with different models
        stack_numbers = self._get_stack_numbers()
        model_stack = np.unique(stack_numbers[stack_numbers != 1])
        if len(model_stack) == 0:
            self.nmodel = 1
        else:
            assert len(model_stack) == 1, 'More than one model stack numbers. Invoke __mul__ if trying to permute.'
            self.nmodel = int(model_stack)

    def _check_observed_data(self, other):
        """ Used to check if two objects have the same observed data """
        if self.y is not None and other.y is not None:
            assert self.y.shape == other.y.shape, 'Observed data have different dimensions.'
            # randomly select one time point to spot check
            rindex = np.random.randint(self.y.shape[0])
            cindex = np.random.randint(self.y.shape[1])
            assert self.y[rindex, cindex] == other.y[rindex, cindex], 'Observed data are not identical.'
        if self.Fs is not None and other.Fs is not None:
            assert self.Fs == other.Fs, 'Observed data have different sampling frequencies.'

    # Syntactic sugar methods - useful methods to make manipulations easier
    def _get_stack_numbers(self):
        """ Obtain the number of models stacked in each attribute """
        check_stack_tuple = tuple([getattr(self, x) for x in self.stackable])
        check_stack_n = len(check_stack_tuple)
        stack_numbers = np.ones(check_stack_n, dtype=np.int_)
        for ii in range(check_stack_n):
            if check_stack_tuple[ii] is not None:
                attr_shape = check_stack_tuple[ii].shape
                if len(attr_shape) > 2:
                    stack_numbers[ii] = attr_shape[2]
        return stack_numbers

    def _stack(self, other):
        """ Stack attributes in two objects """
        self._check_observed_data(other)
        new_obj = self._concat_attr(other)
        new_obj._check_model_stack()
        return new_obj

    def _permute(self, other):
        """ Permute attributes in two objects """
        self._check_observed_data(other)
        new_obj = self._concat_attr(other)
        new_obj._expand_attr()
        return new_obj

    def _concat_attr(self, other, attrs=None, ignore_duplicate=True):
        """
        Concatenate two objects via stacking new model attributes,
        duplicate checking is enabled; therefore repeated attributes
        will be ignored by turning add_flag to False, unless the
        ignore_duplicate flag is set to False
        """
        new_obj = deepcopy(self)
        if attrs is None:
            attrs = ('components', 'y', 'Fs') + self.stackable
        elif type(attrs) is not tuple:
            attrs = tuple(attrs)

        for attr_name in attrs:
            self_attr = getattr(self, attr_name)
            other_attr = getattr(other, attr_name)
            if self_attr is None and other_attr is None:
                pass
            elif other_attr is None:
                pass
            elif self_attr is None:
                setattr(new_obj, attr_name, deepcopy(other_attr))
            else:
                if attr_name == 'Fs':
                    assert self_attr == other_attr, 'Observed data have different sampling frequencies.'
                if attr_name == 'components':
                    assert len(self_attr) == len(other_attr), 'Components have different shapes.'
                elif len(other_attr.shape) > 2:
                    for j in range(other_attr.shape[2]):
                        model_attr = other_attr[:, :, j]
                        add_flag = True
                        if len(self_attr.shape) > 2:
                            for k in range(self_attr.shape[2]):
                                if (self_attr[:, :, k] == model_attr).all():  # type: ignore
                                    add_flag = False
                        else:
                            if (self_attr == model_attr).all():  # type: ignore
                                add_flag = False
                        if add_flag or ignore_duplicate is False:
                            setattr(new_obj, attr_name, np.dstack([getattr(new_obj, attr_name), model_attr]))
                else:
                    model_attr = other_attr
                    add_flag = True
                    if len(self_attr.shape) > 2:
                        for k in range(self_attr.shape[2]):
                            if (self_attr[:, :, k] == model_attr).all():  # type: ignore
                                add_flag = False
                    else:
                        if (self_attr == model_attr).all():  # type: ignore
                            add_flag = False
                    if add_flag or ignore_duplicate is False:
                        setattr(new_obj, attr_name, np.dstack([getattr(new_obj, attr_name), model_attr]))
        new_obj._check_dimensions()
        return new_obj

    def _expand_attr(self):
        """ Expand stacked attributes in an object """
        stack_numbers = self._get_stack_numbers()
        update_idx = [x for x in range(len(stack_numbers)) if (stack_numbers > 1)[x]]
        new_stack_number = int(stack_numbers[update_idx].prod())
        # expand with permutation of attributes to form stacked models
        last_attr_multiplier = 1
        for j in range(len(update_idx)):
            attr_name = self.stackable[update_idx[j]]
            current_attr = getattr(self, attr_name)
            rep_num = int(new_stack_number // last_attr_multiplier // stack_numbers[update_idx[j]])
            temp_attr = np.dstack([np.dstack([current_attr[:, :, n] for _ in range(rep_num)])
                                   for n in range(current_attr.shape[2])])
            setattr(self, attr_name, np.dstack([temp_attr for _ in range(last_attr_multiplier)]))
            last_attr_multiplier = int(last_attr_multiplier * stack_numbers[update_idx[j]])
        self._check_dimensions()
        self._check_model_stack()

    def _auto_populate_components(self, type_str):
        """ Auto-populate components with a given type of components """
        assert isinstance(type_str, str), 'components should be a string in this method.'
        if type_str == 'Osc':  # auto-populate with oscillator components
            self._auto_populate_osc_components()
        elif type_str == 'Arn':
            self._auto_populate_arn_components()
        elif type_str == 'Gen':
            self._auto_populate_gen_components()
        else:
            raise ValueError('Specified component type is not valid.')

    def _auto_populate_osc_components(self):
        """ Initialize all components to be Matsuda oscillators during constructor """
        try:  # this has to stay here to avoid circular import before class definitions
            from .osc import OscillatorModel
        except (ImportError, ModuleNotFoundError):
            from osc import OscillatorModel

        self._check_dimensions((0,))  # fill in the nstate attribute
        assert (self.nstate % 2) == 0, 'Cannot default to Osc components with odd nstate.'
        self.ncomp = self.nstate // 2
        if self.ncomp > 0:
            self.components = [OscillatorModel() for _ in range(self.ncomp)]
            self.comp_nstates = [x.default_G.shape[1] for x in self.components]
        else:
            self.components = None
            self.comp_nstates = [0]

    def _auto_populate_arn_components(self):
        """ Initialize all components to be autoregressive models during constructor """
        try:  # this has to stay here to avoid circular import before class definitions
            from .arn import AutoRegModel
        except (ImportError, ModuleNotFoundError):
            from arn import AutoRegModel

        # F and Q parameters must be specified
        assert self.F is not None, 'F parameter is not available to form Arn components.'
        assert self.Q is not None, 'Q parameter is not available to form Arn components.'
        assert np.all(self.Q(~np.eye(self.Q.shape[0], dtype=bool)) == 0),\
            'Found non-zero non-diagonal elements when trying to auto-populate Arn components.'

        # Guess how many Arn components are contained in the model parameters
        Q_diag = np.diag(self.Q)
        self.comp_nstates = np.diff(np.append(np.nonzero(Q_diag), len(Q_diag)))
        self.ncomp = len(self.comp_nstates)

        # Create the autoregressive model components
        comp_list = []
        for n in range(len(self.comp_nstates)):
            start_idx = sum(self.comp_nstates[:n])
            end_idx = sum(self.comp_nstates[:n+1])
            coeff = self.F[start_idx, start_idx:end_idx]
            sigma2 = self.Q[start_idx, start_idx]
            mu0 = self.mu0[start_idx:end_idx] if self.mu0 is not None else None
            Q0 = self.Q[start_idx, start_idx] if self.Q0 is not None else None
            comp_list.append(AutoRegModel(coeff=coeff, sigma2=sigma2, mu0=mu0, Q0=Q0))

        self.components = comp_list

    def _auto_populate_gen_components(self):
        """ Initialize with a single general state-space model during constructor """
        try:  # this has to stay here to avoid circular import before class definitions
            from .gen import GeneralSSModel
        except (ImportError, ModuleNotFoundError):
            from gen import GeneralSSModel

        self.ncomp = 1
        self.components = [GeneralSSModel(F=self.F, Q=self.Q, mu0=self.mu0, Q0=self.Q0)]
        self.comp_nstates = [self.components[0].nstate]

    def _initialize_from_components(self, components):
        """ Initialize the StateSpaceModel instance from given components during constructor """
        assert not isinstance(components, str), 'components should not be a string in this method.'

        # Form a list of single components without nesting
        components = list(components) if isinstance(components, Iterable) else [components]
        comp_list = []
        for n in range(len(components)):
            current_component = components[n]
            if current_component.ncomp <= 1:
                comp_list.append(current_component)
            else:
                assert type(current_component.components) is list, 'Invalid components attribute.'
                assert (np.asarray([x.ncomp for x in current_component.components]) <= 1).all(), \
                    'Components should not have nested multiple components in them.'
                comp_list += current_component.components

        self.ncomp = len(comp_list)
        self.components = comp_list
        self.comp_nstates = [x.default_G.shape[1] for x in self.components]

        # Use the Ssm.concat_() method to concatenate parameters from components
        concat_model = self.components[0]
        for n in range(1, self.ncomp):
            concat_model = StateSpaceModel.concat_(concat_model, self.components[n], skip_components=True)

        # Fill in model parameters if available from components
        self._setattr_when_not_none(concat_model, ('F', 'Q', 'mu0', 'Q0', 'R', 'Fs'))

        # Fill in observed data if available from components
        self._check_observed_data(concat_model)
        self.y = self._return_not_none(self.y, concat_model.y)

    def _setattr_when_not_none(self, other, attrs):
        """ Update an instance attribute when the other model has non-empty values """
        attrs = tuple(attrs) if type(attrs) is not tuple else attrs
        for attr_name in attrs:
            self_attr = getattr(self, attr_name)
            other_attr = getattr(other, attr_name)
            if other_attr is not None:
                if self_attr is None:
                    setattr(self, attr_name, other_attr)
                else:
                    assert np.all(self_attr == other_attr), \
                        'Incompatible attribute ' + attr_name + ' when initializing from components.'

    def _initialize_observation_matrix(self, G):
        """ Initialize the observation matrix instance attribute during constructor """
        if G is None:
            if self.ncomp > 0:
                nchannel = self.y.shape[0] if self.y is not None else 1  # at least one observation channel
                self.G = np.tile(self._must_be_row(self._process_constructor_input(
                    np.hstack([x.default_G for x in self.components]))), (nchannel, 1))
            else:
                self.G = None
        else:
            self.G = self._must_be_row(self._process_constructor_input(G))

    def stack_attr(self, attr_name, new_attr):
        """ Stack new_attr into the third axis of an instance attribute """
        if attr_name == 'components' or attr_name == 'y' or attr_name == 'Fs':
            raise AttributeError(attr_name + " shouldn't be stacked.")
        # using _concat_attr can automatically ignore duplicated attributes
        temp_obj = self._concat_attr(StateSpaceModel(**{attr_name: new_attr}), attrs=attr_name)
        setattr(self, attr_name, getattr(temp_obj, attr_name))
        self._check_dimensions()
        try:
            self._check_model_stack()
        except AssertionError:
            print("WARNING: nmodel cannot be updated automatically due to ambiguous model stack numbers.")

    def stack_to_array(self):
        """
        Unstack attributes into an array of objects,
        which is duck-typing-equivalent with a list
        """
        self._check_model_stack()
        stack_numbers = self._get_stack_numbers()
        update_idx = [x for x in range(len(stack_numbers)) if (stack_numbers > 1)[x]]
        ssm_array = np.empty(self.nmodel, dtype=StateSpaceModel)  # mutable array
        for ii in range(self.nmodel):
            temp_obj = deepcopy(self)
            for j in range(len(update_idx)):
                temp_attr = deepcopy(getattr(self, self.stackable[update_idx[j]]))
                setattr(temp_obj, self.stackable[update_idx[j]], temp_attr[:, :, ii])
            temp_obj._check_dimensions()
            temp_obj._check_model_stack()
            ssm_array[ii] = temp_obj
        return ssm_array

    def copy(self, drop_y=False):
        """ Make a copy of the StateSpaceModel Object """
        self_copy = deepcopy(self)
        if drop_y:
            self_copy.y = None
        return self_copy

    def append(self, other):
        """
        Append an object to an existing StateSpaceModel object,
        by calling concat_() and copying over attributes
        """
        temp_obj = self.concat_(other)
        attr_dict = self.__dict__
        for attr in attr_dict.keys():
            setattr(self, attr, getattr(temp_obj, attr))

    def rappend(self, other):
        """
        Append an existing StateSpaceModel object to an object,
        by calling concat_() and copying over attributes
        """
        temp_obj = other.concat_(self)
        attr_dict = self.__dict__
        for attr in attr_dict.keys():
            setattr(self, attr, getattr(temp_obj, attr))

    def concat_(self, other, skip_components=False):
        """
        Join two StateSpaceModel objects together by concatenating the
        components, and return a new object with the new total number
        of components. The mutable attribute components has maintained
        memory addresses
        """
        # If both have observed data, they should be identical
        self._check_observed_data(other)
        y = self._return_not_none(self.y, other.y)
        Fs = self._return_not_none(self.Fs, other.Fs)
        if self.R is not None and other.R is not None:
            assert (self.R == other.R).all(), 'Cannot concatenate two objects with conflicting R.'  # type: ignore
            R = self.R
        else:
            R = self._return_not_none(self.R, other.R)

        # Set up the new components attribute to maintain memory addresses
        new_comp = [None] * (self.ncomp + other.ncomp)
        if len(new_comp) == 0 or skip_components:
            new_comp = None
        else:
            new_comp[:self.ncomp] = [None] * self.ncomp if self.components is None else self.components
            new_comp[self.ncomp:] = [None] * other.ncomp if other.components is None else other.components

        # Configure the rest of attributes that are immutable
        # F
        if self.F is None and other.F is None:
            F = None
        elif self.F is None:
            F = block_diag(np.empty((self.nstate, self.nstate), dtype=object), other.F)
        elif other.F is None:
            F = block_diag(self.F, np.empty((other.nstate, other.nstate), dtype=object))
        else:
            F = block_diag(self.F, other.F)

        # Q
        if self.Q is None and other.Q is None:
            Q = None
        elif self.Q is None:
            Q = block_diag(np.empty((self.nstate, self.nstate), dtype=object), other.Q)
        elif other.Q is None:
            Q = block_diag(self.Q, np.empty((other.nstate, other.nstate), dtype=object))
        else:
            Q = block_diag(self.Q, other.Q)

        # mu0
        if self.mu0 is None and other.mu0 is None:
            mu0 = None
        elif self.mu0 is None:
            mu0 = np.vstack([np.zeros((self.nstate, 1), dtype=np.float64), other.mu0])
        elif other.mu0 is None:
            mu0 = np.vstack([self.mu0, np.zeros((other.nstate, 1), dtype=np.float64)])
        else:
            mu0 = np.vstack([self.mu0, other.mu0])

        # Q0
        if self.Q0 is None and other.Q0 is None:
            Q0 = None
        elif self.Q0 is None:
            Q0 = block_diag(self.Q, other.Q0) if self.Q is not None else \
                block_diag(np.empty((self.nstate, self.nstate), dtype=object), other.Q0)
        elif other.mu0 is None:
            Q0 = block_diag(self.Q0, other.Q) if other.Q is not None else \
                block_diag(self.Q0, np.empty((other.nstate, other.nstate), dtype=object))
        else:
            Q0 = block_diag(self.Q0, other.Q0)

        return StateSpaceModel(components=new_comp, F=F, Q=Q, mu0=mu0, Q0=Q0, R=R, y=y, Fs=Fs)  # fill G automatically

    def remove_component(self, comp_idx):
        """ Remove a component from the Ssm object """
        start_idx = sum(self.comp_nstates[:comp_idx])
        end_idx = sum(self.comp_nstates[:comp_idx+1])
        slice_idx = np.s_[start_idx:end_idx]
        self.nstate -= self.comp_nstates[comp_idx]
        self.ncomp -= 1
        _ = self.comp_nstates.pop(comp_idx)
        _ = self.components.pop(comp_idx)
        if self.F is not None:
            self.F = np.delete(self.F, slice_idx, axis=0)
            self.F = np.delete(self.F, slice_idx, axis=1)
        if self.Q is not None:
            self.Q = np.delete(self.Q, slice_idx, axis=0)
            self.Q = np.delete(self.Q, slice_idx, axis=1)
        if self.mu0 is not None:
            self.mu0 = np.delete(self.mu0, slice_idx, axis=0)
        if self.Q0 is not None:
            self.Q0 = np.delete(self.Q0, slice_idx, axis=0)
            self.Q0 = np.delete(self.Q0, slice_idx, axis=1)
        if self.G is not None:
            self.G = np.delete(self.G, slice_idx, axis=1)

        # Set attributes to default values if nothing left
        self.comp_nstates = 0 if len(self.comp_nstates) == 0 else self.comp_nstates
        self.components = None if len(self.components) == 0 else self.components
        self.F = None if len(self.F) == 0 else self.F
        self.Q = None if len(self.Q) == 0 else self.Q
        self.mu0 = None if len(self.mu0) == 0 else self.mu0
        self.Q0 = None if len(self.Q0) == 0 else self.Q0

    def fill_components(self, empty_comp=None, deep_copy=True):
        """
        Fill components with attribute content with deepcopy(),
        observed data y is copied using memory address to save space,
        so it remains mutable (be careful when you slice-alter y)
        """
        # Create an empty component instance
        empty_comp = StateSpaceModel() if empty_comp is None else empty_comp

        # Save the current components in case we need to revert to unfilled status
        components_prefill = deepcopy(self.components)

        for ii in range(self.ncomp):
            current_component: StateSpaceModel = self.components[ii]  # typehint to superclass to be flexible
            start_idx = sum(self.comp_nstates[:ii])
            end_idx = sum(self.comp_nstates[:ii+1])

            # Provide an empty component in the components attribute
            setattr(current_component, 'components', [deepcopy(empty_comp)])

            if getattr(self, 'F') is not None:
                if deep_copy:
                    F = deepcopy(getattr(self, 'F')[start_idx:end_idx, start_idx:end_idx])
                else:
                    F = getattr(self, 'F')[start_idx:end_idx, start_idx:end_idx]
                setattr(current_component, 'F', F)
            if getattr(self, 'Q') is not None:
                if deep_copy:
                    Q = deepcopy(getattr(self, 'Q')[start_idx:end_idx, start_idx:end_idx])
                else:
                    Q = getattr(self, 'Q')[start_idx:end_idx, start_idx:end_idx]
                setattr(current_component, 'Q', Q)
            if getattr(self, 'mu0') is not None:
                if deep_copy:
                    mu0 = deepcopy(getattr(self, 'mu0')[start_idx:end_idx])
                else:
                    mu0 = getattr(self, 'mu0')[start_idx:end_idx]
                setattr(current_component, 'mu0', mu0)
            if getattr(self, 'Q0') is not None:
                if deep_copy:
                    Q0 = deepcopy(getattr(self, 'Q0')[start_idx:end_idx, start_idx:end_idx])
                else:
                    Q0 = getattr(self, 'Q0')[start_idx:end_idx, start_idx:end_idx]
                setattr(current_component, 'Q0', Q0)
            if getattr(self, 'G') is not None:
                if deep_copy:
                    G = deepcopy(getattr(self, 'G')[:, start_idx:end_idx])
                else:
                    G = getattr(self, 'G')[:, start_idx:end_idx]
                setattr(current_component, 'G', G)

            # R remains unchanged across components
            if deep_copy:
                setattr(current_component, 'R', deepcopy(getattr(self, 'R')))
            else:
                setattr(current_component, 'R', getattr(self, 'R'))

            setattr(current_component, 'y', getattr(self, 'y'))  # point to the same memory address
            setattr(current_component, 'Fs', getattr(self, 'Fs'))  # Fs should have immutable datatype

            # Use inherited check methods in component subclasses to update dimension attributes
            current_component._check_dimensions()
            current_component._check_model_stack()
            current_component.ncomp = 1

        return components_prefill

    def unfill_components(self, components_prefill):
        """ Unfill components to a prefill state """
        # Save the current components in case we need to revert to unfilled status
        assert len(self.components) == len(components_prefill), 'Different numbers of components during unfill.'
        for ii in range(self.ncomp):
            attr_dict = components_prefill[ii].__dict__
            for attr in attr_dict.keys():
                setattr(self.components[ii], attr, getattr(components_prefill[ii], attr))

    def get_default_q(self, components=None, E=None):
        """
        Get the default structure of state noise covariance
        matrix Q in the Q_basis block diagonal form
        """
        components = self.components if components is None else components
        default_Q = block_diag(*[x.get_default_q(components=x, E=E) for x in components])
        return default_Q

    @staticmethod
    def _process_constructor_input(a):
        """
        Process inputs to the constructor of Ssm class.
        third axis is reserved for stacking different models instantiated with tuple inputs
        """
        if a is None:
            return a
        elif type(a) is tuple:
            return np.asanyarray(np.dstack(a), dtype=np.float64)
        else:
            a = deepcopy(np.asanyarray(a, dtype=np.float64))  # break the link to input data memory address
            if len(a.shape) == 0:  # always promote to (r,c) 2D arrays
                return a[None, None]
            elif len(a.shape) == 1:
                return a[:, None]
            else:
                return a

    @staticmethod
    def _must_be_row(x):
        """ Ensure input vector is a row vector """
        if x is None:
            return x
        elif x.shape[1] == 1 and len(x.shape) < 3:
            return x.T
        else:
            return x

    @staticmethod
    def _return_not_none(a, b):
        """ Return the first not-None value """
        if a is None:
            return b
        else:
            return a

    @staticmethod
    def setup_array(ssm_array, y=None):
        """ Prepare an array of Ssm objects for other signal processing methods """
        # Convert to an array if the input is a single Ssm instance
        if hasattr(ssm_array, 'nmodel'):  # suggests that this is a single somata class object
            ssm_array = ssm_array.stack_to_array()

        # Fill the models with the same observed data if provided
        if y is not None:
            for m in range(len(ssm_array)):
                ssm_array[m].y = y

        # Verify that observed data are identical across the array
        first_ssm: StateSpaceModel = ssm_array[0]
        for ii in range(1, len(ssm_array)):
            first_ssm._check_observed_data(ssm_array[ii])

        # Array dimensions
        K = len(ssm_array)  # number of models
        T = first_ssm.ntime  # number of time points

        return ssm_array, K, T

    # Kalman filtering and smoothing methods (E step)
    def kalman_filt_smooth(self, y=None, R_weights=None, return_dict=False, EM=False, skip_interp=True, seterr=None):
        """ Wrapper method for classical kalman filtering and smoothing """
        if seterr is not None:  # apply np.seterr() for parallel processes
            old_settings = np.seterr(**seterr)

        y = self.y if y is None else y
        x_t_n, P_t_n, P_t_tmin1_n, logL, x_t_t, P_t_t, K_t, x_t_tmin1, P_t_tmin1, fy_t_interp = kalman(
            F=self.F, Q=self.Q, mu0=self.mu0, Q0=self.Q0, G=self.G, R=self.R, y=y, R_weights=R_weights,
            skip_interp=skip_interp)

        if seterr is not None:
            # noinspection PyUnboundLocalVariable
            np.seterr(**old_settings)

        if EM:  # return minimally necessary variables for EM algorithm
            return dict_of(x_t_n, P_t_n, P_t_tmin1_n, logL)
        else:
            if return_dict:
                return dict_of(x_t_n, P_t_n, P_t_tmin1_n, logL, x_t_t, P_t_t, K_t, x_t_tmin1, P_t_tmin1, fy_t_interp)
            else:
                return x_t_n, P_t_n, P_t_tmin1_n, logL, x_t_t, P_t_t, K_t, x_t_tmin1, P_t_tmin1, fy_t_interp

    def dejong_filt_smooth(self, y=None, R_weights=None, return_dict=False, EM=False, skip_interp=True, seterr=None):
        """ Wrapper method for De Jong version kalman filtering and smoothing """
        if seterr is not None:  # apply np.seterr() for parallel processes
            old_settings = np.seterr(**seterr)

        y = self.y if y is None else y
        x_t_n, P_t_n, P_t_tmin1_n, logL, x_t_t, P_t_t, K_t, x_t_tmin1, P_t_tmin1, fy_t_interp = djkalman(
            F=self.F, Q=self.Q, mu0=self.mu0, Q0=self.Q0, G=self.G, R=self.R, y=y, R_weights=R_weights,
            skip_interp=skip_interp)

        if seterr is not None:
            # noinspection PyUnboundLocalVariable
            np.seterr(**old_settings)

        if EM:  # return minimally necessary variables for EM algorithm
            return dict_of(x_t_n, P_t_n, P_t_tmin1_n, logL)
        else:
            if return_dict:
                return dict_of(x_t_n, P_t_n, P_t_tmin1_n, logL, x_t_t, P_t_t, K_t, x_t_tmin1, P_t_tmin1, fy_t_interp)
            else:
                return x_t_n, P_t_n, P_t_tmin1_n, logL, x_t_t, P_t_t, K_t, x_t_tmin1, P_t_tmin1, fy_t_interp

    @staticmethod
    def par_kalman(ssm_array, y=None, method='kalman', R_weights=None, skip_interp=True, return_dict=False):
        """
        Parallel run kalman filtering and smoothing on
        an array of StateSpaceModel objects
        """
        ssm_array, K, T = StateSpaceModel.setup_array(ssm_array, y=y)

        if method == 'kalman':
            kalman_func = getattr(StateSpaceModel, 'kalman_filt_smooth')
        elif method == 'dejong':
            kalman_func = getattr(StateSpaceModel, 'dejong_filt_smooth')
        else:
            raise ValueError('Specified method is invalid for parallel kalman calls.')

        if R_weights is None:
            R_weights = np.tile(np.array([None]), K)
        else:
            R_weights = tuple([R_weights[m, :] for m in range(K)])

        # Run kalman filtering and smoothing on parallel processes
        n_jobs = max(cpu_count()-1, 1)
        results = Parallel(n_jobs=n_jobs)(delayed(kalman_func)(model, y, weights, False, False,
                                                               skip_interp, np.geterr())
                                          for model, weights in zip(ssm_array, R_weights))

        # Unpack results into separate variables to return
        (x_t_n_all, P_t_n_all, P_t_tmin1_n_all,
         x_t_t_all, P_t_t_all, K_t_all, x_t_tmin1_all, P_t_tmin1_all) = tuple([[None]*K for _ in range(8)])
        logL_all = np.zeros((K, T), dtype=np.float64)
        fy_t_interp_all = np.zeros((K, T), dtype=np.float64)
        for m in range(K):
            (x_t_n_all[m], P_t_n_all[m], P_t_tmin1_n_all[m],
             logL_all[m, :], x_t_t_all[m], P_t_t_all[m], K_t_all[m],
             x_t_tmin1_all[m], P_t_tmin1_all[m], fy_t_interp_all[m, :]) = results[m]

        if return_dict:
            return dict_of(x_t_n_all, P_t_n_all, P_t_tmin1_n_all, logL_all,
                           x_t_t_all, P_t_t_all, K_t_all, x_t_tmin1_all, P_t_tmin1_all, fy_t_interp_all)
        else:
            return x_t_n_all, P_t_n_all, P_t_tmin1_n_all, logL_all, \
                x_t_t_all, P_t_t_all, K_t_all, x_t_tmin1_all, P_t_tmin1_all, fy_t_interp_all

    # Parameter estimation methods (M step)
    # noinspection PyUnusedLocal
    def m_estimate(self, y=None, x_t_n=None, P_t_n=None, P_t_tmin1_n=None, h_t=None, logL=None,
                   priors=None, A=None, B=None, C=None, T=None, force_ABC=False,
                   update_param=('F', 'Q', 'mu0', 'Q0', 'G', 'R'), keep_param=(),
                   return_dict=None):
        """
        Maximum likelihood or Maximum a posteriori estimation to update
        parameters. Ssm class doesn't have explicit state-equation parameter
        update rules, therefore calling the inherited _m_update_<param>
        methods in component subclasses

        Reference:
            Shumway, R. H., & Stoffer, D. S. (1982). An approach to time series
            smoothing and forecasting using the EM algorithm. Journal of time series
            analysis, 3(4), 253-264.

            Ghahramani, Z., & Hinton, G. E. (1996). Parameter estimation for linear
            dynamical systems.

            Ghahramani, Z., & Hinton, G. E. (2000). Variational learning for
            switching state-space models. Neural computation, 12(4), 831-864.

        Inputs:
        :param self: Ssm class instance
        :param y: observed data
        :param x_t_n: smoothed estimates (posterior) of state mean
        :param P_t_n: smoothed estimates (posterior) of state conditional covariance
        :param P_t_tmin1_n: smoothed estimates (posterior) of state lag1 conditional cross-covariance
        :param h_t: responsibility vector (i.e. 1/R_weights in E step)
        :param logL: unused variable, pure syntactic sugar to allow pythonic **d inputs
        :param priors: a list of dictionaries specifying priors for each component, if None -> MLE
        :param A: sum of square terms
        :param B: sum of square terms
        :param C: sum of square terms
        :param T: number of time points in sum of square terms
        :param force_ABC: flag to compute ABC regardless of state-equation parameters
        :param update_param: a tuple of strings for parameters to update
        :param keep_param: a tuple of strings for parameters to keep and not update
        :param return_dict: None -> no return, True -> return dict, False -> return tuple of variables
        """
        # Obtain boolean flags of scopes of updates
        update_FQ, update_mu0Q0G = self._m_estimate_scope(update_param, keep_param)

        # Initialize parameters
        y = self.y if y is None else y
        if priors is None:
            priors = [None] * self.ncomp
        elif type(priors) == dict:
            priors = [priors]  # so that it can be indexed to position 0

        # Attempt to skip sums of squares computation
        if update_FQ or force_ABC:
            A = self._m_compute_ss('A', x_t_n, P_t_n=P_t_n) if A is None else A
            B = self._m_compute_ss('B', x_t_n, P_t_tmin1_n=P_t_tmin1_n) if B is None else B
            C = self._m_compute_ss('C', x_t_n, P_t_n=P_t_n) if C is None else C
            T = x_t_n.shape[1] - 1 if T is None else T

        # Initialize the responsibility vector h_t
        if h_t is None:
            h_t_length = T if T is not None else y.shape[1]
            h_t = np.ones(h_t_length, dtype=np.float64)  # default responsibility is 1 for all timepoints

        # Update parameters for each independent component -- F, Q, mu0, Q0, G (component specific priors)
        if update_FQ or update_mu0Q0G:
            for ii in range(self.ncomp):  # iterate through components
                current_component: StateSpaceModel = self.components[ii]  # typehint to superclass to be flexible
                start_idx = sum(self.comp_nstates[:ii])
                end_idx = sum(self.comp_nstates[:ii+1])
                if update_FQ:
                    A_tmp = A[start_idx:end_idx, start_idx:end_idx]
                    B_tmp = B[start_idx:end_idx, start_idx:end_idx]
                    C_tmp = C[start_idx:end_idx, start_idx:end_idx]
                else:
                    A_tmp, B_tmp, C_tmp = (None, None, None)

                # Call the _m_update_<param> methods specific to the component subclass
                if 'F' in update_param and 'F' not in keep_param:
                    self.F[start_idx:end_idx, start_idx:end_idx] = \
                        current_component._m_update_f(A=A_tmp, B=B_tmp, C=C_tmp, priors=priors[ii])

                if 'Q' in update_param and 'Q' not in keep_param:
                    self.Q[start_idx:end_idx, start_idx:end_idx] = \
                        current_component._m_update_q(A=A_tmp, B=B_tmp, C=C_tmp, T=T,
                                                      F=self.F[start_idx:end_idx, start_idx:end_idx],
                                                      priors=priors[ii])

                if 'mu0' in update_param and 'mu0' not in keep_param:
                    self.mu0[start_idx:end_idx, 0] = \
                        current_component._m_update_mu0(x_0_n=x_t_n[start_idx:end_idx, 0])[:, 0]

                if 'Q0' in update_param and 'Q0' not in keep_param:
                    self.Q0[start_idx:end_idx, start_idx:end_idx] = \
                        current_component._m_update_q0(x_0_n=x_t_n[start_idx:end_idx, 0],
                                                       P_0_n=P_t_n[start_idx:end_idx, start_idx:end_idx, 0],
                                                       mu0=self.mu0[start_idx:end_idx, 0][:, None])

                if 'G' in update_param and 'G' not in keep_param:
                    comp_G = current_component._m_update_g(y=y, x_t_n=x_t_n[start_idx:end_idx, :],
                                                           P_t_n=P_t_n[start_idx:end_idx, start_idx:end_idx, :],
                                                           h_t=h_t)
                    self.G[start_idx:end_idx, start_idx:end_idx] = \
                        comp_G if comp_G is not None else self.G[start_idx:end_idx, start_idx:end_idx]

        # Update observation noise covariance -- R
        if 'R' in update_param and 'R' not in keep_param:
            R_ss = self._m_update_r(y=y, x_t_n=x_t_n, P_t_n=P_t_n, h_t=h_t, G=self.G, priors=priors[0])
        else:
            R_ss = self._m_update_r(y=y, x_t_n=x_t_n, P_t_n=P_t_n, h_t=h_t, G=self.G, keep_R=True)

        if return_dict is None:
            pass
        elif return_dict:
            return dict_of(self.F, self.Q, self.mu0, self.Q0, self.G, self.R, R_ss, A, B, C)
        else:
            return self.F, self.Q, self.mu0, self.Q0, self.G, self.R, R_ss, A, B, C

    def update_comp_param(self):
        """ Update component specific parameters, override in subclasses """
        return

    def e_step(self, y=None, logL_list=None, **kwargs):
        """ Exposed E step method for run_em() """
        e_results: dict = self.dejong_filt_smooth(y=y, EM=True, **kwargs)
        e_logL = e_results['logL'].sum()
        stop_var = e_logL - logL_list[-1]  # logL should monotonically increase
        logL_list.append(e_logL)
        return e_results, stop_var

    def m_step(self, y=None, **kwargs):
        """ Exposed M step method for run_em() """
        self.m_estimate(y=y, **kwargs)
        return

    def initialize_priors(self):
        """ Initialize priors for each component in the object """
        assert self.components is not None, 'Cannot initialize priors when components is None.'
        priors = [x.initialize_priors() for x in self.components]
        return priors

    def _initialize_priors_recursive_list(self, prior_value):
        """ Expand a prior value to the length of components for the recursive case """
        if prior_value is None:
            prior_value = [None] * self.ncomp
        elif isinstance(prior_value, str) or not isinstance(prior_value, Iterable) or len(prior_value) == 1:
            prior_value = [prior_value] * self.ncomp
        else:
            assert len(prior_value) == self.ncomp, 'Different lengths of specified prior value and components.'
        return prior_value

    def _m_update_r(self, y=None, x_t_n=None, P_t_n=None, h_t=None, G=None, R_ss=None, T=None, priors=None,
                    keep_R=False):
        """ Update observation noise covariance """
        if R_ss is None:
            assert len(h_t) == y.shape[1], 'Different lengths of h_t and y. Cannot proceed with _m_update_r().'
            R_ss = (h_t * (y - G @ x_t_n[:, 1:])) @ (y - G @ x_t_n[:, 1:]).T + \
                G @ (h_t * P_t_n[:, :, 1:]).sum(axis=2) @ G.T

        if not keep_R:  # proceed to update self.R
            T = h_t.sum() if T is None else T

            if self._m_update_if_mle('R_sigma2', priors):
                # MLE
                self.R = R_ss / T
            else:
                # MAP with inverse gamma prior
                R_init = priors['R_sigma2']
                R_hp = priors['R_hyperparameter'] if 'R_hyperparameter' in priors else 0.1
                alpha = T * R_hp / 2  # scales with data length T according to the hyperparameter
                beta = R_init * (alpha + 1)  # setting the mode of inverse gamma prior to be R_init
                self.R = (beta + R_ss/2) / (alpha + T/2 + 1)  # new R is the mode of inverse gamma posterior

        return R_ss

    @staticmethod
    def _m_update_if_mle(params, priors):
        """ Check whether doing MLE during _m_update_<param> methods """
        if priors is None:
            return True

        if isinstance(params, str) or not isinstance(params, Iterable):
            params = [params]

        MLE_flag = np.zeros(len(params), dtype=bool)
        for param, idx in zip(params, range(len(params))):
            if param not in priors:
                MLE_flag[idx] = True
            elif isinstance(priors[param], str):
                if priors[param].upper() == 'MLE':
                    MLE_flag[idx] = True

        return MLE_flag.all()

    # noinspection PyUnusedLocal
    @staticmethod
    def _m_update_f(A=None, B=None, C=None, priors=None):
        pass

    # noinspection PyUnusedLocal
    @staticmethod
    def _m_update_q(A=None, B=None, C=None, T=None, F=None, priors=None):
        pass

    @staticmethod
    def _m_update_mu0(x_0_n=None):
        """ Update initial state mean -- mu0 """
        mu0 = x_0_n[:, None].copy()
        return mu0

    @staticmethod
    def _m_update_q0(x_0_n=None, P_0_n=None, mu0=None):
        """ Update initial state covariance -- Q0 """
        Q0 = P_0_n + x_0_n[:, None] @ x_0_n[:, None].T \
            - x_0_n[:, None] @ mu0.T - mu0 @ x_0_n[:, None].T + mu0 @ mu0.T
        return Q0

    @staticmethod
    def _m_update_g(y=None, x_t_n=None, P_t_n=None, h_t=None):
        """ Update observation matrix -- G """
        assert len(h_t) == y.shape[1], 'Different lengths of h_t and y. Cannot proceed with _m_update_g().'
        P = (h_t * P_t_n[:, :, 1:]).sum(axis=2) + (h_t * x_t_n[:, 1:]) @ x_t_n[:, 1:].T
        approach = 'svd' if P.shape[0] >= 5 else 'gaussian'
        G = (h_t * y) @ x_t_n[:, 1:].T @ inverse(P, approach=approach)
        return G

    @staticmethod
    def _m_estimate_scope(update_param, keep_param):
        """ Sort out the scopes of updates in m_estimate() """
        if 'F' not in update_param and 'Q' not in update_param:
            update_FQ = False
        elif 'F' in keep_param and 'Q' in keep_param:
            update_FQ = False
        else:
            update_FQ = True

        if 'mu0' not in update_param and 'Q0' not in update_param and 'G' not in update_param:
            update_mu0Q0G = False
        elif 'mu0' in keep_param and 'Q0' in keep_param and 'G' in keep_param:
            update_mu0Q0G = False
        else:
            update_mu0Q0G = True

        return update_FQ, update_mu0Q0G

    @staticmethod
    def _m_compute_ss(ss_term, x_t_n, P_t_n=None, P_t_tmin1_n=None):
        """
        Compute the sum of squares terms for m_estimate().
        Definitions of A,B,C follow the notations in equations (9,10,11)
        of S&S 1982, and these terms also correspond to maximizing <H>_Q
        in G&H 2000.

        Reference:
            Shumway, R. H., & Stoffer, D. S. (1982). An approach to time series
            smoothing and forecasting using the EM algorithm. Journal of time series
            analysis, 3(4), 253-264.

            Ghahramani, Z., & Hinton, G. E. (2000). Variational learning for
            switching state-space models. Neural computation, 12(4), 831-864.
        """
        if ss_term == 'A':
            return P_t_n[:, :, :-1].sum(axis=2) + x_t_n[:, :-1] @ x_t_n[:, :-1].T
        elif ss_term == 'B':
            return P_t_tmin1_n[:, :, 1:].sum(axis=2) + x_t_n[:, 1:] @ x_t_n[:, :-1].T
        elif ss_term == 'C':
            return P_t_n[:, :, 1:].sum(axis=2) + x_t_n[:, 1:] @ x_t_n[:, 1:].T
        else:
            raise ValueError('Invalid ss_term is specified.')
