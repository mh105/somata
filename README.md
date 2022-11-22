# somata

Github: https://github.com/mh105/somata

**State-space Oscillator Modeling And Time-series Analysis (SOMATA)** is a Python library for state-space neural signal
processing algorithms developed in the [Purdon Lab](https://purdonlab.mgh.harvard.edu).
Basic state-space models are introduced as class objects for flexible manipulations.
Classical exact and approximate inference algorithms are implemented and interfaced as class methods.
Advanced neural oscillator modeling techniques are brought together to work synergistically.

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/mh105/pot/commits/master)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![License: BSD 3-Clause Clear](https://img.shields.io/badge/License-BSD%203--Clause%20Clear-lightgrey.svg)](https://spdx.org/licenses/BSD-3-Clause-Clear.html)
[![DOI](https://zenodo.org/badge/556083594.svg)](https://zenodo.org/badge/latestdoi/556083594)

---

## Table of Contents
* [Requirements](#requirements)
* [Install](#install)
* [Basic state-space models](#basic-state-space-models)
    * [StateSpaceModel](#class-statespacemodel)
    * [OscillatorModel](#class-oscillatormodelstatespacemodel)
    * [AutoRegModel](#class-autoregmodelstatespacemodel)
    * [GeneralSSModel](#class-generalssmodelstatespacemodel)
* [Advanced neural oscillator methods](#advanced-neural-oscillator-methods)
* [Authors](#authors)
* [Citation](#citation)
* [License](#license)

---

## Requirements
somata is built on `numpy` arrays for computations. `joblib` is used for multithreading. Additional dependencies include
`matplotlib`, `scipy`, `tqdm`, `codetiming`, and `sorcery`. Full requirements for each release version will be updated
under `install_requires` in the `setup.cfg` file. If the `environment.yml` file is used to create a new conda
environment, all and only the required packages will be installed.

## Install

```
pip install somata
```

### (For development only)

- ### Fork this repo to personal git
    [How to: GitHub fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo)    

- ### Clone forked copy to local computer
    ``` git clone <forked copy ssh url> ```

- ### Install conda
    [Recommended conda distribution: miniconda](https://docs.conda.io/en/latest/miniconda.html)

    _Apple silicon Mac: choose conda native to the ARM64 architecture instead of Intel x86_

- ### Create a new conda environment
    ``` conda install mamba -n base -c conda-forge ```\
    ``` cd <repo root directory with environment.yml> ```\
    ``` mamba env create -f environment.yml ```\
    ``` conda activate somata ```\
    _You may also install somata in an existing environment by skipping this step._

- ### Install somata as a package in editable mode
    ``` cd <repo root directory with setup.py> ```\
    ``` pip install -e . ```

- ### Configure IDEs to use the conda environment
    [How to: Configure an existing conda environment](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html)

---

## Basic state-space models
Somata, much like a neuron body supported by dendrites, is built on a set of basic state-space models introduced as class objects.

The motivations are to:
- develop a standardized format to store model parameters of state-space equations
- override Python dunder methods so `__repr__` and `__str__` return something useful
- define arithmetic-like operations such as `A + B` and `A * B`
- emulate `numpy.array()` operations including `.append()`
- implement inference algorithms like Kalman filtering and parameter update (m-step) equations as callable class methods

At present, and in the near future, somata will be focused on **time-invariant Gaussian linear dynamical systems**.
This limit on models we consider simplifies basic models to avoid embedded classes such as `transition_model` and
`observation_model`, at the cost of restricting somata to classical algorithms with only some extensions to
Bayesian inference and learning. This is a deliberate choice to allow easier, faster, and cleaner applications of
somata on neural data analysis, instead of to provide a full-fledged statistical inference package.

---

### _class_ StateSpaceModel
```python
somata.StateSpaceModel(components=None, F=None, Q=None, mu0=None, Q0=None, G=None, R=None, y=None, Fs=None)
```
`StateSpaceModel` is the parent class of basic state-space models. The corresponding Gaussian linear dynamical system is:

$$
\mathbf{x}_ {t} = \mathbf{F}\mathbf{x}_{t-1} + \boldsymbol{\eta}_t, \boldsymbol{\eta}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})
$$

$$
\mathbf{y}_ {t} = \mathbf{G}\mathbf{x}_{t} + \boldsymbol{\epsilon}_t, \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R})
$$

$$
\mathbf{x}_0 \sim \mathcal{N}(\mathbf{\mu}_0, \mathbf{Q}_0)
$$

Most of the constructor input arguments correspond to these model parameters, which are stored as instance attributes.
There are two additional arguments: `Fs` and `components`.

`Fs` is the sampling frequency of observed data `y`.

`components` is a list of independent components underlying the hidden states $\mathbf{x}$. The independent components are
assumed to appear in block-diagonal form in the state equation. For example, $\mathbf{x}_t$ might have two independent autoregressive
models (AR) of order 1, and the observation matrix is simply $[1, 1]$ that sums these two components. In this case, `components`
would be a list of two AR1 models. Note that each element of the `components` list should be an instance of one of basic model
class objects. To break the recursion, often the `components` attribute of a component is set to `None`, i.e.,
`components[0].components = None`.

1. `StateSpaceModel.__repr__()`

The double-under method `__repr__()` is overwritten to provide some unique identification info:

```python
>>> s1 = StateSpaceModel()
>>> s1
Ssm(0)<f4c0>
```
where the number inside parenthesis indicates **the number of components** (the `ncomp` attribute) in the model, and the four-digits in `<>` are the last four digits of the memory address of the object instance.

2. `StateSpaceModel.__str__()`

The double-under method `__str__()` is overwritten so `print()` returns useful info:
```python
>>> print(s1)
<Ssm object at 0x102a8f4c0>
 nstate   = 0     ncomp    = 0
 nchannel = 0     ntime    = 0
 nmodel   = 1
 components = None
 F  .shape = None       Q  .shape = None
 mu0.shape = None       Q0 .shape = None
 G  .shape = None       R  .shape = None
 y  .shape = None       Fs = None
```

3. Model _stacking_ in `StateSpaceModel`

In many applications, there are several possible parameter values for a given state-space model structure. Instead of duplicating
the same values in multiple instances, somata uses _stacking_ to store multiple model values in the same object instance. Stackable
model parameters are `F, Q, mu0, Q0, G, R`. For example:

```python
>>> s1 = StateSpaceModel(F=1, Q=2)
>>> s2 = StateSpaceModel(F=2, Q=2)
>>> print(s1)
<Ssm object at 0x11fd7bfa0>
 nstate   = 1     ncomp    = 0
 nchannel = 0     ntime    = 0
 nmodel   = 1
 components = None
 F  .shape = (1, 1)     Q  .shape = (1, 1)
 mu0.shape = None       Q0 .shape = None
 G  .shape = None       R  .shape = None
 y  .shape = None       Fs = None

>>> print(s2)
<Ssm object at 0x102acc130>
 nstate   = 1     ncomp    = 0
 nchannel = 0     ntime    = 0
 nmodel   = 1
 components = None
 F  .shape = (1, 1)     Q  .shape = (1, 1)
 mu0.shape = None       Q0 .shape = None
 G  .shape = None       R  .shape = None
 y  .shape = None       Fs = None

>>> s3 = s1+s2
>>> print(s3)
<Ssm object at 0x102acc280>
 nstate   = 1     ncomp    = 0
 nchannel = 0     ntime    = 0
 nmodel   = 2
 components = None
 F  .shape = (1, 1, 2)  Q  .shape = (1, 1)
 mu0.shape = None       Q0 .shape = None
 G  .shape = None       R  .shape = None
 y  .shape = None       Fs = None
```
Invoking the arithmetic operator `+` stacks the two instances `s1` and `s2` into a new instance, where the third dimension of the
`F` attribute is now `2`, with the two values from `s1` and `s2`. The `nmodel` attribute is also updated to `2`.
```python
>>> s3.F
array([[[1., 2.]]])
```
Notice how the third dimension of the `Q` attribute is still `None`. This is because the `+` operator has a built-in duplication check
such that the identical model parameters will not be stacked. This behavior of `__add__` and `__radd__` generalizes to all model parameters, and it is convenient when bootstrapping or testing different parameter values during neural data analysis. Manual stacking of a particular
model parameter is also possible with `.stack_attr()`.

4. Model _expanding_ in `StateSpaceModel`

Similar to _stacking_, there is a related concept called _expanding_. Expanding a model is useful when we want to permutate multiple model
parameters each with several possible values. For example:

```python
>>> s1 = StateSpaceModel(F=1, Q=3, R=5)
>>> s2 = StateSpaceModel(F=2, Q=4, R=5)
>>> print(s1+s2)
<Ssm object at 0x1059626b0>
 nstate   = 1     ncomp    = 0
 nchannel = 1     ntime    = 0
 nmodel   = 2
 components = None
 F  .shape = (1, 1, 2)  Q  .shape = (1, 1, 2)
 mu0.shape = None       Q0 .shape = None
 G  .shape = None       R  .shape = (1, 1)
 y  .shape = None       Fs = None

>>> s3 = s1*s2
>>> print(s3)
<Ssm object at 0x1059626b0>
 nstate   = 1     ncomp    = 0
 nchannel = 1     ntime    = 0
 nmodel   = 4
 components = None
 F  .shape = (1, 1, 4)  Q  .shape = (1, 1, 4)
 mu0.shape = None       Q0 .shape = None
 G  .shape = None       R  .shape = (1, 1)
 y  .shape = None       Fs = None

>>> s3.F
array([[[1., 1., 2., 2.]]])
>>> s3.Q
array([[[3., 4., 3., 4.]]])
```
Multiplying two `StateSpaceModel` instances with more than one differing model parameters results in expanding these parameters into all possible combinations while keeping other identical attributes intact.

5. Arrays of `StateSpaceModel`

Building on _stacking_ and _expanding_, we can easily form an array of `StateSpaceModel` instances using `.stack_to_array()`:

```python
>>> s_array = s3.stack_to_array()
>>> s_array
array([Ssm(0)<4460>, Ssm(0)<4430>, Ssm(0)<4520>, Ssm(0)<4580>],
      dtype=object)
```

Note that a `StateSpaceModel` array is duck-typing with a Python `list`, which means one can also form a valid array with `[s1, s2]`.

6. `StateSpaceModel.__len__()`

Invoking `len()` returns the number of stacked models:

```python
>>> len(s2)
1
>>> len(s3)
4
```

7. `StateSpaceModel.append()`

Another useful class method on `StateSpaceModel` is `.append()`. As one would expect, appending a model to another results in
combining them in block-diagonal form in the state equation. Compatibility checks happen in the background to make sure no conflict
exists on the respective observation equations and observed data, if any.

```python
>>> s1 = StateSpaceModel(F=1, Q=3, R=5)
>>> s2 = StateSpaceModel(F=2, Q=4, R=5)
>>> s1.append(s2)
>>> print(s1)
<Ssm object at 0x1057cb4c0>
 nstate   = 2     ncomp    = 0
 nchannel = 1     ntime    = 0
 nmodel   = 1
 components = None
 F  .shape = (2, 2)     Q  .shape = (2, 2)
 mu0.shape = None       Q0 .shape = None
 G  .shape = None       R  .shape = (1, 1)
 y  .shape = None       Fs = None

>>> s1.F
array([[1., 0.],
       [0., 2.]])
>>> s1.Q
array([[3., 0.],
       [0., 4.]])
```

Notice that the `nstate` attribute is now updated to `2`, which is different from the `+` operator that updates the `nmodel` attribute to `2`.

8. Inference and learning with `StateSpaceModel`

Two different implementations of Kalman filtering and fixed-interval smoothing are callable class methods:

```python
.kalman_filt_smooth(y=None, R_weights=None, return_dict=False, EM=False, skip_interp=True, seterr=None)

.dejong_filt_smooth(y=None, R_weights=None, return_dict=False, EM=False, skip_interp=True, seterr=None)
```

With an array of `StateSpaceModel`, one can easily run Kalman filtering and smoothing on all array elements with multithreading using the **static** method `.par_kalman()`:

```python
.par_kalman(ssm_array, y=None, method='kalman', R_weights=None, skip_interp=True, return_dict=False)
```

M-step updates are organized using `m_estimate()` that will recurse into each element of the `components` list and use
the appropriate m-step update methods associated with different types of state-space models.

**Below we explain three kinds of basic state-space models currently supported in somata.**

---
### _class_ OscillatorModel(StateSpaceModel)
```python
somata.OscillatorModel(a=None, freq=None, w=None, sigma2=None, add_dc=False,
                       components='Osc', F=None, Q=None, mu0=None, Q0=None, G=None, R=None, y=None, Fs=None)
```
`OscillatorModel` is a child class of `StateSpaceModel`, which means it inherits all the class methods explained above. It has a particular form of the state-space model:

$$
\begin{bmatrix}x_{t, 1}\newline x_{t, 2}\end{bmatrix} = \mathbf{F}\begin{bmatrix}x_{t-1, 1}\newline x_{t-1, 2}\end{bmatrix} + \mathbf{\eta}_t, \mathbf{\eta}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})
$$

$$
\mathbf{y}_ {t} = \mathbf{G}\begin{bmatrix}x_{t, 1}\newline x_{t, 2}\end{bmatrix} + \mathbf{\epsilon}_t, \mathbf{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R})
$$

$$
\begin{bmatrix}x_{0, 1}\newline x_{0, 2}\end{bmatrix} \sim \mathcal{N}(\mathbf{\mu}_0, \mathbf{Q}_0)
$$

$$
\mathbf{F} = a\begin{bmatrix}\cos\omega & -\sin\omega\newline \sin\omega & \cos\omega\end{bmatrix}, \mathbf{Q} = \begin{bmatrix}\sigma^2 & 0\newline 0 & \sigma^2\end{bmatrix}, \mathbf{G} = \begin{bmatrix}1 & 0 \end{bmatrix}
$$

To create a simple oscillator model with rotation frequency $15$ Hz (under $100$ Hz sampling frequency) and damping factor $0.9$:

```python
>>> o1 = OscillatorModel(a=0.9, freq=15, Fs=100)
>>> o1
Osc(1)<81f0>
>>> print(o1)
<Osc object at 0x1058081f0>
 nstate   = 2     ncomp    = 1
 nchannel = 0     ntime    = 0
 nmodel   = 1
 components = [Osc(0)<4b50>]
 F  .shape = (2, 2)     Q  .shape = (2, 2)
 mu0.shape = (2, 1)     Q0 .shape = (2, 2)
 G  .shape = (1, 2)     R  .shape = None
 y  .shape = None       Fs = 100.0 Hz
 damping a = [0.9]
 freq Hz   = [15.]
 sigma2    = [3.]
 obs noise R = None
 dc index  = None
```

Notice the `components` attribute auto-populates with a spaceholder `OscillatorModel` instance, which is different from the `o1` instance
as can be recognized by different memory addresses. State noise variance $\sigma^2$ defaults to $3$ when not specified and can be changed
with the `sigma2` argument to the constructor method.

### _class_ AutoRegModel(StateSpaceModel)
```python
somata.AutoRegModel(coeff=None, sigma2=None,
                    components='Arn', F=None, Q=None, mu0=None, Q0=None, G=None, R=None, y=None, Fs=None)
```
`AutoRegModel` is a child class of `StateSpaceModel`, which means it inherits all the class methods explained above. It has a particular form of the state-space model. For example, an auto-regressive model of order 3 can be expressed as:

$$
\begin{bmatrix}x_{t}\newline x_{t-1}\newline x_{t-2}\end{bmatrix} = \mathbf{F}\begin{bmatrix}x_{t-1}\newline x_{t-2}\newline x_{t-3}\end{bmatrix} + \mathbf{\eta}_t, \mathbf{\eta}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})
$$

$$
\mathbf{y}_ {t} = \mathbf{G}\begin{bmatrix}x_{t}\newline x_{t-1}\newline x_{t-2}\end{bmatrix} + \mathbf{\epsilon}_t, \mathbf{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R})
$$

$$
\begin{bmatrix}x_{0}\newline x_{-1}\newline x_{-2}\end{bmatrix} \sim \mathcal{N}(\mathbf{\mu}_0, \mathbf{Q}_0)
$$

$$
\mathbf{F} = \begin{bmatrix} a_1 & a_2 & a_3 \newline 1 & 0 & 0 \newline 0 & 1 & 0 \end{bmatrix}, \mathbf{Q} = \begin{bmatrix}\sigma^2 & 0 & 0\newline 0 & 0 & 0\newline 0 & 0 & 0\end{bmatrix}, \mathbf{G} = \begin{bmatrix}1 & 0 & 0\end{bmatrix}
$$

To create an AR3 model with parameters $a_1=0.5, a_2=0.3, a_3=0.1$ and $\sigma^2=1$:

```python
>>> a1 = AutoRegModel(coeff=[0.5,0.3,0.1], sigma2=1)
>>> a1
Arn=3<24d0>
>>> print(a1)
<Arn object at 0x1035524d0>
 nstate   = 3     ncomp    = 1
 nchannel = 0     ntime    = 0
 nmodel   = 1
 components = [Arn=3<2680>]
 F  .shape = (3, 3)     Q  .shape = (3, 3)
 mu0.shape = (3, 1)     Q0 .shape = (3, 3)
 G  .shape = (1, 3)     R  .shape = None
 y  .shape = None       Fs = None
 AR order  = [3]
 AR coeff  = ([0.5 0.3 0.1])
 sigma2    = [1.]
```

Note that `__repr__()` is slightly different for `AutoRegModel`, since the key information is not how many components but rather the AR order. We display the order of the auto-regressive model with an `=` sign as shown above instead of showing the number of components in
`()` as for `OscillatorModel` and `StateSpaceModel`.

### _class_ GeneralSSModel(StateSpaceModel)
```python
somata.GeneralSSModel(components='Gen', F=None, Q=None, mu0=None, Q0=None, G=None, R=None, y=None, Fs=None)
```
`GeneralSSModel` is a child class of `StateSpaceModel`, which means it inherits all the class methods explained above. The same general Gaussian linear dynamic system as before is followed:

$$
\mathbf{x}_ t = \mathbf{F}\mathbf{x}_{t-1} + \boldsymbol{\eta}_t, \boldsymbol{\eta}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})
$$

$$
\mathbf{y}_ {t} = \mathbf{G}\mathbf{x}_{t} + \boldsymbol{\epsilon}_t, \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R})
$$

$$
\mathbf{x}_0 \sim \mathcal{N}(\mathbf{\mu}_0, \mathbf{Q}_0)
$$

`GeneralSSModel` is added to somata so that one can perform the most general Gaussian updates for a state-space model without special structures as specified in `OscillatorModel` and `AutoRegModel`. In other words, with non-sparse structures in the model parameters
`F, Q, Q0, G, R`. To create a simple general state-space model:

```python
>>> g1 = GeneralSSModel(F=[[1,2],[3,4]])
>>> g1
Gen(1)<2440>
>>> print(g1)
<Gen object at 0x103552440>
 nstate   = 2     ncomp    = 1
 nchannel = 0     ntime    = 0
 nmodel   = 1
 components = [Gen(0)<2710>]
 F  .shape = (2, 2)     Q  .shape = None
 mu0.shape = None       Q0 .shape = None
 G  .shape = None       R  .shape = None
 y  .shape = None       Fs = None
```

### For more in-depth working examples with the basic models in somata
Look at the demo script [basic_models_demo_01102022.py](examples/basic_models_demo_01102022.py) and execute the code line by line to get familiar with class objects and methods of somata basic models.

---

## Advanced neural oscillator methods
1. [Oscillator Model Learning](#osc)
2. [Iterative Oscillator Algorithm](#ioa)

---
1. ### Oscillator Model Learning <a name="osc"></a> [<img src="https://img.shields.io/badge/Status-Functional-success.svg?logo=Python">](#osc)
---
2. ### Iterative Oscillator Algorithm <a name="ioa"></a> [<img src="https://img.shields.io/badge/Status-Functional-success.svg?logo=Python">](#ioa)

**N.B.:** We recommend downsampling to 120 Hz or less, depending on the oscillations present in your data. Highly oversampled data will make it more difficult to identify oscillatory components, increase the computational time, and could also introduce high frequency noise.

One major goal of this method was to produce an algorithm that required minimal user intervention, if any. We recommend starting with the algorithm as is, but in the case of poor fitting, we suggest the following alterations:
1. If the pole initialized from the one-step prediction is between two oscillations, causing poor fitting of this oscillation as it attempts to explain multiple oscillations, we recommend increasing the order of the AR model used to approximate the OSPE. Increase in increments of two, which will allow additional pairs of complex poles.

2. Conversely to point 1, if the order of the AR model is too high then multiple pairs of roots will be attributed to the same oscillation, diluting the strength needed for each of them and possibly leading to none of them being selected as the strongest root in the iterative process to initialize the next oscillation, even though together they describe the strongest oscillation. This can be identified using the innovations plot with all of the AR roots plotted. In this case we recommend decreasing the AR order in increments of 2, to decrease the number of pairs of complex poles.

3. If the initialization of the additional oscillations describes a single oscillation well, but the fitting of this oscillation attempts to explain multiple oscillations and causes poor fitting, we recommend increasing the concentration hyperparameter in the Von Mises prior. This will increase the weight on the initial frequency and stop the oscillation from shifting to explain other oscillations.

4. If the model does not choose the correct number of oscillations, we recommend looking at all fitted models and selecting the best fitting model based on other selection criteria or using your best judgement. You can also choose a subset of well-fitted oscillations and run the kalman filter to estimate oscillations using those fitted parameters.

5. Note that this algorithm assumes a stationary signal, and therefore stationary parameters. Although the Kalman filtering allows some flexibility in this requirement, enabling the model to work on some time-varying signal, the success of the method depends on the strength and duration of the signal components. The weaker and more brief the time-varying component is, the more poorly the model will capture it, if it does at all. We recommend decreasing the length of your window until you have a more stationary signal.

This algorithm is designed to fit well automatically in most situations, but there will still be some data sets where it does not fit well without intervention.

When using this module, please cite the following [paper](https://www.biorxiv.org/content/10.1101/2022.10.30.514422.abstract):

Beck, A. M., He, M., Gutierrez, R. G., & Purdon, P. L. (2022). An iterative search algorithm to identify oscillatory dynamics in neurophysiological time series. bioRxiv.

---

## Authors
Mingjian He, Proloy Das, Amanda Beck, Patrick Purdon

## Citation
Use different citation styles at: https://doi.org/10.5281/zenodo.7242130

## License
SOMATA is licensed under the [BSD 3-Clause Clear license](https://spdx.org/licenses/BSD-3-Clause-Clear.html). \
Copyright © 2022. All rights reserved.
