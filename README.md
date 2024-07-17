# somata

Github: https://github.com/mh105/somata

**State-space Oscillator Modeling And Time-series Analysis (SOMATA)** is a Python library for state-space neural signal
processing algorithms developed in the [Purdon Lab](https://purdonlab.stanford.edu).
Basic state-space models are introduced as class objects for flexible manipulations.
Classical exact and approximate inference algorithms are implemented and interfaced as class methods.
Advanced neural oscillator modeling techniques are brought together to work synergistically.

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?kill_cache=1)](https://github.com/mh105/pot/commits/master)
[![Version](https://img.shields.io/badge/Version-0.5.2-green?kill_cache=1)](https://github.com/mh105/somata/releases)
[![Last-Update](https://anaconda.org/conda-forge/somata/badges/latest_release_date.svg?kill_cache=1)](https://anaconda.org/conda-forge/somata)
[![License: BSD 3-Clause Clear](https://img.shields.io/badge/License-BSD%203--Clause%20Clear-lightgrey.svg?kill_cache=1)](https://spdx.org/licenses/BSD-3-Clause-Clear.html)
[![DOI](https://zenodo.org/badge/556083594.svg?kill_cache=1)](https://zenodo.org/badge/latestdoi/556083594)

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
[`somata`](https://pypi.org/project/somata/) is built on [`numpy`](https://numpy.org) arrays for computations. [`joblib`](https://joblib.readthedocs.io/en/stable/) is used for multithreading.
Additional dependencies include [`scipy`](https://scipy.org), [`matplotlib`](https://matplotlib.org), [`cmdstanpy`](https://mc-stan.org/cmdstanpy/), and [`spectrum`](https://pyspectrum.readthedocs.io/en/latest/index.html).
The source localization module also requires [`pytorch`](https://pytorch.org) and [`MNE-python`](https://mne.tools/stable/index.html).

- Full package requirements for each release will be updated in the [`requirements-*.txt`](.requirements) files. The [`pyproject.toml`](pyproject.toml) file is specified to dynamically retrieve the metadata of dependencies for [`setuptools`](https://setuptools.pypa.io/en/latest/) during [`pip install`](https://pip.pypa.io/en/stable/cli/pip_install/) to verify that runtime dependent packages of compatible versions have been installed. When [`pip install`](https://pip.pypa.io/en/stable/cli/pip_install/) is used, missing dependencies will be fetched from [Python Package Index (PyPI)](https://pypi.org) and installed.

- For development or installing `somata` into a [new conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands), [`requirements-*.txt`](.requirements) files can also be [passed to `conda create` via the `--file` directive](https://docs.conda.io/projects/conda/en/latest/commands/create.html#named-arguments) to [create a new conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments) with all and only the required packages installed in the new conda environment. When [`conda create`](https://docs.conda.io/projects/conda/en/latest/commands/create.html) or [`conda install`](https://docs.conda.io/projects/conda/en/latest/commands/install.html) is used, missing dependencies will be fetched from [conda channels](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/channels.html#channels) and installed.

### Some notes on package dependency requirements

> The need for specifying dependencies arises in multiple ways for Python.
There are non-negligible complexities due to the existence of different [build backends](https://packaging.python.org/en/latest/tutorials/packaging-projects/#choosing-a-build-backend) as well as different Python dependency management and packaging tools such as [`pdm`](https://pdm-project.org/en/latest/), [`poetry`](https://python-poetry.org), [`pip`](https://pip.pypa.io/en/stable/), [`conda`](https://docs.conda.io/projects/conda/en/stable/), etc.
[One modern standard](https://peps.python.org/pep-0621/) is to [use a declarative config `pyproject.toml` file](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#transitioning-from-setup-py-to-declarative-config) for package building and dependency management, which is becoming popular across build backends and frontends.

In this project, we use [`pyproject.toml`](pyproject.toml) with the [`setuptools`](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html) backend for building `somata`, and we use [`conda`](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#) to manage development environments.
We have deliberately forgone the uses of [`setup.py`](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#setup-py) and [`setup.cfg`](https://setuptools.pypa.io/en/latest/userguide/declarative_config.html) for package building and the use of [`environment.yml`](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for conda environment creation.
Instead, we use a set of [`requirements-*.txt`](.requirements) files. These simple one-liner entries can be passed with `--file` directives into both [`setuptools`](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html#dynamic-metadata) and [`conda`](https://docs.conda.io/projects/conda/en/latest/commands/create.html#named-arguments), allowing single sourcing the [core dependency list](.requirements/requirements-core.txt).
They are also minimalistic in style, so one can easily re-write them into a desired dependency list, such as a [PEP 621](https://peps.python.org/pep-0621/) compliant `dependencies =` key under the [`[project]` table](https://packaging.python.org/en/latest/specifications/pyproject-toml/#dependencies-optional-dependencies) in [`pyproject.toml`](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#dependencies-and-requirements) that can be accepted across tools and backends.

## Install
```
$ pip install somata
```

### conda-forge channel
> There is a known issue with `conda install somata` on _Windows OS_ because the `pytorch` dependency is not available over the `conda-forge` channel for `win-64` builds. One should install `pytorch` first (see [torch requirement](#torch-requirement) below) with `conda install pytorch -c pytorch` and then install `somata` following this section.

While [`pip install`](https://pip.pypa.io/en/stable/cli/pip_install/) usually works, [an alternative way](https://pythonspeed.com/articles/conda-vs-pip/) to install `somata` is through the [conda-forge](https://conda-forge.org/docs/index.html) [channel](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/channels.html#what-is-a-conda-channel), which utilizes [continuous integration (CI)](https://conda-forge.org/docs/user/ci-skeleton.html) [across OS platforms](https://conda-forge.org/docs/user/introduction.html#why-conda-forge).
This means that [conda-forge packages](https://conda-forge.org/feedstock-outputs/index.html) are more [compatible with each other](https://conda-forge.org/docs/maintainer/adding_pkgs.html#avoid-external-dependencies) compared to [PyPI packages](https://pypi.org) installed via [`pip` by default](https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-from-pypi).
If `pip install somata` fails to resolve some dependencies, the [conda-forge somata](https://github.com/conda-forge/somata-feedstock) [feedstock](https://github.com/conda-forge/conda-feedstock#terminology) can be used to install `somata`:
```
$ conda install somata -c conda-forge
```

_When `somata` is installed into an [existing conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#viewing-a-list-of-your-environments), unmet dependencies are automatically searched, downloaded, and installed from the same repository of packages (currently either [PyPI](https://pypi.org/search/) or [conda-forge channel](https://conda-forge.org/packages/)) requested to provide the `somata` build distribution._

### torch requirement
If the [`torch`](https://pytorch.org) dependency is not resolved correctly for your [OS](https://whatsmyos.com) (such as installed the `cpu-only` version when GPU processing is needed), first [install `pytorch` manually](https://pytorch.org/get-started/locally/) in a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) that you want to install `somata` in, and then rerun either of the above two lines to install `somata`. Please be aware of [a common mixup](https://pypi.org/project/pytorch/) that PyTorch is distributed as [`torch` on PyPI](https://pypi.org/project/torch/) but as [`pytorch` on conda-forge](https://anaconda.org/conda-forge/pytorch/). If using [`conda`](https://pytorch.org/get-started/locally/) to install, Windows OS needs to download `pytorch` from the `pytorch` channel, as [`win-64` is not built on the `conda-forge`](https://anaconda.org/conda-forge/pytorch/) channel.

### (For development only)

- ### Fork this repo to personal git
    [How to: GitHub fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo)    

- ### Clone forked copy to local computer
    [How to: GitHub clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)

- ### Install conda
    [Recommended conda distribution: Miniforge3](https://github.com/conda-forge/miniforge#miniforge3)

    _[Apple silicon Mac](https://support.apple.com/en-us/HT211814): choose Miniforge3 native to the [ARM64 architecture](https://www.anaconda.com/blog/new-release-anaconda-distribution-now-supporting-m1) instead of [Intel x86](https://en.wikipedia.org/wiki/X86)._

- ### Create a new conda environment
    _You may also directly [install `somata` in an existing conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#using-pip-in-an-environment) by skipping this step._

    ``` $ cd <repo root directory with pyproject.toml> ```\
    ``` $ mamba create -n somata -c pytorch -c conda-forge --file .requirements/requirements-core.txt --file .requirements/requirements-dev.txt ```\
    ``` $ mamba activate somata ```

- ### Install somata as a package in development mode
    ``` $ cd <repo root directory with pyproject.toml> ```\
    ``` $ pip install -e . --config-settings editable_mode=compat ```

    _[What is: Editable Installs](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)_

- ### Configure IDEs to use the conda environment
    [How to: Configure an existing conda environment](https://code.visualstudio.com/docs/python/environments)

---

## Basic state-space models
`somata`, much like a neuron body supported by dendrites, is built on a set of basic state-space models introduced as class objects.

The motivations are to:
- develop a standardized format to store model parameters of state-space equations
- override Python dunder methods so `__repr__` and `__str__` return something useful
- define arithmetic-like operations such as `A + B` and `A * B`
- emulate `numpy.array()` operations including `.append()`
- implement inference algorithms like Kalman filtering and parameter update (m-step) equations as callable class methods

At present, and in the near future, `somata` will be focused on **time-invariant Gaussian linear dynamical systems**.
This limit on models we consider simplifies basic models to avoid nested classes such as `transition_model` and
`observation_model`, at the cost of restricting `somata` to classical algorithms with only some extensions to
Bayesian inference and learning. This is a deliberate choice to allow easier, faster, and cleaner applications of
`somata` in neural data analysis, instead of to provide a full-fledged statistical inference package.

---

### _class_ StateSpaceModel
```python
somata.StateSpaceModel(components=None, F=None, Q=None, mu0=None, S0=None, G=None, R=None, y=None, Fs=None)
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
 mu0.shape = None       S0 .shape = None
 G  .shape = None       R  .shape = None
 y  .shape = None       Fs = None
```

3. Model _stacking_ in `StateSpaceModel`

In many applications, there are several possible parameter values for a given state-space model structure. Instead of duplicating
the same values in multiple instances, somata uses _stacking_ to store multiple model values in the same object instance. Stackable
model parameters are `F, Q, mu0, S0, G, R`. For example:

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
 mu0.shape = None       S0 .shape = None
 G  .shape = None       R  .shape = None
 y  .shape = None       Fs = None

>>> print(s2)
<Ssm object at 0x102acc130>
 nstate   = 1     ncomp    = 0
 nchannel = 0     ntime    = 0
 nmodel   = 1
 components = None
 F  .shape = (1, 1)     Q  .shape = (1, 1)
 mu0.shape = None       S0 .shape = None
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
 mu0.shape = None       S0 .shape = None
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
 mu0.shape = None       S0 .shape = None
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
 mu0.shape = None       S0 .shape = None
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
 mu0.shape = None       S0 .shape = None
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
                       components='Osc', F=None, Q=None, mu0=None, S0=None, G=None, R=None, y=None, Fs=None)
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
 mu0.shape = (2, 1)     S0 .shape = (2, 2)
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
                    components='Arn', F=None, Q=None, mu0=None, S0=None, G=None, R=None, y=None, Fs=None)
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
 mu0.shape = (3, 1)     S0 .shape = (3, 3)
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
somata.GeneralSSModel(components='Gen', F=None, Q=None, mu0=None, S0=None, G=None, R=None, y=None, Fs=None)
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
`F, Q, S0, G, R`. To create a simple general state-space model:

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
 mu0.shape = None       S0 .shape = None
 G  .shape = None       R  .shape = None
 y  .shape = None       Fs = None
```

### For more in-depth working examples with the basic models in somata
Look at the demo script [basic_models_demo_04092024.py](examples/basic_models_demo_04092024.py) and execute the code in this file _line by line_ to get familiar with the class objects and methods of `somata` basic models.

---

## Advanced neural oscillator methods
1. [Oscillator Model Learning](#1-oscillator-model-learning)
2. [Phase Amplitude Coupling Estimation](#2-phase-amplitude-coupling-estimation)
3. [Oscillator Search Algorithms](#3-oscillator-search-algorithms)
4. [Switching State-Space Inference](#4-switching-state-space-inference)
5. [Multi-channel Oscillator Component Analysis](#5-multi-channel-oscillator-component-analysis)
6. [State-Space Event Related Potential](#6-state-space-event-related-potential)
7. [Dynamic Source Localization](#7-dynamic-source-localization)

---
<picture>
   <img align="right" src="https://img.shields.io/badge/Status-Functional-success.svg?logo=Python">
</picture>

### 1. Oscillator Model Learning

For fitting data with oscillator models, it boils down to three steps:
  - Initialize an oscillator model object
  - Perform state estimation, i.e., E-step
  - Update model parameters, i.e., M-step

Given some univariate time series `data`, we can fit an oscillator to the data using the expectation-maximization (EM) algorithm.
```python
from somata.basic_models import OscillatorModel as Osc
o1 = Osc(freq=1, Fs=100, y=data)  # create an oscillator object instance
_ = [o1.m_estimate(**o1.kalman_filt_smooth(EM=True))for x in range(50)]  # 50 EM steps
```

---
<picture>
   <img align="right" src="https://img.shields.io/badge/Status-Functional-success?logo=Python">
</picture>

### 2. Phase Amplitude Coupling Estimation

Quantifying phase amplitude coupling, as described by [Soulat et al. 2022](https://www.nature.com/articles/s41598-022-18475-3), consists of three steps:

1. Fitting an oscillator model to compute instantaneous phase and amplitude (either to full-length data or windowed epochs).
2. Fitting a constrained regression to the estimated phase ($\phi_t$) and amplitude ($A_t$) vectors:

$$
A_t = \beta_0 + \beta_1 \cos (\phi_t) + \beta_2 \sin (\phi_t) + \epsilon_t, \epsilon_t \sim \mathcal{N}(0, \sigma_\beta^2),
$$

$$
s.t. \ \ \beta_1^2 + \beta_2^2 \leq \beta_0^2.
$$

- Note that with $A_0 = \beta_0$, $K^{\text{mod}} = \sqrt{\beta_1^2 + \beta_2^2} / \beta_0$, and $\phi^{\text{mod}} = \tan^{-1}\left(\beta_2/\beta_1\right)$, this regression equation is equivalent to:

$$
A_t = A_0\left[1 + K^{\text{mod}} \cos \left(\phi_t - \phi^{\text{mod}}\right)\right] + \epsilon_t, \epsilon_t \sim \mathcal{N}(0, \sigma_\beta^2),
$$

$$
s.t. \ \ 0 \leq K^{\text{mod}} \lt 1.
$$

3. _(If there are multiple windows)_ Smoothing using an AR(p) model with observation noise. Model parameters are first learned through an instance of the EM algorithm initialized by numerical optimization of modified Yule-Walker equations. Kalman smoothing is then applied to $\boldsymbol{\beta}$ estimates across windows.

Step 1 can be accomplished using [Oscillator Model Learning](#1-oscillator-model-learning) or [Oscillator Search Algorithms](#3-oscillator-search-algorithms). We accomplish step 2 through Markov Chain Monte Carlo sampling using CmdStanPy; see [pac_model.py](somata/pac/pac_model.py). Functions to facilitate step 3 are also implemented in this module and can be called.

When using this module, please cite the following [paper](https://www.nature.com/articles/s41598-022-18475-3):

Soulat, H., Stephen, E. P., Beck, A. M., & Purdon, P. L. (2022). State space methods for phase amplitude coupling analysis. Scientific Reports, 12(1), 15940.

---
<picture>
   <img align="right" src="https://img.shields.io/badge/Status-Functional-success.svg?logo=Python">
</picture>

### 3. Oscillator Search Algorithms

There are two similarly flavored univariate oscillator search methods in this module: iterative oscillator search (iOsc) and decomposition oscillator search (dOsc) algorithms.

- iOsc: for a well-commented example script, see [IterOsc_example.py](examples/IterOsc_example.py).

_**N.B.:** We recommend downsampling to 120 Hz or less, depending on the oscillations present in your data. Highly oversampled data will make it more difficult to identify oscillatory components, increase the computational time, and could also introduce high frequency noise._

One major goal of this method was to produce an algorithm that requires minimal user intervention, if any. This algorithm is designed to fit well automatically in most situations, but there will still be some data sets where it does not fit well without intervention. We recommend starting with the algorithm as is, but in the case of poor fitting, we suggest the following modifications:

1. If the model does not choose the correct number of oscillations, we recommend looking at all fitted models and selecting the best fitting model based on other selection criteria or using your best judgement. You can also choose a subset of well-fitted oscillations and run `kalman_filt_smooth()` to estimate oscillations using those fitted parameters.

2. This algorithm assumes stationary parameters, and therefore a stationary signal. Although the Kalman smoothing allows the model to work with some time-varying signal, the success of the method depends on the strength and duration of the signal components. The weaker and more brief the time-varying component is, the more poorly the model will capture it, if at all. We recommend decreasing the length of your window until you have a more stationary signal.

When using the original iOsc algorithm, please cite the following [paper](https://www.biorxiv.org/content/10.1101/2022.10.30.514422):

Beck, A. M., He, M., Gutierrez, R. G., & Purdon, P. L. (2022). An iterative search algorithm to identify oscillatory dynamics in neurophysiological time series. bioRxiv, 2022-10.

- dOsc: for a well-commented example script, see [DecOsc_example.py](examples/DecOsc_example.py).

---
<picture>
   <img align="right" src="https://img.shields.io/badge/Status-Functional-success.svg?logo=Python">
</picture>

### 4. Switching State-Space Inference

When using this module, please cite the following [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011395):

He, M., Das, P., Hotan, G., & Purdon, P. L. (2023). Switching state-space modeling of neural signal dynamics. PLOS Computational Biology, 19(8), e1011395.

---
<picture>
   <img align="right" src="https://img.shields.io/badge/Status-Missing-critical.svg?logo=Python">
</picture>

### 5. Multi-channel Oscillator Component Analysis

---
<picture>
   <img align="right" src="https://img.shields.io/badge/Status-Missing-critical.svg?logo=Python">
</picture>

### 6. State-Space Event Related Potential

---
<picture>
   <img align="right" src="https://img.shields.io/badge/Status-Functional-success.svg?logo=Python">
</picture>

### 7. Dynamic Source Localization

---

## Authors
Mingjian He, Proloy Das, Ran Liu, Amanda Beck, Patrick Purdon

## Citation
Use different citation styles at: https://doi.org/10.5281/zenodo.7242130

## License
SOMATA is licensed under the [BSD 3-Clause Clear license](https://spdx.org/licenses/BSD-3-Clause-Clear.html).\
Copyright © 2024. All rights reserved.
