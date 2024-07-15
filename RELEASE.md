# Release log

## 0.5.1
*July 2024*

#### New features

- Add a phase amplitude coupling (PAC) analysis module
- Add a new class to perform decimated EM learning with state-space models

#### Closed issues

- Fix an issue with log likelihood calculations with multivariate observed data y
- Rename initial state noise covariance matrix at time point t=0 from Q0 to S0
- Correctly unfill components when calling m_estimate on Ssm class objects
- Fix how different basic models can be concatenated together (e.g., Arn and Osc)

## 0.4.1
*March 2024*

#### New features

- Introduce a spectral factorization method to initialize oscillator parameters
- Introduce DecomposedOscillatorModel to supersede the iOsc algorithm in most applications
- Rename iterative_oscillator module to oscillator_search
- Introduce diagnostic plotting and statistical tests for analyzing residuals
- Introduce dynamic source localization using oscillator models with GPU acceleration

#### Closed issues

- Fix an issue with h_t and y lengths in _m_update_r
- Fix an extra parameter in basic_models_demo
- Unify the output arguments of spectrum as (psd, freq)
- Update instruction on editable pip install to work with static code analyzers

## 0.3.1
*June 2023*

#### New features

- Update iterative oscillator algorithm to use new routines
- Introduce switching module for switching state-space models

#### Closed issues

- Fix an issue of iOsc only adding AR1 component to slow oscillation
- Fix plotting issues of the multitaper module

## 0.2.1
*November 2022*

#### New features

- Introduce iterative oscillator algorithm

#### Closed issues

- Fix mathjax rendering of state-space equations in README.md

## 0.1.1
*October 2022*

#### New features

- Introduce four basic state-space model classes: StateSpaceModel, OscillatorModel, AutoRegModel, GeneralSSModel
- Add exact inference algorithms with Gaussian noise processes for the introduced basic models
- Add expectation-maximization(EM) learning algorithm using a general run_em() wrapper function

#### Closed issues

- StateSpaceModel.\_initialize_from_components() method now works correctly with the .concat\_() method
