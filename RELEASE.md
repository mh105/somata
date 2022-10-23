# Release log

## 0.1.1
*October 2022*

#### New features

- Introduce four basic state-space model classes: StateSpaceModel, OscillatorModel, AutoRegModel, GeneralSSModel
- Add exact inference algorithms with Gaussian noise processes for the introduced basic models
- Add expectation-maximization(EM) learning algorithm using a general run_em() wrapper function

#### Closed issues

- StateSpaceModel.\_initialize_from_components() method now works correctly with the .concat\_() method
