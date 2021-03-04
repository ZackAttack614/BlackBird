# BlackBird ![Documentation Status](https://readthedocs.org/projects/blackbirdai/badge/?version=latest&style=plastic)

BlackBird is a self-learning algorithm designed for board games, based on the AlphaZero implementation. Future versions of this project will include extensions from state-of-the-art research into reinforcement learning optimization.

Code-level documentation can be found [here](http://blackbirdai.readthedocs.io/en/latest).

## Requirements and Suggested Tools
- This project is not compatible with Python 2.x; please use 3.5+.
- It is strongly advised that you use the GPU version of tensorflow, should you have a compatible Nvidia GPU. If you have not yet installed this, you can follow the directions [here](https://www.tensorflow.org/install/install_windows#requirements_to_run_tensorflow_with_gpu_support).

## Getting Started

Depending on the level of interaction you want with BlackBird, you can either run the default setup, or modify the parameters and the game that BlackBird learns.

- The learning script `main.py` expects that there is a file called `parameters.yaml`, which describes the setup for the algorithm to learn from. By default, this file does not exist, but you should copy the `parameters_template.yaml` file into `parameters.yaml`, and modify it to your liking.
- Because this project builds its documentation automatically with Sphinx, note that there are Sphinx-related packages in the `requirements.txt` file.
- The packages that are required to run BlackBird may be unrelated to your workflow elsewhere, so it is recommended that you follow these steps to create a virtual environment in which to install them.
  1. `pip install virtualenv`
  1. `python -m virtualenv blackbird`
  1. `source blackbird/bin/activate`

### Quickstart

To get started quickly, run the following commands in a terminal shell.

```bash
$ git clone https://github.com/ZackWolf614/BlackBird
$ cd BlackBird
$ pip install -r requirements.txt
$ cp parameters_template.yaml parameters.yaml
$ python main.py
```

### Tuning Parameters

The `parameters.yaml` file contains the setup for BlackBird's neural network architecture, and how it learns from its self-generated games. The major parameter headings are
- `mcts`: Settings determining how the Monte Carlo Tree Search algorithm performs.
- `network`: Settings determining the structure of the neural network, and learning parameters.
- `selfplay`: Settings determining how many games BlackBird should play against itself - for game generation and for performance testing.
