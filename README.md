# AI4REALNET-T3.4
Repository for experimental development of multi-agent control systems in the Flatland environment.

This version extends the original framework with:
- a Human-in-the-Loop interface (HMI)
- integration of a PPO-based controller
- additional approaches (Hybrid and Co-Learning)

## Installation
These packages were developed with python 3.10.11. For package versions see the ``requirements.txt``. In the .vscode folder, a ``launch.json`` and ``settings.json`` are available to run the different models and perform unittesting. 

**Example:** to run a simple, non-parallelised PPO Training run, set your desired settings in a config file, for example ``PPO_FNN.yaml`` and pass it as a command line argument when running ``ppo.py``. Alternatively, use the ``launch.json`` to start the script directly under ``PPO`` in the run and debug tab. The controller, learner and environment will automatically be setup according to the given configuration. 

## HMI Integration
This repository includes a graphical Human-Machine Interface (HMI) for interactive control during simulation.
The HMI allows injecting high-level decisions at runtime, while the PPO controller remains the base decision layer.

## Running the HMI Demo
To run the interactive HMI application:
python app_hmi_flatland.py

## Running without HMI
To run a simple Flatland simulation without the HMI:
python run_flatland.py

## Repo Structure
- ``.vscode`` $\rightarrow$ contains examples for launching for model training
- ``Co-Learning Approach`` → alternative multi-agent learning implementation
- ``Hybrid Approach`` → hybrid planning + HMI integration
- ``imgs`` $\rightarrow$ contains images for READMEs
- ``models`` $\rightarrow$  contains saved models from training
- ``run`` $\rightarrow$ contains model training scripts which can be run either from VSCode or from commandline - more information on how to train models is available in the [run README](./run/README.md)
- ``src`` $\rightarrow$ contains relevant source code (algorithms, networks, utility functions, etc.)
- ``test`` $\rightarrow$ contains all test functions for the sourcecode
- ``app_hmi_flatland.py`` → to run the interactive HMI application
- ``run_flatland.py`` → basic Flatland execution script without HMI