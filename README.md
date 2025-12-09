# AI4REALNET-T3.4
Repository for experimental development of fully autonomous AI for the AI4REALNET project. The work focuses on the development of solutions for multi-agent systems, using the flatland simulation environment. 

## Installation
These packages were developed with python 3.10.11 For package versions see the ``requirements.txt``. In the .vscode folder, a ``launch.json`` and ``settings.json`` are available to run the different models and perform unittesting. 

**Example:** to run a simple, non-parallelised PPO Training run, set your desired settings in a config file, for example ``PPO_FNN.yaml`` and pass it as a command line argument when running ``ppo.py``. Alternatively, use the ``launch.json`` to start the script directly under ``PPO`` in the run and debug tab. The controller, learner and environment will automatically be setup according to the given configuration. 

## Repo Structure
- ``.vscode`` $\rightarrow$ contains examples for launching for model training
- ``imgs`` $\rightarrow$ contains images for READMEs
- ``models`` $\rightarrow$  contains saved models from training
- ``run`` $\rightarrow$ contains model training scripts which can be run either from VSCode or from commandline - more information on how to train models is available in the [run README](./run/README.md)
- ``src`` $\rightarrow$ contains relevant source code (algorithms, networks, utility functions, etc.)
- ``test`` $\rightarrow$ contains all test funcitons for the sourcecode

