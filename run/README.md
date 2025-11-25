# Using Training Scripts
Each training script corresponds to a mode in the prepared ``launch.json``, with which they can be started directly from "run and debug" tab or altenatively, from the command line. To simplify the commandline, configurations are primarily given via the configuration ``.yaml`` files found in [config](../src/configs/). The only arguments that can be passed via the command line are: 
- ``config_path`` $\rightarrow$ the path to the ``.yaml`` config file
- ``wandb_project`` $\rightarrow$ the weights and biases project name for logging
- ``wandb_entity`` $\rightarrow$ the weights and biases entity name for logging
- ``random_seed`` $\rightarrow$ optional random seed for reproducibility
- ``device`` $\rightarrow$ optional device

## Training Scripts Overview
This folder contains example training scripts for the various algorithms in ``src.algorithms``, specifically for Flatland. Currently implemented are: 
- Proximal Poliy Optimisation (PPO)
    - Single worker version (``src.algorithms.PPO``)
    - Synchronous Multiprocessing version (``src.algorithms.Sync_PPO``)
- Asynchronous PPO (``src.algorithms.IMPALA``)

