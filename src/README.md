# Main Code

This codebase is structured as follows: 
- **algorithms:** algorithm implementations and configurations, each consisting of the following: 
    - **Learner:** contains all learning logic, updates the policy based on experiences gathered from rollouts.
    - **Worker:** a parallel instance which manages a ``Runner`` and returns experiences in the form of ``Rollouts``
- **configs:** contains all config classes and ``.yaml`` config files.
- **controllers:** interacts with the environment according to the current policy
- **environments:** contains file and scripts to initialise flatland environments for various testing scenarios
- **memory:** various memory classes, including the default ``MultiAgentRolloutBuffer``
- **negotiation:** contains all classes and functions relating to agent negotiation.
- **networks:** network modules used to construct controllers.
- **utils:** utility functions, including the graph representation of the environment and associated functions.

Each module contains its own README with details on the contents of the module, a brief summary of the most important modules are given here. 

### Algorithms
1. Simple PPO algorithm
2. Synchronous, parallel PPO algorithm
3. Asynchronous, IMPALA-style PPO algorithm

### Controllers
Currently, two main controller types have been implemented: 
1. Actor-critic PPO controller 
2. Recurrent LSTM controller

### Environments
The environments contain a series of flatland scenarios and the loading functions for them, ranging in complexity from extremely simple, two-agent scenarios to larger scenarious with multiple agents and stations. 

### Negotiation
The negotiation module contains the backend logic for the negotiation process when conflicts are detected. This is a WIP and currently only contains skeleton functions. 

### Reward
All reward functions / classes are contained here. 

### Utility Functions
Within the ``utils`` folder, some important submodules for the T3.4 functions have been programmed: 
1. ``graph`` $\rightarrow$ contains translation from flatland environment to a ``NetworkX`` graph class and a series of functions performed on the graph.
2. ``observation`` $\rightarrow$ contains all utility functions pertaining to environment observation (normalisation, running means / standard deviations, and so on)
3. ``solution`` $\rightarrow$ contains  all utility functions pertaining to solutions, such as encoding and decoding of solutions from flatland actions $\longleftrightarrow$ human-readable solutions