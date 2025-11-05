# Controllers

Controllers contain the logic for action selection and value estimation. The following controllers have been implemented: 
- **PPOController:** an actor-critic controller with simple feed-forward networks
- **LSTMController:** an actor-critic controller with feature extraction and LSTM layers, followed by action and value heads


## Base Controller
All controllers are structured according to ``BaseController``, which guarantees compatibility with synchronous and asynchronous training pipelines. This controller contains the following basic functions: 
- ``update_weights``: allows for the updating of network parameters from the learner to the workers
- ``get_state_dict``: returns the current network parameters
- ``sample_action``: returns actions sampled from the distribution, the log_probs, the estimated state values and "extras" (in the case of LSTM, these are the hidden states and cell states). 
- ``state_values``: returns the state values for the given states


## PPOController
The ``PPOController`` builds three feedforward networks: the encoder, the actor and critic. The sizes of these networks are controlled using the "encoder", "actor_config" and "critic_config" elements of the PPOControllerConfig. 

<img src="../../imgs/PPOControllerNetwork.jpg" alt="Visualisation of the PPOController Network" width="400" />

## LSTMController
The ``LSTMController`` extends the ``BaseController`` to consider a LSTM network structure, building a similar network architecture to the ``PPOController`` with the addition of a LSTM Cell in between the feature extraction (encoder) network.

<img src="../../imgs/LSTMControllerNetwork.jpg" alt="Visualisation of the PPOController Network" width="400" />