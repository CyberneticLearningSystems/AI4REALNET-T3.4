# T3.4 Hybrid Approach Flatland + Blackbox + HMI Demo
This project combines the T3.4 Flatland environment with a custom Blackbox planner (based on PP and CBS) and a simple HMI interface for interactive decisions.
The idea is to explore how classical planning methods can be combined with more dynamic or human-in-the-loop approaches in a multi-agent setting.


## Features
* Uses Flatland RailEnv scenarios (JSON)
* Supports CBS (Conflict-Based Search) and PP (Prioritized Planning)
* Simple HMI interaction via tokens (e.g. PRIORITY)
* Replanning logic based on user input
* Custom scenario loading from JSON
* Conversion from plans to executable actions


## Reproducibility
There are two requirement files:

* requirements.txt → basic setup to run the project
* requirements_lock.txt → full environment (from pip freeze)

If you want the exact same setup:
pip install -r requirements_lock.txt


## Running the demo
The repository provides different entry points depending on the use case.

1. Main demo (HMI)
python app_hmi_demo.py
Runs the interactive demo with HMI token selection.

2. Controller demo
python run_controller.py
Runs the planner and controller logic (CBS / PP) without HMI.

3. Basic scenario run
python run_scenario.py
Runs a predefined scenario without replanning or interaction. Useful for testing and debugging.

Scenarios can be changed in the script, for example:
scenario_path = Path("src/environments/simple_avoidance.json")


## HMI interaction
Currently, the system supports simple tokens like:
* PRIORITY → gives one agent preference over another

This mainly affects:
* planning order
* agent delays
* replanning behaviour


## Project structure
Note: The `flatland_blackbox` folder contains a local copy of the Blackbox planner used for CBS and PP.

```
.
├── src/
│   ├── environments/
│   │   ├── scenario_loader.py
│   │   └── *.json                     # Flatland scenarios
│   │
│   ├── planners/
│   │   ├── blackbox_adapter.py       # CBS / PP integration
│   │   ├── plan_follower.py          # Plan → actions
│   │   ├── state_extraction.py       # Extract planner state from env
│   │   └── token_utils.py            # Token handling logic
│   │
│   ├── utils/
│   │   └── env_reference.py          # Environment helpers / references
│   │
│   ├── widgets/
│   │   ├── action_token_selector.py  # HMI token selection
│   │   └── human_input.py            # User input handling
│   │
│   └── app_hmi_demo.py               # Main HMI demo entry point
│
├── flatland_blackbox/                # External Blackbox integration (CBS / PP)
│
├── run_controller.py                # Planner/controller execution
├── run_scenario.py                  # Basic scenario execution
│
├── requirements.txt
├── requirements_lock.txt
└── README.md
```

## Planning
* CBS: more optimal but slower
* PP: faster but can fail in some cases (deadlocks)
* Priority mode: tries to guide planning using HMI input


## Notes
Developed and tested on Windows (conda environment)
Written in Python
Mainly intended as a demo / experiment setup


## Context / Related Projects
This project builds on top of the following external components:
- Flatland (rail environment)
- Blackbox planner (CBS / PP implementation)
This repository does not include a full standalone implementation of these systems.
It mainly provides an integration layer and uses the Blackbox planner for CBS and PP logic.


## Acknowledgements
This project builds on ideas and tools from:
- Flatland environment  
- AI4REALNET  
- multi-agent path finding methods (CBS, PP)  
