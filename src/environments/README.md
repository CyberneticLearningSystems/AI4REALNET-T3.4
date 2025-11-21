# Flatland Environments
This module contains a set of flatland scenarios and functions pertaining to these functions. To begin, the Flatland drawing board (credit: Manuel Meyer @ Flatland Association) allows custom topologies, lines and schedules to be made in an intuitive, graphical interface. These scenarios / environments are then saved as ``.json`` files, can be loaded with ``scenario_loader.py`` and rendered with ``scenario_redering.py``. 

Following is an overview of the current scenarios, at the end a quick summary of the drawing board interface and how to create scenarios. 

:warning: Currently known issues with drawing board/scenarios that still need to be fixed :warning:

- Stations that aren't the target of an agent aren't rendered properly (flatland doesn't plan on fixing this in the near future), a workaround could be to plan additional schedules with waypoints as targets and end the simulation early.
- Complex ordering switch types need to be changed, as the target station isn't reachable from some of the starting points. 

:construction: Work in Progress :construction:
- Currently working on creating scenario variants as has been done for the simple avoidance scenarios, with pre-calculated delays that allow for experimentation with human agents.
- Establishing more complex topologies and schedules with FHNW APS that contain multiple conflicts that need to be managed for experimentation with human agents. 

## Flatland Test Scenarios
The environments and scenarios contained here are created using the drawing board created by Manuel Meyer @ Flatland Association. Each scenario contains a physical network topology, stations, lines and schedule. Lines consist of a series of stations, schedules define the earliest departure and latest arrive times at each station, and is operated by a specific train type. Train types currently only differentiate themselves in speed, but can also be used to set priorities (i.e., an InterCity could be prioritised over cargo trains). In the following, the test scenarios are visualised together with their schedules, and the characteristic aimed to be tested with the scenario. 

### Simple Avoidance
![Simple Avoidance Scenario](./renders/scenario_simple_avoidance.png)


| Station | Latest Arrival Time         | Earliest Departure Time      |
|---------|-----------------------------|------------------------------|
| **Schedule W-E** | 
| 1       | N/A                         | 1.0                          |
| 2       | 8                           | N/A                          |
| **Schdule E-W** |
| 2       | N/A                         | 1.0                          |
| 1       | 8                           | N/A                          |

In this scenario, to avoid a deadlock, one of the two trains must be routed over the longer track to the north, leading to it being delayed. There are two variants of this scenario, which allow for different types of problem solving. **Scenario 1** contains two trains of the same speed, albeit of different train classes and is aimed at train-type prioritisation. **Scenario 2** contains trains of different speed and class, making the problem less straight-forward and allowing for more complex problem solving.  


| Scenario Version | Train W-E Type | Train W-E Speed | Train W-E Delay | Train E-W Type | Train E-W Speed | Train E-W Delay | 
| ---------------- | ------------ | ------------- | ------------- | ------------ | ------------- | ------------- | 
| 1 | Inter-City | 1 | 1.8 min | Express | 1 | 1.8 min | 
| 2 | Inter-City | 1 | 1.8 min | Regional | 0.8 | 2 min | 


## Complex Avoidance
Complex avoidance is topologically identical to simple avoidance, however in this scenario, three trains are considered - one moving from west-east, and two moving east-west. The aim of this scenario is to test whether the agents can identify that the optimal solution is to route the two agents moving east-west via the shorter path. 


| Station | Latest Arrival Time         | Earliest Departure Time      |
|---------|-----------------------------|------------------------------|
| **Line W-E** |
| 1       | N/A                         | 1.0                          |
| 2       | 10                          | N/A                          |
| **Line E-W 1** |
| 2       | N/A                         | 1.0                          |
| 1       | 10                          | N/A                          |
| **Line E-W 2** |
| 2       | N/A                         | 3.0                          |
| 1       | 12                          | N/A                          |


### Simple Ordering
The simple ordering scenario aims to test the ability of the agents to organise themselves according to given priorities when merging at a switch. Two lines exist: 
- Line 1.1: from upper rail of station 1 via station 7 to final station 8
- Line 1.2: from lower rail of station 1 via station 7 to final station 8

(currently there is an issue with station visualisation - only final stations are visualised by the flatland renderer. Station 7 is at ``(1,2)`` and ``(2,2)``, station 8 is at ``(2,6)``)

![Simple Avoidance Scenario](./renders/scenario_simple_ordering.png)


| Station | Latest Arrival Time         | Earliest Departure Time      |
|---------|-----------------------------|------------------------------|
| **Line 1.1** |
| 1       | N/A                         | 1.0                          |
| 7       | 6                           | 8                            |
| 8       | 11                          | N/A                          |
| **Line 1.2** |
| 1       | N/A                         | 0                            |
| 7       | 5                           | 7                            |
| 8       | 10                          | N/A                          |


### Complex Ordering
The complex ordering scenario tests the same behaviour as the simple ordering scenario, with more tracks merging before the stations.

(Currently not working & needs to be fixed - merge switch)

![Simple Avoidance Scenario](./renders/scenario_complex_ordering.png)


### Overtaking
This scenario, as the name indicates, tests how agents initiate overtaking when the train behind is faster than the train ahead. 

![Simple Avoidance Scenario](./renders/scenario_overtaking.png)

| Station | Latest Arrival Time         | Earliest Departure Time      |
|---------|-----------------------------|------------------------------|
| **Line W-E Fast** |
| 1       | N/A                         | 2.0                          |
| 2       | 10                          | N/A                          |
| **Line W-E Slow** |
| 1       | N/A                         | 1.0                          |
| 7       | 14                          | N/A                          |


## Drawing Board
Launching the drawing board and loading the example ``simple_avoidance_1.json`` via the ``import all`` button will show a view of the network topology with the following options for editing:
- **Grid sizing:** determine the number of rows and columns in the grid, as well as the size at which it is displayed in the interface (px). To apply your changes, click ``Inititialise/Resize``, ``Clear Grid`` resets to the standard 15x20 grid.
- **Lines:** define routes along which trains can move, essentially a series of stations that must be followed
- **Schedules:** specify the type of train, the latest arrival and the earliest departure time for a specific line. A single line can have multiple schedules

![drawing board interface view](../../imgs/drawing_board_scenario_1.png)

Clicking on the ``Lines & Schedules`` button opens the "Operations Manager" which gives the option to edit the lines, schedules and the train types available in the environment. 

![drawing board operations manager view](../../imgs/drawing_board_om.png)

To define a line, a series of station IDs must be entered, and for origin / target stations that have multiple tracks, a staring and end cell can optionally be defined. 


![drawing board lines view](../../imgs/drawing_board_lines.png)

Once lines have been defined schedules can be defined on them. This required a schedule name, a train class, and a standard dwell time at each station (optional). The shift functions is meant to help with regular schedules, allowing for all times to be shifted by a factor (for example 60 to simulate an hourly schedule). The simulator pre-calculates the earliest possible arrival time based on the shortest path in the ``Arrival`` column and the earliest possible departure time based on the dwell time in the ``Departure`` column. Within these limits, the latest arrival and earliest departure times can be set arbitrarily. 

![drawing board schedule view](../../imgs/drawing_board_schedule.png)

Train classes are set simply by providing a name and giving it a speed, which corresponds to the number of cells it can traverse in one environment step. When you have finished designing your grid, lines and schedules, you can export it to a ``.json`` which can then be loaded as a ``RailEnv`` using ``scenario_loader.py``. Importing such a ``.json`` allows it to be edited. 

