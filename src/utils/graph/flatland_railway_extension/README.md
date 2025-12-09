# Flatland Railway Extension
This extension contains function which exten the flatland rail simulator, and are partially developed by [Adrian Egli](https://github.com/aiAdrian), taken from the [flatland_railway_extension](https://github.com/aiAdrian/flatland_railway_extension/tree/master?tab=readme-ov-file) repo, and extended by [me](https://github.com/ozmanda). 

## RailroadSwitchAnalyser and RailroadSwitchCluster
The ``RailroadSwitchCluster`` utilises the ``RailroadSwitchAnalyser`` to identify switches and tracks which belong to one cluster. The ``RailroadSwitchAnalyser`` searches the network for all switch cells, where a switch is defined as any cell where more than one transition exists. It also cotains the following functions: 
- ``check_agent_decision`` $\rightarrow$ checks (returns boolean values) whether the agent is: 
    1. At a switch cell
    2. In a cell neighbouring a switch cell

- ``is_diamond_crossing`` $\rightarrow$ checks whether the current cell is a diamond crossing
- ``is_dead_end`` $\rightarrow$ checks whether the current cell is a dead end
- ``is_switch_neighbor`` $\rightarrow$ checks whether the current cell neighbours a switch cell

## FlatlandGraphBuilder
The ``FlatlandGraphBuilder`` builds a graph similar to the ``MultiDiGraphBuilder``, also using the ``networx`` package, however structures the graph differently (see [graph module README](../README.md))
