# Solution Utils
The functions in this package serve to support the translation of atomic actions in flatland to human-readable solutions

## Solution Translation
The ``SolutionTranslator`` encodes and decodes solutions, essentially serving as the intersection between the artificial agents and the human agents. Two core functions are considered: 
1. **Solution Encoding:** artificial agent $\rightarrow$ human agent
    - The artificial agent passes a series of waypoints in the form of switches, which is translated to a path given as a list of track IDs (extracted using [flatland_railway_extension](../graph/flatland_railway_extension/README))
2. **Solution Decoding:** human agent $\rightarrow$ artificial agent
    - The human agent passes a path in the form of track IDs, which are decoded into an explicit path the agent can then follow. 