from typing import List, Tuple
from flatland.envs.rail_env import RailEnv
from src.utils.graph.flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser
from src.utils.graph.flatland_railway_extension.RailroadSwitchCluster import RailroadSwitchCluster
from src.utils.graph.MultiDiGraphBuilder import MultiDiGraphBuilder

class SolutionTranslator(): 
    def __init__(self, env: RailEnv):
        self.multidigraph = MultiDiGraphBuilder(env)

    def encode_solution(self, coordinate_path) -> List[int]:
        """
        Translates a list of coordinates into a solution path for the Flatland environment.
        """
        # TODO: implement translation coordinates -> path
        # raise NotImplementedError("Solution generation not implemented yet.")
        pass
    
    def decode_solution(self, track_path) -> List[Tuple[int, int]]:
        """
        Translates a solution path from the Flatland environment into a list of coordinates that the agent can follow. 
        """
        # TODO: implement translation path -> coordinates
        # raise NotImplementedError("Solution decoding not implemented yet.")    
        pass
    
