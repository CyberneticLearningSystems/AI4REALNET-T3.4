from flatland.envs.rail_env import RailEnv

def SolutionTranslator(): 
    def __init__(self, env: RailEnv):
        self.env = env

    def encode_solution(self, coordinate_path):
        # TODO: implement translation coordinates -> path
        raise NotImplementedError("Solution generation not implemented yet.")
    
    def decode_solution(self, track_path):
        # TODO: implement translation path -> coordinates
        raise NotImplementedError("Solution decoding not implemented yet.")    
    
