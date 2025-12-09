import unittest
from typing import List, Tuple
# from src.utils.solution.solution_translation import SolutionTranslator
from src.environments.scenario_loader import load_scenario_from_json
from src.environments.env_small import small_flatland_env
from flatland.envs.rail_env import RailEnv

from src.utils.graph.flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser
from src.utils.graph.flatland_railway_extension.RailroadSwitchCluster import RailroadSwitchCluster

from src.utils.solution.solution_translation import SolutionTranslator


class TestSolutionTranslator(unittest.TestCase):
    def setUp(self):
        scenario_1 = "src/environments/simple_avoidance_1.json"
        self.env1 = load_scenario_from_json(scenario_1)
        self.env1.reset()
        self.env1_path1 = [
            (2,2),
            (2,3),
            (2,4),
            (1,4),
            (1,5),
            (1,6),
            (2,6),
            (2,7),
            (2,8)
        ]
        self.env1_path2 = [
            (2,2),
            (2,3),
            (2,4),
            (2,5),
            (2,6),
            (2,7),
            (2,8)
        ]

        scenario_2 = "src/environments/simple_ordering.json"
        self.env2 = load_scenario_from_json(scenario_2)
        self.env2.reset()
        self.env2_path1 = [
            (1,2),
            (1,3),
            (1,4),
            (1,5),
            (2,5),
            (2,6),
            (2,7),
            (2,8)
        ]
        self.env2_path2 = [
            (2,2),
            (2,3),
            (2,4),
            (2,5),
            (2,6),
            (2,7),
            (2,8)
        ]

        # TODO: setup environment for solution testing using pre-defined scenarios (e.g. simple_avoidance)

    def test_solution_translator(self) -> None:
        solution1, solution2 = self.encode_solution()
        self.decode_solution

    def encode_solution(self) -> Tuple[List[int], List[int]]:
        # TODO: implement test for encoding solution
        solution_translator: SolutionTranslator = SolutionTranslator(self.env1)
        
        # Test path -> solution encoding
        solution_translator.encode_solution(self.env1_path1)
        pass

    def decode_solution(self):
        # TODO: implement test for decoding solution
        pass
