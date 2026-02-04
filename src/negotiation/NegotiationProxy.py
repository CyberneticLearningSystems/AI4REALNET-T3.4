from typing import Dict


class NegotiationProxy:
    def __init__(self):
        self.params: Dict = {}

    def _init_params(self, negotiation_params: Dict):
        self.params = negotiation_params

    def add_params(self, new_params: Dict):
        """
        Add new negotiation parameters during runtime based on human input.
        """
        for key, value in new_params.items():
            if key not in self.params:
                self.params[key] = value    

    def adjust_param(self, new_param: str, new_value):
        """
        Adjust a single negotiation parameter during runtime based on human input.
        """
        if new_param in self.params:
            self.params[new_param] = new_value
        else: 
            Warning(f"Parameter {new_param} not found in negotiation parameters.")

    def adjust_params(self, new_params: Dict):
        """
        Adjust of negotiation parameters during runtime based on human input.
        """
        for key, value in new_params.items():
            if key in self.params:
                self.params[key] = value
            else:
                Warning(f"Parameter {key} not found in negotiation parameters.")

    def perform_negotiation(self):
        """
        Initialize the negotiation process with the current parameters.
        """
        pass
