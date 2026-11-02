import json


class SumEnv:
    def __init__(self) -> None:
        self.func_mapping = {"Observe": 0, "Done": 1, "Earn": 2}

    def step(self, action: str) -> None:
        call = json.loads(action)
        params = call["parameters"]
        action_code = self.func_mapping[call["name"]]
        if action_code == 0:
            self.Observe(params.get("nested", []))
        elif action_code == 1:
            self.Done(params["answer"])
        elif action_code == 2:
            self.Earn(params["answer"])

    def Observe(self, nested: list) -> str:
        """Observe the current values."""
        return "[1, 2]"

    def Done(self, answer) -> str:
        """Submit the sum. answer (int): The proposed result."""
        return str(answer == 3)

    def Earn(self, answer: int) -> str:
        """Award an intermediate reward."""
        return str(answer == 3)

    def get_ref_answer(self) -> int:
        return 3
