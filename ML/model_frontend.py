from abc import abstractmethod


class ModelFrontend:
    @abstractmethod
    def predict_labels(self, content: str) -> list[str]:
        pass


class MockupFrontend(ModelFrontend):
    def __init__(self, default_response: list[str] = ['Hi']):
        self.default_response = default_response

    def predict_labels(self, content: str) -> list[str]:
        return self.default_response
