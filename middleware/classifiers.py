import random
from abc import ABC, abstractmethod
from models.ml import LabelModelOutput


class MultilabelClassifier(ABC):
    labels = [
        "Artificial Intelligence Introduction",
        "Intelligent Agents",
        "Search Strategies",
        "Adversarial Search",
        "First Order Logic",
        "Quantifying Uncertainty",
        "Bayesian Network",
        "Fuzzy Set",
        "Fuzzy Logic",
        "Optimization",
        "Genetic Algorithm",
        "Introduction to Machine Learning",
        "Linear Regression",
        "Decision Tree",
        "Support Vector Machine (SVM)",
        "K-Nearest Neighbour",
        "K-Means",
        "Principal Component Analysis",
        "Neural Network and Deep Learning Introduction",
        "Artificial Intelligence Ethics",
        "Application in Artificial Intelligence"
    ]

    @abstractmethod
    def predict_labels(self, content: str) -> list[LabelModelOutput]:
        pass


class RandomClassifier(MultilabelClassifier):
    def __init__(self):
        self.label_names = self.labels

    def predict_labels(self, content: str) -> list[LabelModelOutput]:
        return [LabelModelOutput(label_name, random.random())
                for label_name
                in self.label_names]
