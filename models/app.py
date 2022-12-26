from models.ml import LabelModelOutput


class LabelDisplayDetails:
    def __init__(self, output: LabelModelOutput):
        self.name = output.name
        self.percentage_str = str(int(output.confidence * 100)) + "%"
        