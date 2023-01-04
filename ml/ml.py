import json
import torch
from transformers import DistilBertModel, DistilBertTokenizer


class ClassifierModel(torch.nn.Module):
    def __init__(self):
        super(ClassifierModel, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 20)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


class CourseLabelClassifier:
    def __init__(self, model: ClassifierModel, tokenizer: DistilBertTokenizer, label_refs: dict[int, str]):
        self.model = model
        self.tokenizer = tokenizer
        self.label_refs = label_refs

    def predict(self, text: str) -> dict[str, float]:
        tokenized = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=300,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = torch.tensor(tokenized['input_ids'], dtype=torch.long).unsqueeze(0)
        mask = torch.tensor(tokenized['attention_mask'], dtype=torch.long).unsqueeze(0)
        token_type_ids = torch.tensor(tokenized["token_type_ids"], dtype=torch.long).unsqueeze(0)
        raw_output = torch.sigmoid(self.model(ids, mask, token_type_ids)).detach().numpy().tolist()[0]
        return {self.label_refs[i]: val for i, val in enumerate(raw_output)}


def load_model(model_path: str, vocabulary_path: str, labelrefs_path: str) -> CourseLabelClassifier:
    model = ClassifierModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained(vocabulary_path)
    with open(labelrefs_path, 'r') as f:
        label_refs = json.load(f)
    label_refs = {int(i): val for i, val in label_refs.items()}
    return CourseLabelClassifier(model, tokenizer, label_refs)
