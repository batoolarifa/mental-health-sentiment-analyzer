import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder

class SentimentModel:
    def __init__(self, model_dir, labels):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()  # evaluation mode

        # Label encoder
        self.le = LabelEncoder()
        self.le.fit(labels)

        self.max_len = 200

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        pred_idx = torch.argmax(logits, dim=1).item()
        return self.le.inverse_transform([pred_idx])[0]