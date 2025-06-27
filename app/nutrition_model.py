import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# Load model + tokenizer once at startup
model_path = r"C:\Users\alici\Desktop\code\misc\recipe_recommender\models\finetuned_miniLM"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create label list and save it as a json 
df = pd.read_csv(r"C:\Users\alici\Desktop\code\misc\recipe_recommender\data\nutrition_intent_dataset.csv")
label_list = sorted(df["Label"].unique().tolist())

with open (r"C:\Users\alici\Desktop\code\misc\recipe_recommender\data\labels.json", "w") as f:
    json.dump(label_list, f)

#Apply trained model onto user query, returning list of filter labels
def predict_nutrition_intents(query: str, threshold: float = 0.3):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits).numpy()[0]
    binary_preds = (probs > threshold).astype(int)
    predicted_labels = [label_list[i] for i, val in enumerate(binary_preds) if val == 1]
    return predicted_labels
