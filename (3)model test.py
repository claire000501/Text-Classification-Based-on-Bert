import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# Define the paths for the model and tokenizer
model_dir = '/Desktop/BERT/results/checkpoint-3240'

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# Preprocess input text
def preprocess(text, tokenizer, max_length=128):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return inputs

# Make predictions
def predict(text, model, tokenizer):
    model.eval()
    inputs = preprocess(text, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.item()

# Load data
file_path = '/Desktop/BERT/sample.xlsx'
data = pd.read_excel(file_path)

# Label encoding
label_encoder = LabelEncoder()
data['trust classification encoding'] = label_encoder.fit_transform(data['trust classification'])

# Read the first XX rows of data
first_XX_data = data.head(XX)

# Make predictions for the first XX rows of data
predictions = []
for text in first_XX_data['comment']:
    predicted_label = predict(text, model, tokenizer)
    predictions.append(predicted_label)

# Convert numerical labels back to original category labels
predicted_labels = label_encoder.inverse_transform(predictions)

# Print prediction results and compare with true labels
correct_predictions = 0
total_predictions = len(predictions)

for text, true_label, pred_label in zip(first_XX_data['comment'], first_XX_data['trust classification'], predicted_labels):
    print(f"Text: {text}")
    print(f"True label: {true_label}")
    print(f"Predicted label: {pred_label}")
    print()
    if true_label == pred_label:
        correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_predictions
print(f"Accuracy: {accuracy * 100:.2f}%")
