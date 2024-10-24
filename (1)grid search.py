import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
import torch
import shutil
import random

# Load data
file_path = '/Desktop/BERT/sample.xlsx'
data = pd.read_excel(file_path)
data = data.tail(xxxx)  #select the required number of text entries

# Data augmentation function
def augment_text(text):
    words = text.split()
    if len(words) > 3:
        idx = random.randint(0, len(words) - 1)
        words[idx] = random.choice(words)
        return ' '.join(words)
    else:
        return text

# Increase data size
augmented_texts = []
augmented_labels = []
for i in range(len(data)):
    for _ in range(5):  # Generate 5 augmented samples for each data point
        augmented_texts.append(augment_text(data['comment'].iloc[i]))
        augmented_labels.append(data['trust classification'].iloc[i])

augmented_data = pd.DataFrame({
    'comment': augmented_texts,
    'trust classification': augmented_labels
})

# Combine original data and augmented data
data = pd.concat([data, augmented_data], ignore_index=True)

# Label encoding
label_encoder = LabelEncoder()
data['trust classification encoding'] = label_encoder.fit_transform(data['trust classification'])

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Define a function to encode the text data
def encode_data(text_list, tokenizer, max_length=128):
    return tokenizer(text_list, padding=True, truncation=True, max_length=max_length)

# Encode the text data
encoded_data = encode_data(data['comment'].tolist(), tokenizer)

# Ensure labels are converted to torch.tensor and of type long
labels = torch.tensor(data['trust classification encoding'].values, dtype=torch.long)

# Custom dataset class
class TrustDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings            
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# Split data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(data['comment'].tolist(), labels, test_size=0.2, random_state=42)

# Encode training and testing sets
train_encodings = encode_data(train_texts, tokenizer)
test_encodings = encode_data(test_texts, tokenizer)

# Create dataset instances
train_dataset = TrustDataset(train_encodings, train_labels)
test_dataset = TrustDataset(test_encodings, test_labels)

# Create parameter grid
param_grid = {
    'num_train_epochs': [3, 5, 10],
    'per_device_train_batch_size': [4, 8, 16],
    'learning_rate': [2e-5, 3e-5, 5e-5],
    'warmup_steps': [100, 200, 500]
}

# Define a function to create a Trainer instance
def create_trainer(output_dir, logging_dir, num_train_epochs, per_device_train_batch_size, learning_rate, warmup_steps):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_dir=logging_dir,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    return trainer

# Define evaluation metrics
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Grid search
best_score = float('-inf')
best_params = None

for epochs in param_grid['num_train_epochs']:
    for batch_size in param_grid['per_device_train_batch_size']:
        for lr in param_grid['learning_rate']:
            for warmup in param_grid['warmup_steps']:
                output_dir = f'/Desktop/BERT/results/epochs_{epochs}_batch_{batch_size}_lr_{lr}_warmup_{warmup}'
                logging_dir = f'/Desktop/BERT/logs/epochs_{epochs}_batch_{batch_size}_lr_{lr}_warmup_{warmup}'

                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                os.makedirs(output_dir)
                
                if os.path.exists(logging_dir):
                    shutil.rmtree(logging_dir)
                os.makedirs(logging_dir)
                
                trainer = create_trainer(output_dir, logging_dir, epochs, batch_size, lr, warmup)
                
                trainer.train()
                eval_result = trainer.evaluate()
                
                eval_score = eval_result['eval_accuracy']  # Use accuracy as the comparison metric
                
                if eval_score > best_score:
                    best_score = eval_score
                    best_params = {
                        'num_train_epochs': epochs,
                        'per_device_train_batch_size': batch_size,
                        'learning_rate': lr,
                        'warmup_steps': warmup
                    }

print("Best score:", best_score)
print("Best params:", best_params)
