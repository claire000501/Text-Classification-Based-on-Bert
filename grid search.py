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

# 加载数据
file_path = '/Users/clairewang/Desktop/platform/bert/sample1000.xlsx'
data = pd.read_excel(file_path)
data = data.tail(1000)  # 这里假设你只有1000条数据

# 数据增强函数
def augment_text(text):
    words = text.split()
    if len(words) > 3:
        idx = random.randint(0, len(words) - 1)
        words[idx] = random.choice(words)
        return ' '.join(words)
    else:
        return text

# 增加数据量
augmented_texts = []
augmented_labels = []
for i in range(len(data)):
    for _ in range(5):  # 每条数据生成5个增强样本
        augmented_texts.append(augment_text(data['提问'].iloc[i]))
        augmented_labels.append(data['信任程度'].iloc[i])

augmented_data = pd.DataFrame({
    '提问': augmented_texts,
    '信任程度': augmented_labels
})

# 合并原始数据和增强数据
data = pd.concat([data, augmented_data], ignore_index=True)

# 标签编码
label_encoder = LabelEncoder()
data['信任程度编码'] = label_encoder.fit_transform(data['信任程度'])

# 加载BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# 定义一个函数来对文本进行编码
def encode_data(text_list, tokenizer, max_length=128):
    return tokenizer(text_list, padding=True, truncation=True, max_length=max_length)

# 编码文本数据
encoded_data = encode_data(data['提问'].tolist(), tokenizer)

# 确保标签转换为torch.tensor，并转换为长整型
labels = torch.tensor(data['信任程度编码'].values, dtype=torch.long)

# 自定义数据集类
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

# 将数据划分为训练集和测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(data['提问'].tolist(), labels, test_size=0.2, random_state=42)

# 编码训练集和测试集
train_encodings = encode_data(train_texts, tokenizer)
test_encodings = encode_data(test_texts, tokenizer)

# 创建数据集实例
train_dataset = TrustDataset(train_encodings, train_labels)
test_dataset = TrustDataset(test_encodings, test_labels)

# 创建参数网格
param_grid = {
    'num_train_epochs': [3, 5, 10],
    'per_device_train_batch_size': [4, 8, 16],
    'learning_rate': [2e-5, 3e-5, 5e-5],
    'warmup_steps': [100, 200, 500]
}

# 定义一个函数来创建Trainer实例
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

# 定义评估指标
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

# 网格搜索
best_score = float('-inf')
best_params = None

for epochs in param_grid['num_train_epochs']:
    for batch_size in param_grid['per_device_train_batch_size']:
        for lr in param_grid['learning_rate']:
            for warmup in param_grid['warmup_steps']:
                output_dir = f'E:\\trust_classification\\results\\epochs_{epochs}_batch_{batch_size}_lr_{lr}_warmup_{warmup}'
                logging_dir = f'E:\\trust_classification\\logs\\epochs_{epochs}_batch_{batch_size}_lr_{lr}_warmup_{warmup}'
                
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                os.makedirs(output_dir)
                
                if os.path.exists(logging_dir):
                    shutil.rmtree(logging_dir)
                os.makedirs(logging_dir)
                
                trainer = create_trainer(output_dir, logging_dir, epochs, batch_size, lr, warmup)
                
                trainer.train()
                eval_result = trainer.evaluate()
                
                eval_score = eval_result['eval_accuracy']  # 使用准确率作为比较指标
                
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
