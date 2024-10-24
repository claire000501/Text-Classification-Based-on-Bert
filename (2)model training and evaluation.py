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
file_path = '/Users/clairewang/Desktop/platform/bert/sample2.xlsx'
data = pd.read_excel(file_path)
data = data.tail(900)  # 这里假设你只有900条数据

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
        augmented_texts.append(augment_text(data['comment'].iloc[i]))
        augmented_labels.append(data['信任程度'].iloc[i])

augmented_data = pd.DataFrame({
    'comment': augmented_texts,
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
encoded_data = encode_data(data['comment'].tolist(), tokenizer)

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
train_texts, test_texts, train_labels, test_labels = train_test_split(data['comment'].tolist(), labels, test_size=0.2, random_state=42)

# 编码训练集和测试集
train_encodings = encode_data(train_texts, tokenizer)
test_encodings = encode_data(test_texts, tokenizer)

# 创建数据集实例
train_dataset = TrustDataset(train_encodings, train_labels)
test_dataset = TrustDataset(test_encodings, test_labels)

# 使用最佳参数设置训练参数
best_params = {
    'num_train_epochs': 3,
    'per_device_train_batch_size': 4,
    'learning_rate': 2e-5,
    'warmup_steps': 200
}

output_dir = '/Users/clairewang/Desktop/platform/bert/best_model2/results'
logging_dir = '/Users/clairewang/Desktop/platform/bert/best_model2/logs'

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

if os.path.exists(logging_dir):
    shutil.rmtree(logging_dir)
os.makedirs(logging_dir)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=best_params['num_train_epochs'],
    per_device_train_batch_size=best_params['per_device_train_batch_size'],
    per_device_eval_batch_size=best_params['per_device_train_batch_size'],
    learning_rate=best_params['learning_rate'],
    warmup_steps=best_params['warmup_steps'],
    weight_decay=0.01,
    logging_dir=logging_dir,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50
)

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

# 创建Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# 训练模型
trainer.train()

# 评估模型
eval_results = trainer.evaluate()

# 打印评估结果
print("Evaluation Results:")
print(f"Accuracy: {eval_results['eval_accuracy']}")
print(f"Precision: {eval_results['eval_precision']}")
print(f"Recall: {eval_results['eval_recall']}")
print(f"F1 Score: {eval_results['eval_f1']}")
