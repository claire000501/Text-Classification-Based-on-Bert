import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# 定义模型和分词器的路径
# model_dir = 'E:\\trust classification\\large_results\\checkpoint-20'
# model_dir = 'E:\\trust classification\\results\\checkpoint-64'
model_dir = '/Users/clairewang/Desktop/platform/bert/best_model/results/checkpoint-3240'

# 加载模型和分词器
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# 预处理输入文本
def preprocess(text, tokenizer, max_length=128):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return inputs

# 进行预测
def predict(text, model, tokenizer):
    model.eval()
    inputs = preprocess(text, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.item()

# 加载数据
file_path = '/Users/clairewang/Desktop/platform/bert/sample1000.xlsx'
data = pd.read_excel(file_path)

# 标签编码
label_encoder = LabelEncoder()
data['信任程度编码'] = label_encoder.fit_transform(data['信任程度'])

# 读取前100条数据
first_100_data = data.head(100)

# 对前100条数据进行预测
predictions = []
for text in first_100_data['提问']:
    predicted_label = predict(text, model, tokenizer)
    predictions.append(predicted_label)


# 将数值标签转换回原始的类别标签
predicted_labels = label_encoder.inverse_transform(predictions)

# 打印预测结果并与原始标签对比
correct_predictions = 0
total_predictions = len(predictions)

for text, true_label, pred_label in zip(first_100_data['提问'], first_100_data['信任程度'], predicted_labels):
    print(f"Text: {text}")
    print(f"True label: {true_label}")
    print(f"Predicted label: {pred_label}")
    print()
    if true_label == pred_label:
        correct_predictions += 1

# 计算正确率
accuracy = correct_predictions / total_predictions
print(f"Accuracy: {accuracy * 100:.2f}%")
