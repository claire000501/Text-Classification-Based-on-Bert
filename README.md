# Text_Classification_Based_on_Bert
This project involves training and fine-tuning a BERT model to classify each investor comment.  

1.Training Steps are as following:  
(1)Manually annotate a certain amount of text for training. We plan to classify the text into three categories: "Trust," "Distrust," and "Unknown."  
(2)Load the BERT tokenizer and model, create a parameter grid, and traverse it to determine the optimal parameter combination. Additionally, use data augmentation algorithms to enhance the model's learning effects.  
(3)Train the model based on the optimal parameter combination and evaluate its performance, using metrics such as accuracy, precision, recall, and F1 score.  
(4)Use the trained model to classify the test set and calculate the accuracy.  

After experimenting with base BERT and other BERT-based variant models, it was found that the base version performed moderately well, likely due to its average performance with Chinese text. Consequently, a Chinese pre-trained model, Chinese-RoBERTa-wwm-ext, was tried, which demonstrated superior classification results for Chinese. By dividing the training and prediction sets in a 1:1 ratio, an accuracy of over 85% was achieved.
