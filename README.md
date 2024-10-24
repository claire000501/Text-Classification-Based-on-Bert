# Text-Classification-Based-on-Bert
This project involves training and fine-tuning a BERT model to classify each investor comment.  

1.Training Steps are as following:  
(1)Manually annotate a certain amount of text for training. We plan to classify the text into three categories: "Trust," "Distrust," and "Unknown."  
(2)Load the BERT tokenizer and model, create a parameter grid, and traverse it to determine the optimal parameter combination. Additionally, use data augmentation algorithms to enhance the model's learning effects.  
(3)Train the model based on the optimal parameter combination and evaluate its performance, using metrics such as accuracy, precision, recall, and F1 score.  
(4)Use the trained model to classify the test set and calculate the accuracy.  
