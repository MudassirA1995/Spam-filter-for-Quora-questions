# Quora Question Spam Classification using Deep Learning

## 📌 Introduction

This project is a solution to the Deep Learning Expert Course assignment from Edvancer. The task is to build a binary classification model to identify whether a question posted on Quora is spam or not. The dataset and pre-trained GloVe embeddings were provided as part of the assignment resources.

The model was built using TensorFlow and Keras with Conv1D architecture, utilizing GloVe embeddings for word representation and trained on a cleaned and stratified subset of the data.

---

## 🧠 Code Explanation

### 1. **Data Loading**
The dataset (`train.csv`) is loaded in memory-efficient chunks using pandas. This helps to handle large datasets without running into memory issues.

### 2. **Text Preprocessing**
The question text is cleaned using a regular expression to retain only alphabetic characters and converted to lowercase.

### 3. **Sampling**
A stratified sample of 100,000 rows is taken to ensure class balance and maintain representative distributions of spam and non-spam questions.

### 4. **Train-Validation-Test Split**
The data is split into training (70%), validation (15%), and test (15%) sets using `train_test_split` with stratification to preserve class distribution.

### 5. **Tokenization**
Keras `Tokenizer` is used to convert text into sequences and pad them to a uniform length of 50 tokens. A vocabulary size of 30,000 is used.

### 6. **Pretrained Word Embeddings**
100-dimensional GloVe embeddings (`glove.6B.100d.txt`) are loaded and used to initialize the embedding layer. This layer is set to non-trainable to retain semantic integrity.

### 7. **Model Architecture**
- Embedding Layer (pre-trained GloVe)
- Conv1D Layer (64 filters)
- GlobalMaxPooling
- Dense Layer (32 units)
- Dropout Layer (0.3)
- Output Layer (Sigmoid for binary classification)

The model uses the Adam optimizer and binary cross-entropy loss function, and it is evaluated using Accuracy, Precision, and Recall.

### 8. **Model Training**
- Early stopping and model checkpointing are used.
- Model is trained with a batch size of 128 over 8 epochs.

### 9. **Evaluation**
The best saved model is evaluated in chunks on the test set to avoid memory issues. Metrics are calculated and printed.

### 10. **Predictions**
Predicted results on the test set are saved to a CSV file (`quora_predictions.csv`) containing:
- Cleaned text
- True label
- Predicted label
- Prediction probability

### 11. **Training Visualization**
Accuracy and loss plots are saved to `training_history.png`.

---

## 📊 Model Performance and Results

| Metric         | Value   |
|----------------|---------|
| Test Accuracy  | 0.9476  |
| Test Precision | 0.6217  |
| Test Recall    | 0.4379  |

The model achieves high overall accuracy, indicating good general classification. However, precision and recall for spam detection show that while the model is reasonably good at identifying spam, it may miss some spam instances (moderate recall).

---

## 🚀 Use Cases and Applications

This spam classifier can be used in:
- **Online Q&A Platforms** (e.g., Quora, StackOverflow) to filter out spammy content automatically.
- **Customer Support Forums** to maintain quality by filtering misleading or irrelevant questions.
- **Community Platforms and Forums** for spam moderation and content integrity.

---

## 📁 Prediction File Explanation

The output file `quora_predictions.csv` contains:
- **text**: The cleaned version of the original question.
- **true_label**: The actual class label (0 for non-spam, 1 for spam).
- **predicted_label**: The predicted class label by the model.
- **prediction_prob**: The probability assigned by the model for the question being spam.

This file can be used for post-analysis, manual validation, or building a dashboard to visualize model predictions.

---

## 🧾 Submission Instructions

Please upload the code and README to a GitHub repository and send the repository link to: `lalit.sachan@edvancer.in`.

---

## 🛠 Requirements

Ensure the following libraries are installed:
- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib

You can install them via:

```bash
pip install -r requirements.txt
