# Quora Question Spam Classification using Deep Learning

## 📌 Introduction

This project is a solution to the Deep Learning Expert Course assignment from Edvancer. The task is to build a binary classification model to identify whether a question posted on Quora is spam or not. The dataset and pre-trained GloVe embeddings were provided as part of the assignment resources.

The model was built using TensorFlow and Keras with Conv1D architecture, utilizing GloVe embeddings for word representation and trained on a cleaned and stratified subset of the data.

---
## model_training_code 

```python
# Import required libraries
import pandas as pd
import numpy as np
import re
import os
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPool1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Memory-Efficient Data Loading
def load_data_in_chunks(file_path, chunk_size=50000):
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    return pd.concat([chunk for chunk in chunks])

file_path = r"C:\Users\Mudassir\Desktop\Edvancer Assignment Submission\Python 2\Assignment 2\train.csv"
print("Loading data in chunks...")
df = load_data_in_chunks(file_path)

# 2. Text Preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

print("Cleaning text data...")
df['cleaned_text'] = df['question_text'].apply(clean_text)
del df['question_text']
gc.collect()

# 3. Stratified Sampling - CORRECTED VERSION
sample_size = 100000
print(f"Creating balanced sample of {sample_size} rows...")
df_sample = df.groupby('target', group_keys=False)\
              .apply(lambda x: x.sample(n=min(len(x), int(sample_size * len(x) / len(df)))))\
              .sample(frac=1)\
              .reset_index(drop=True)
del df
gc.collect()

# 4. Train-Validation-Test Split
X = df_sample['cleaned_text']
y = df_sample['target']
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# 5. Tokenization
max_words = 30000
max_len = 50
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

def tokenize_texts(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len)

X_train_seq = tokenize_texts(X_train)
X_val_seq = tokenize_texts(X_val)
X_test_seq = tokenize_texts(X_test)
del X_train, X_val, X_test
gc.collect()

# 6. GloVe Embeddings
def load_glove_embeddings(glove_path):
    embeddings_index = {}
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if len(values) == embedding_dim + 1:
                embeddings_index[word] = np.asarray(values[1:], dtype='float32')
    return embeddings_index

embedding_dim = 100
glove_path = "glove.6B.100d.txt"
print("Loading GloVe embeddings...")
embeddings_index = load_glove_embeddings(glove_path)

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < max_words and word in embeddings_index:
        embedding_matrix[i] = embeddings_index[word]

# 7. Model Architecture
def build_model():
    model = Sequential([
        Embedding(max_words, embedding_dim, weights=[embedding_matrix], 
                 input_length=max_len, trainable=False),
        Conv1D(64, 3, activation='relu'),
        GlobalMaxPool1D(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', Precision(), Recall()])
    return model

model = build_model()
print(model.summary())

# 8. Training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=2),
    ModelCheckpoint('best_model.keras', save_best_only=True)
]

print("Training model...")
history = model.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=8,
    batch_size=128,
    callbacks=callbacks
)

# 9. Evaluation
def load_saved_model():
    if os.path.exists('best_model.keras'):
        return load_model('best_model.keras')
    elif os.path.exists('best_model.h5'):
        return load_model('best_model.h5')
    else:
        raise FileNotFoundError("No trained model found")

print("Evaluating best model...")
best_model = load_saved_model()

def evaluate_in_chunks(model, X, y, chunk_size=10000):
    metrics = []
    for i in range(0, len(X), chunk_size):
        chunk_X = X[i:i+chunk_size]
        chunk_y = y[i:i+chunk_size]
        metrics.append(model.evaluate(chunk_X, chunk_y, verbose=0))
    return np.mean(metrics, axis=0)

test_metrics = evaluate_in_chunks(best_model, X_test_seq, y_test)
print(f"\nTest Accuracy: {test_metrics[1]:.4f}")
print(f"Test Precision: {test_metrics[2]:.4f}")
print(f"Test Recall: {test_metrics[3]:.4f}")

def save_predictions(model, X_seq, texts_series, y_true, filename):
    chunks = []
    texts = texts_series.reset_index(drop=True)
    y_true = y_true.reset_index(drop=True)
    for i in range(0, len(X_seq), 10000):
        chunk_X = X_seq[i:i+10000]
        chunk_pred = model.predict(chunk_X, verbose=0)
        chunks.append(pd.DataFrame({
            'text': texts.iloc[i:i+10000],
            'true_label': y_true.iloc[i:i+10000],
            'predicted_label': (chunk_pred > 0.5).astype(int).flatten(),
            'prediction_prob': chunk_pred.flatten()
        }))
    pd.concat(chunks).to_csv(filename, index=False)

# Call remains the same
save_predictions(best_model, X_test_seq, X_temp.loc[y_test.index], y_test, 'quora_predictions.csv')

# 11. Training History Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
print("\nTraining plots saved to training_history.png")


```
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
