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

1. Data Loading in Chunks
```python

chunk_size = 100000
chunks = pd.read_csv("train.csv", chunksize=chunk_size)
df = pd.concat(chunks)
```
Explanation:

Loads the large CSV file (train.csv) in chunks of 100,000 rows using pandas.read_csv with chunksize.

pd.concat() merges the chunks into a single DataFrame df. This approach prevents memory errors when working with large datasets.

2. Text Cleaning Function
```python

def clean_text(text):
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()                      # Convert to lowercase
    return text

df['cleaned_text'] = df['question_text'].astype(str).apply(clean_text)
```
Explanation:

A custom function clean_text removes non-alphabetic characters using regex and converts text to lowercase.

The cleaning is applied to each question using .apply(). The cleaned version is stored in a new column cleaned_text.

3. Stratified Sampling
```python

df_sample = df.groupby('target', group_keys=False).apply(lambda x: x.sample(50000))
df_sample = df_sample.sample(frac=1).reset_index(drop=True)  # Shuffle
```
Explanation:

Samples 50,000 rows each from the spam (target=1) and non-spam (target=0) groups.

This creates a balanced dataset of 100,000 rows, shuffled for randomness.

4. Train-Test Split
```python

X = df_sample['cleaned_text']
y = df_sample['target']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
```
Explanation:

Splits the data into training (70%), validation (15%), and test (15%) sets.

Stratification ensures the class ratio is preserved across all subsets.

5. Tokenization and Padding
```python

tokenizer = Tokenizer(num_words=30000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 50
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')
```
Explanation:

Tokenizer converts text into sequences of integers based on word frequency.

Sequences are padded to a uniform length of 50 using pad_sequences.

Padding ensures input shape consistency for the neural network.

6. Loading Pre-trained GloVe Embeddings
```python

embedding_index = {}
with open('glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coeffs
```
Explanation:

Loads GloVe embeddings into a dictionary embedding_index with words as keys and vectors as values.

These embeddings provide semantic meaning to words learned from a large corpus.

7. Creating the Embedding Matrix
```python

embedding_dim = 100
embedding_matrix = np.zeros((30000, embedding_dim))
word_index = tokenizer.word_index

for word, i in word_index.items():
    if i < 30000:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
```
Explanation:

Constructs a 2D matrix where each row is the embedding vector for a word.

If a word is not found in GloVe, its vector remains as zeros.

8. Model Architecture
```python

model = Sequential()
model.add(Embedding(30000, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False))
model.add(Conv1D(64, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
```
Explanation:

Embedding layer uses pre-trained GloVe vectors and is frozen (non-trainable).

Conv1D layer detects local patterns in text.

GlobalMaxPooling1D reduces output to the most important features.

Dense and Dropout layers help with learning and regularization.

Output Dense layer with sigmoid activation outputs a probability (binary classification).

9. Compilation and Training
```python

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', Precision(), Recall()])

checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True)
earlystop = EarlyStopping(patience=2, restore_best_weights=True)

history = model.fit(X_train_pad, y_train, 
                    epochs=8, batch_size=128, 
                    validation_data=(X_val_pad, y_val), 
                    callbacks=[checkpoint, earlystop])
```
Explanation:

Model is compiled with binary cross-entropy loss and Adam optimizer.

Accuracy, Precision, and Recall are tracked.

ModelCheckpoint saves the best model based on validation loss.

EarlyStopping halts training if no improvement is seen.

10. Model Evaluation in Chunks
```python

model = load_model("best_model.keras", custom_objects={"Precision": Precision, "Recall": Recall})

y_pred_probs = []
y_pred_labels = []
y_true = []

chunk_size = 10000
for i in range(0, len(X_test_pad), chunk_size):
    X_batch = X_test_pad[i:i+chunk_size]
    y_batch = y_test.iloc[i:i+chunk_size]
    probs = model.predict(X_batch)
    preds = (probs > 0.5).astype(int)
    
    y_pred_probs.extend(probs.flatten())
    y_pred_labels.extend(preds.flatten())
    y_true.extend(y_batch)
```
Explanation:

The best model is loaded for evaluation.

Test data is processed in batches to handle memory efficiently.

Predictions and true labels are collected for metric calculation.

11. Saving Predictions to CSV
```python

output_df = pd.DataFrame({
    "text": X_test.values,
    "true_label": y_test.values,
    "predicted_label": y_pred_labels,
    "prediction_prob": y_pred_probs
})
output_df.to_csv("quora_predictions.csv", index=False)
```

Explanation:

A DataFrame is created with actual text, true labels, predictions, and prediction probabilities.

This file is useful for error analysis or reporting.

12. Plotting Training History
    
```python

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")

plt.savefig("training_history.png")
plt.show()

```

Explanation:

Visualizes training and validation accuracy/loss over epochs.

Saved as training_history.png for documentation and insights.



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
