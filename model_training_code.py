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

