# %%
"""
# Implementation of Deep RNN
"""

# %%
from keras.optimizers import Adam
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing import sequence
import matplotlib.pyplot as plt

# Load the IMDb dataset
max_features = 10000
maxlen = 100
batch_size = 32

print("Loading data...")
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=max_features)
print(len(train_x), "train sequences")
print(len(test_x), "test sequences")

# Pad sequences to a fixed length
print("Pad sequences (samples x time)")
train_x = sequence.pad_sequences(train_x, maxlen=maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=maxlen)

# Create a Deep RNN model with LSTM layers
deep_rnn_model = Sequential()
deep_rnn_model.add(Embedding(max_features, 128))
deep_rnn_model.add(LSTM(64, return_sequences=True))
deep_rnn_model.add(LSTM(64))
deep_rnn_model.add(Dense(1, activation='sigmoid'))

deep_rnn_model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the Deep RNN model
deep_rnn_history = deep_rnn_model.fit(train_x, train_y, batch_size=batch_size, epochs=5, validation_data=(test_x, test_y))

# Plot training curves
def plot_training_curves(history, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{title} - Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'{title} - Accuracy')

    plt.show()

plot_training_curves(deep_rnn_history, 'Deep RNN')

# Inference and prediction
def predict_sentiment(model, text):
    word_index = imdb.get_word_index()
    text = text.lower().split()
    text = [word_index[word] if word in word_index and word_index[word] < max_features else 2 for word in text]
    text = sequence.pad_sequences([text], maxlen=maxlen)
    prediction = model.predict(text)
    return 'Positive' if prediction > 0.5 else 'Negative'

sample_review = "This movie was fantastic and really captivating."
print(f"Sample Review: '{sample_review}'")
print(f"Deep RNN Prediction: {predict_sentiment(deep_rnn_model, sample_review)}")
