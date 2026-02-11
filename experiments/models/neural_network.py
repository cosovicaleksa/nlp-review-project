import tensorflow as tf
from tensorflow.keras import layers, models 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# num_words (vocab size) - How many top frequent words to keep 
# Maximum review length - df["tokens"].apply(len).describe()
# pass tokenized column
# Fit tokenizer on train tokens
# Transform both train & test tokens

def prepare_nn_data(X_train_df, X_test_df, y_train, y_test, max_len=150, vocab_size=50000):

    tokenizer = Tokenizer(num_words=vocab_size) # Builds a dictionary, converts words to integers.
    tokenizer.fit_on_texts(X_train_df["tokens"])

    train_seq = tokenizer.texts_to_sequences(X_train_df["tokens"])
    test_seq = tokenizer.texts_to_sequences(X_test_df["tokens"])

    X_train_pad = pad_sequences(train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(test_seq, maxlen=max_len)

    y_train_nn = y_train.values - 1
    y_test_nn = y_test.values 

    return X_train_pad, X_test_pad, y_train_nn, y_test_nn, tokenizer

# vocab_size = unique words in vocab, embedding_dim = with how much numbers is each word represented, max len - length of each review
# vocab_size = unique words in vocab, embedding_dim = with how much numbers is each word represented, max len - length of each review
def nn_classifier(vocab_size  =50000, embedding_dim = 128, max_len = 150):
    
    model = models.Sequential([
        layers.Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length = max_len), # out dim (batch_size, max_len, embedding_dim)
       
        layers.GlobalAveragePooling1D(), # out dim (batch_size, embedding_dim) better then flatten bcs of number of params
       
        layers.Dense(128,activation = 'relu'),
        layers.Dropout(0.5),

        layers.Dense(5, activation = 'softmax')
    ])

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    return model


def nn2_classifier(vocab_size=50000, embedding_dim=128, max_len=150):
    model = models.Sequential([
        layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim,input_length=max_len),
        layers.GlobalAveragePooling1D(),

        layers.Dense(256,activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(128,activation="relu"),
        layers.Dropout(0.4),

        layers.Dense(64,activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(5, activation="softmax")
    ])

    model.compile(optimizer = 'adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model



def train_and_evaluate_nns(model, X_train, y_train, X_test, y_test, epochs, batch_size, validation_split = 0.1, callbacks = None):
    history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_split = validation_split, verbose=1, callbacks = callbacks)

    y_preds_probs = model.predict(X_test)
    y_pred = np.argmax(y_preds_probs, axis=1) + 1 # index of the largest probability + 1 - nisam trebao + 1 jer predvijda od 0 do 4 a i z test mi je od 0 do 4

    plot_train_and_val_score(history)

    class_report = classification_report(y_test, y_pred, output_dict=True)
    acc = accuracy_score(y_true = y_test, y_pred = y_pred)

    print(class_report)

    results = {
        'model': model,
        'test_accuracy': acc,
        'class_report': class_report
        }

    return results 

def plot_train_and_val_score(history):
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')

    plt.title('CNN Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()