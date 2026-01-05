# train.py
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping


DATA_PATH = "data/IMDB_Dataset.csv"   # same dataset used in notebook
VOCAB_SIZE = 5000
MAX_LEN = 200
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 5

MODEL_PATH = "model.h5"
TOKENIZER_PATH = "tokenizer.pkl"


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "Dataset not found. Place IMDB_Dataset.csv inside data/ folder"
        )

    df = pd.read_csv(DATA_PATH)

    
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

    X = df["review"].values
    y = df["sentiment"].values

    return train_test_split(X, y, test_size=0.2, random_state=42)


def preprocess_text(X_train, X_test):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)

    return X_train_pad, X_test_pad, tokenizer


def build_model():
    model = Sequential()
    model.add(Embedding(input_dim=VOCAB_SIZE,
                        output_dim=EMBEDDING_DIM,
                        input_length=MAX_LEN))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return model


def train():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    print("Preprocessing text...")
    X_train_pad, X_test_pad, tokenizer = preprocess_text(X_train, X_test)

    print("Building model...")
    model = build_model()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    print("Training model...")
    model.fit(
        X_train_pad,
        y_train,
        validation_data=(X_test_pad, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    print("Evaluating model...")
    loss, acc = model.evaluate(X_test_pad, y_test, verbose=0)
    print(f"Test Accuracy: {acc:.4f}")

    print("Saving model and tokenizer...")
    model.save(MODEL_PATH)

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    print("Training completed successfully!")


if __name__ == "__main__":
    train()
