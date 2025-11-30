import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping


def hamming_accuracy(y_true, y_pred):
    # Cast labels to float32 so types match
    y_true = tf.cast(y_true, tf.float32)
    
    # Convert predictions to binary using threshold 0.5
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
    
    # Compare element-wise
    correct = tf.cast(tf.equal(y_true, y_pred_binary), tf.float32)
    
    # Mean over all labels & samples
    return tf.reduce_mean(correct)



# Load dataset
df = pd.read_csv("cleaned_audio_dataset.csv")
df["genre_list"] = df["genres"].apply(lambda x: str(x).split("|"))

drop_cols = ["file", "genres", "song_name", "artist_name", "url", "id"]
X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["genre_list"])
num_classes = y.shape[1]


# 80/10/10 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)


# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# Baseline NN Model
def build_baseline_model():
    model = models.Sequential([
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", hamming_accuracy]
    )
    return model


model = build_baseline_model()

es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)


# Train
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[es],
    verbose=1
)


# Evaluate on Test Set
test_loss, test_accuracy, test_hamming = model.evaluate(X_test, y_test, verbose=0)

print("Baseline Test Loss:", test_loss)
print("Baseline Test Accuracy:", test_accuracy)
print("Baseline Test Per-Label Accuracy:", test_hamming)
