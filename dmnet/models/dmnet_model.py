"""
Phase 3: DMNet — Hybrid CNN-LSTM Architecture
==============================================

Why CNN?
  Convolutional layers act as local feature extractors.
  They detect patterns (trends, spikes) within short windows
  of the time-series (e.g., sudden glucose spike over 3 months).

Why LSTM?
  Long Short-Term Memory networks capture long-range temporal
  dependencies. They "remember" that a patient had high glucose
  6 months ago while processing the current timestep.

Why Hybrid?
  CNN extracts local temporal patterns first (fast, parameter-efficient).
  LSTM then models the sequence of those patterns over time.
  Together they achieve higher fidelity than either alone —
  matching the DMNet paper's approach.

Architecture:
  Input (batch, 12, 11)
    → Conv1D(64, kernel=3, relu)
    → MaxPooling1D(pool=2)
    → Conv1D(128, kernel=3, relu)
    → LSTM(64, return_sequences=False)
    → Dropout(0.3)
    → Dense(32, relu)
    → Dense(1, sigmoid)   → diabetes probability
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import os


def build_dmnet(
    n_timesteps : int   = 12,
    n_features  : int   = 11,
    lstm_units  : int   = 64,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """
    Build and return the compiled DMNet CNN-LSTM model.

    Args:
        n_timesteps  : sequence length (months)
        n_features   : number of input features
        lstm_units   : hidden size of LSTM layer
        dropout_rate : dropout probability to prevent overfitting

    Returns:
        Compiled Keras model.
    """
    inputs = layers.Input(shape=(n_timesteps, n_features), name="input_sequence")

    # ── CNN Block 1 ────────────────────────────────────────────────────────────
    # kernel_size=3 looks at 3 consecutive timesteps simultaneously
    x = layers.Conv1D(
        filters=64, kernel_size=3, activation="relu", padding="same", name="conv1"
    )(inputs)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)  # halves sequence length

    # ── CNN Block 2 ────────────────────────────────────────────────────────────
    x = layers.Conv1D(
        filters=128, kernel_size=3, activation="relu", padding="same", name="conv2"
    )(x)
    # No pooling here — we want to retain temporal resolution for LSTM

    # ── LSTM Block ─────────────────────────────────────────────────────────────
    # return_sequences=False → outputs only the final hidden state
    x = layers.LSTM(lstm_units, return_sequences=False, name="lstm")(x)

    # ── Regularization ─────────────────────────────────────────────────────────
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    # ── Dense Head ─────────────────────────────────────────────────────────────
    x = layers.Dense(32, activation="relu", name="dense1")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="DMNet")

    # ── Compile ────────────────────────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def get_callbacks(model_save_path: str = "models/dmnet_best.h5") -> list:
    """
    Training callbacks:
      - EarlyStopping    : stops if val_loss doesn't improve for 10 epochs
      - ModelCheckpoint  : saves the best weights automatically
      - ReduceLROnPlateau: halves LR when val_loss plateaus
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    )

    checkpoint = callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )

    return [early_stop, checkpoint, reduce_lr]


if __name__ == "__main__":
    model = build_dmnet()
    model.summary()
